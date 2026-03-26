import torch
import copy
import ipdb
import dgl
import dgl.function as fn
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from Backbones.gnns import SGC_Agg
from Baselines.grace import ModelGrace, traingrace, LogReg

class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels)) # 该向量全局共享
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb

class GPFplusAtt(nn.Module):
    """ 解决单一 prompt 表达能力有限，不同节点对 prompt 的需求不同的问题 """
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        # prompt 库：包含 p_num 个长度为 in_channels 的向量
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels)) 
        # 注意力层：根据节点特征 x 计算对每个 prompt 的注意力权重
        self.a = nn.Linear(in_channels, p_num) 
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        # 1. 生成注意力分数 [Batch, p_num]
        score = self.a(x)
        # 2. 归一化权重
        weight = F.softmax(score, dim=1)
        # 3. 加权加总 prompt 库中的向量 [Batch, in_channels]
        p = weight.mm(self.p_list)
        # 4. 将生成的 adaptive prompt 加回到原始特征中
        return x + p

class LocalContrastiveAdapter(nn.Module):
    """
    对 backbone 输出加一个轻量投影头，只在训练中使用，为supervised_contrastive_loss服务
    """
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        z = self.proj(h)
        z = F.normalize(z, p=2, dim=-1) # 映射到单位球面，便于 cosine similarity
        return z

def supervised_contrastive_loss(z: Tensor, y: Tensor, tau: float = 0.2) -> Tensor:
    """ 为了增强类内聚集，拉开类间距离"""
    device = z.device
    B = z.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device)

   
    sim = torch.matmul(z, z.t()) / tau  # cosine similarity + 温度缩放，size:[B, B]
    
    logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)
    sim = sim * logits_mask

    
    y = y.view(-1, 1)
    pos_mask = (y == y.t()).float() * logits_mask  # 正样本掩码，size:[B, B]

    # stabilize
    sim = sim - sim.max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)  # size:[B]

    loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

    valid = (pos_count > 0).float()
    loss = (loss * valid).sum() / (valid.sum() + 1e-12)
    return loss

class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super(NET, self).__init__()
        self.task_manager = task_manager
        self.n_tasks = args.n_tasks
        self.model = model
        self.args = args

        self.drop_edge = args.safer_args["pe"]
        self.drop_feature = args.safer_args["pf"]


        self.anchor_model = copy.deepcopy(model)
        for p in self.anchor_model.parameters():
            p.requires_grad = False
        self.anchor_model.eval()


        self.task_subspaces = {}
        self.pca_rank = 16

        self.ppr_alpha = float(args.safer_args.get("ppr_alpha", 0.15))
        self.ppr_k = int(args.safer_args.get("ppr_k", 10))
        self.ppr_scale = float(args.safer_args.get("ppr_scale", 0.2))
        self._ppr_cache_key = f"_ppr_a{self.ppr_alpha}_k{self.ppr_k}"


        num_promt = int(args.safer_args["prompts"])
        if num_promt < 2:
            prompt = SimplePrompt(args.d_data).cuda()
        else:
            prompt = GPFplusAtt(args.d_data, num_promt).cuda()

        cls_head = LogReg(args.hidden, args.n_cls_per_task).cuda()
        self.classifications = ModuleList([copy.deepcopy(cls_head) for _ in range(args.n_tasks)])
        self.prompts = ModuleList([copy.deepcopy(prompt) for _ in range(args.n_tasks - 1)])

        proj_dim = int(args.safer_args.get("con_proj_dim", 128))
        self.contrastive_adapters = ModuleList(
            [LocalContrastiveAdapter(args.hidden, proj_dim=proj_dim).cuda() for _ in range(args.n_tasks)]
        )
        self.con_lambda = float(args.safer_args.get("con_lambda", 0.1))
        self.con_tau = float(args.safer_args.get("con_tau", 0.2))
        self.con_max_samples = int(args.safer_args.get("con_max_samples", 1024))

        self.optimizers = []
        for taskid in range(args.n_tasks):
            groups = []
            if taskid == 0:
                groups.append({"params": self.classifications[taskid].parameters()})
                groups.append({"params": self.contrastive_adapters[taskid].parameters()})
            else:
                groups.append({"params": self.prompts[taskid - 1].parameters()})
                groups.append({"params": self.classifications[taskid].parameters()})
                groups.append({"params": self.contrastive_adapters[taskid].parameters()})
            self.optimizers.append(torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay))

        self.ce = torch.nn.functional.cross_entropy


    def refresh_anchor(self):
        self.anchor_model.load_state_dict(self.model.state_dict())
        self.anchor_model.eval()

    def _unwrap_backbone_out(self, out):
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _get_anchor_embeddings(self, g, features):
        with torch.no_grad():
            emb = self.anchor_model(g, features)
            emb = self._unwrap_backbone_out(emb)
            emb = F.normalize(emb, p=2, dim=1)
        return emb


    @torch.no_grad()
    def _rw_propagate_fused(self, g, H):

        device = H.device

        with g.local_scope():
            deg = g.out_degrees().to(device).float().clamp(min=1.0)
            inv_deg = (1.0 / deg).unsqueeze(1)

            g.srcdata["_h"] = H * inv_deg
            g.update_all(fn.copy_u("_h", "m"), fn.sum("m", "_hout"))
            H_prop = g.dstdata["_hout"]

        return H_prop

    @torch.no_grad()
    def _ppr_diffuse_features(self, g, x: Tensor, alpha: float, k: int) -> Tensor:
        if hasattr(g, "is_block") and g.is_block:
            k_eff = 1
        else:
            k_eff = k

        H0 = x
        H = x
        for _ in range(k_eff):
            H_prop = self._rw_propagate_fused(g, H)
            H = (1.0 - alpha) * H_prop + alpha * H0
        return H

    @torch.no_grad()
    def _get_ppr_cached(self, g, features: Tensor) -> Tensor:
        cache = getattr(g, "_net_cache", None)
        if cache is None:
            cache = {}
            setattr(g, "_net_cache", cache)

        if self._ppr_cache_key in cache:
            return cache[self._ppr_cache_key]

        x_ppr = self._ppr_diffuse_features(g, features, alpha=self.ppr_alpha, k=self.ppr_k).detach()
        cache[self._ppr_cache_key] = x_ppr
        return x_ppr

    def _build_prompt_input(self, g, features: Tensor) -> Tensor:
        x_ppr = self._get_ppr_cached(g, features)
        return features + self.ppr_scale * x_ppr

    def _rebuild_heads_and_optimizers(self, in_dim: int, device):
        args = getattr(self, "args", None)
        if args is None:
            raise RuntimeError("args not found; please set self.args = args in __init__")

        cls_head = LogReg(in_dim, args.n_cls_per_task).to(device)
        self.classifications = ModuleList([copy.deepcopy(cls_head) for _ in range(args.n_tasks)])

        proj_dim = int(args.safer_args.get("con_proj_dim", 128))
        self.contrastive_adapters = ModuleList(
            [LocalContrastiveAdapter(in_dim, proj_dim=proj_dim).to(device) for _ in range(args.n_tasks)]
        )

        self.optimizers = []
        for taskid in range(args.n_tasks):
            groups = []
            if taskid == 0:
                groups.append({"params": self.classifications[taskid].parameters()})
                groups.append({"params": self.contrastive_adapters[taskid].parameters()})
            else:
                groups.append({"params": self.prompts[taskid - 1].parameters()})
                groups.append({"params": self.classifications[taskid].parameters()})
                groups.append({"params": self.contrastive_adapters[taskid].parameters()})
            self.optimizers.append(torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay))

    def update_subspace_for_task(self, task_id, g, features, train_ids):
        all_emb = self._get_anchor_embeddings(g, features)
        z_train = all_emb[train_ids]

        mean_vec = torch.mean(z_train, dim=0)
        mean_vec = F.normalize(mean_vec, p=2, dim=0)

        z_centered = z_train - mean_vec
        N_samples = z_centered.shape[0]
        actual_rank = min(self.pca_rank, N_samples)

        if actual_rank == 0:
            basis = torch.randn(z_centered.shape[1], self.pca_rank).to(z_centered.device)
            basis = F.normalize(basis, p=2, dim=0)
        else:
            try:
                _, _, Vh = torch.linalg.svd(z_centered, full_matrices=False)
                basis = Vh[:actual_rank, :].T
            except Exception:
                basis = torch.randn(z_centered.shape[1], actual_rank).to(z_centered.device)
                basis = F.normalize(basis, p=2, dim=0)

        self.task_subspaces[task_id] = {"mean": mean_vec.detach(), "basis": basis.detach()}

    def predict_task_id(self, g, features, test_ids, tasks_seen_so_far, return_details=False):
        all_emb = self._get_anchor_embeddings(g, features)
        z_test = all_emb[test_ids]

        z_query = torch.mean(z_test, dim=0)
        z_query = F.normalize(z_query, p=2, dim=0)

        min_score = float("inf")
        predicted_task = 0

        for t in range(tasks_seen_so_far):
            if t not in self.task_subspaces:
                continue
            sub = self.task_subspaces[t]
            mu = sub["mean"]
            U = sub["basis"]

            diff = z_query - mu
            coeff = torch.matmul(U.t(), diff)
            proj = torch.matmul(U, coeff)

            residual = torch.norm(diff - proj).item() ** 2
            distance_sq = torch.norm(diff).item() ** 2
            dist = residual + 1.0 * distance_sq

            if dist < min_score:
                min_score = dist
                predicted_task = t

        if return_details:
            return predicted_task, (-min_score)
        return predicted_task

    def getpred(self, g, features, taskid):
        self.model.eval()

        if taskid == 0:
            out = self._unwrap_backbone_out(self.model(g, features))
            return self.classifications[0](out)

        if taskid - 1 < len(self.prompts):
            prompt_t = self.prompts[taskid - 1]
            x_in = self._build_prompt_input(g, features)
            features = prompt_t.add(x_in)

        out = self._unwrap_backbone_out(self.model(g, features))
        return self.classifications[taskid](out)


    def pretrain(self, args, g, features, batch_size=None):
        self.model.eval()
        with torch.no_grad():
            tmp_out = self._unwrap_backbone_out(self.model(g, features))
            actual_hidden_dim = tmp_out.shape[1]

        num_proj_hidden = 2 * actual_hidden_dim
        gracemodel = ModelGrace(self.model, actual_hidden_dim, num_proj_hidden, tau=0.5).cuda()
        traingrace(
            gracemodel, g, features, batch_size,
            drop_edge_prob=self.drop_edge,
            drop_feature_prob=self.drop_feature,
        )
        self.refresh_anchor()

    def observe_il(self, g, features, labels, t, train_ids, ids_per_cls, offset1, dataset):
        self.model.eval()
        labels = labels - offset1

        cls_head = self.classifications[t]
        con_adapter = self.contrastive_adapters[t]
        cls_head.train()
        con_adapter.train()
        cls_head.zero_grad()
        con_adapter.zero_grad()
        optimizer_t = self.optimizers[t]

        if t > 0:
            prompt_t = self.prompts[t - 1]
            prompt_t.train()
            prompt_t.zero_grad()

            x_in = self._build_prompt_input(g, features)
            features_for_cls = prompt_t.add(x_in)
        else:
            features_for_cls = features

        h_cls = self._unwrap_backbone_out(self.model(g, features_for_cls))

        expected = self.contrastive_adapters[t].proj[0].in_features
        actual = h_cls.size(1)
        if expected != actual:
            self._rebuild_heads_and_optimizers(in_dim=actual, device=h_cls.device)
            cls_head = self.classifications[t]
            con_adapter = self.contrastive_adapters[t]
            optimizer_t = self.optimizers[t]

        z = con_adapter(h_cls)
        logits_cls = cls_head(h_cls)

        if not torch.is_tensor(train_ids):
            idx = torch.as_tensor(train_ids, dtype=torch.long, device=labels.device)
        else:
            idx = train_ids.to(labels.device, dtype=torch.long)

        y_train = labels[idx].long()
        pred_logits = logits_cls[idx]
        z_train = z[idx]

        if len(idx) > self.con_max_samples:
            perm = torch.randperm(len(idx), device=idx.device)[: self.con_max_samples]
            z_loss = z_train[perm]
            y_loss = y_train[perm]
        else:
            z_loss = z_train
            y_loss = y_train
        supcon_loss = supervised_contrastive_loss(z_loss, y_loss, tau=self.con_tau)

        unique, counts = torch.unique(y_train, return_counts=True)
        inv_freq = {}
        for c, cnt in zip(unique.tolist(), counts.tolist()):
            inv_freq[c] = (len(y_train) / (len(unique) * cnt))

        n_cls = pred_logits.size(1)
        weights = torch.ones(n_cls, device=labels.device)
        for c, w in inv_freq.items():
            weights[c] = w
        weights = weights * (n_cls / weights.sum())

        ce_loss = F.cross_entropy(pred_logits, y_train, weight=weights)

        use_focal = getattr(self, "use_focal_loss", True)
        focal_gamma = getattr(self, "focal_gamma", 3.0)
        focal_alpha = getattr(self, "focal_alpha", None)
        if use_focal:
            alpha = weights.detach() if focal_alpha is None else torch.as_tensor(focal_alpha, device=labels.device)
            probs = torch.softmax(pred_logits, dim=1)
            pt = probs.gather(1, y_train.unsqueeze(1)).squeeze(1)
            log_pt = torch.log(pt + 1e-12)
            alpha_t = alpha[y_train] if len(alpha) > int(y_train.max().item()) else 1.0
            focal_term = (1.0 - pt) ** focal_gamma
            focal_loss = (-alpha_t * focal_term * log_pt).mean()
        else:
            focal_loss = torch.tensor(0.0, device=labels.device)

        lambda_focal = getattr(self, "lambda_focal", 1.0)
        loss = ce_loss + lambda_focal * focal_loss + self.con_lambda * supcon_loss

        loss.backward()
        optimizer_t.step()
            
def addedges(subgraph):
    subgraph = copy.deepcopy(subgraph)
    nodedegree = subgraph.in_degrees().cpu()
    isolated_nodes = torch.where(nodedegree == 1)[0]
    connected_nodes = torch.where(nodedegree != 1)[0]
    isolated_nodes = isolated_nodes.numpy()
    connected_nodes = connected_nodes.numpy()
    randomnode = np.random.choice(connected_nodes, isolated_nodes.shape[0])
    srcs = np.concatenate([isolated_nodes, randomnode])
    dsts = np.concatenate([randomnode, isolated_nodes])
    subgraph.add_edges(srcs, dsts)
    return subgraph