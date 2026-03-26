import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
import ipdb

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(graph, drop_prob):
    graph = copy.deepcopy(graph)
    num_edges = graph.number_of_edges()
    if num_edges == 0:
        return graph
    k = int(drop_prob * num_edges)
    if k <= 0:
        return graph

    edge_delete = np.random.choice(num_edges, k, replace=False)
    src, dst = graph.edges()
    not_equal = src[edge_delete].cpu() != dst[edge_delete].cpu()
    edge_delete = edge_delete[not_equal]
    if len(edge_delete) > 0:
        graph.remove_edges(edge_delete)
    return graph


class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super(ModelGrace, self).__init__()
        self.model = model
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, graph, features):
        output = self.model(graph, features)
        if isinstance(output, tuple):
            output = output[0]
        z = F.elu(self.fc1(output))
        z = self.fc2(z)
        return z

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + 1e-12)
        )

    def batched_semi_loss(self, z1, z2, batch_size):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            diag_part = between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            denom = (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                     + 1e-12)
            losses.append(-torch.log(diag_part / denom))

            torch.cuda.empty_cache()
        return torch.cat(losses)

    def loss(self, h1, h2, batch_size):
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        return ((l1 + l2) * 0.5).mean()


@torch.no_grad()
def degree_aware_seed_sampling(graph, sample_size, uniform_ratio=0.5, pow_=0.5, eps=1.0):
    num_nodes = graph.num_nodes()
    if sample_size >= num_nodes:
        return torch.arange(num_nodes, device=graph.device)

    device = graph.device

    deg = (graph.in_degrees() + graph.out_degrees()).to(device).float()

    k_uniform = int(sample_size * uniform_ratio)
    k_lowdeg = sample_size - k_uniform

    perm = torch.randperm(num_nodes, device=device)
    uniform_nodes = perm[:k_uniform]

    if k_lowdeg <= 0:
        return uniform_nodes

    w = 1.0 / torch.pow(deg + eps, pow_)
    w = w / (w.sum() + 1e-12)


    w2 = w.clone()
    w2[uniform_nodes] = 0.0
    s = w2.sum()
    if s <= 0:
        rest = perm[k_uniform:k_uniform + k_lowdeg]
        return torch.unique(torch.cat([uniform_nodes, rest], dim=0))[:sample_size]

    w2 = w2 / (s + 1e-12)
    lowdeg_nodes = torch.multinomial(w2, k_lowdeg, replacement=False)

    selected = torch.unique(torch.cat([uniform_nodes, lowdeg_nodes], dim=0))

    if selected.numel() < sample_size:
        need = sample_size - selected.numel()
        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask[selected] = False
        remain = torch.nonzero(mask, as_tuple=False).view(-1)
        extra = remain[torch.randperm(remain.numel(), device=device)[:need]]
        selected = torch.cat([selected, extra], dim=0)

    return selected[:sample_size]


def traingrace( modelgrace, graph, features, batch_size=None, drop_edge_prob=0.2, drop_feature_prob=0.3, epochs=200,
                lr=1e-3, sample_size=10000, uniform_ratio=0.5, lowdeg_pow=0.5, lowdeg_eps=1.0):

    modelgrace.train()
    optimizer = torch.optim.Adam(modelgrace.parameters(), lr=lr, weight_decay=1e-5)

    num_nodes = graph.number_of_nodes()
    device = features.device

    for epoch in range(epochs):
        optimizer.zero_grad()
        if num_nodes > sample_size:
            selected_nodes = degree_aware_seed_sampling(
                graph, sample_size=sample_size,
                uniform_ratio=uniform_ratio,
                pow_=lowdeg_pow,
                eps=lowdeg_eps
            ).to(device)
            sub_g = dgl.node_subgraph(graph, selected_nodes)   
            sub_feat = features[selected_nodes]
        else:
            sub_g = graph
            sub_feat = features

        graph_aug = mask_edge(sub_g, drop_edge_prob)
        feat_aug = drop_feature(sub_feat, drop_feature_prob)

        z1 = modelgrace(sub_g, sub_feat)
        z2 = modelgrace(graph_aug, feat_aug)
        loss = modelgrace.loss(z1, z2, batch_size=batch_size)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    modelgrace.eval()