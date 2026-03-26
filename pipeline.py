# pipeline.py
import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluatewp, NodeLevelDataset, evaluate_batch
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
import time
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def compute_oscr(unknown_scores, known_scores, known_preds, known_labels):
    """
    计算 OSCR (Open Set Classification Rate)
    :param unknown_scores: 未知类样本的置信度 (list/array)
    :param known_scores: 已知类样本的置信度 (list/array)
    :param known_preds: 已知类样本的预测类别 (list/array)
    :param known_labels: 已知类样本的真实类别 (list/array)
    """
    x1 = np.array(unknown_scores) # Unknown
    x2 = np.array(known_scores)   # Known
    pred = np.array(known_preds)
    labels = np.array(known_labels)

    # 标记预测正确的已知样本 (Correct Classification)
    correct_indices = (pred == labels)
    m_x2 = np.zeros(len(x2))
    m_x2[correct_indices] = 1 

    # 合并 Known 和 Unknown
    y_score = np.concatenate([x1, x2])
    y_true_indicator = np.concatenate([np.zeros(len(x1)), m_x2])

    # 排序
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true_indicator[desc_score_indices]
    
    # 区分 Unknown 和 Known
    # 在排序后的列表中，哪些是原本属于 unknown 的
    is_unknown = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])[desc_score_indices]
    
    # 计算累积分布
    # FP: 接受了 Unknown 样本
    # TP: 接受了 Correct Known 样本
    cum_fp = np.cumsum(is_unknown)
    cum_tp = np.cumsum(y_true_sorted)

    total_unknown = len(x1)
    total_known = len(x2) # OSCR 的分母通常是所有已知样本，不仅仅是正确的

    if total_unknown == 0 or total_known == 0:
        return 0.0

    fpr = cum_fp / total_unknown
    ccr = cum_tp / total_known 

    # 计算曲线下面积
    m_oscr = np.trapz(ccr, fpr)
    return m_oscr

def get_pipeline(args):
    if args.minibatch:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge


def data_prepare(args, dataset):
    torch.cuda.set_device(args.gpu)
    n_cls_so_far = 0
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            print(f'preparing data for task {task}')
            if args.inter_task_edges:
                mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)

def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0 
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args) 
    
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu) if valid else None  
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if not valid:
        life_model_ins = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)

    n_cls_so_far = 0
    history_auc = []
    history_oscr = []

    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task - 1)[1]

        # ---------------- 训练阶段 ----------------
        if task == 0 and valid and args.method in ['safer']:
            life_model_ins.pretrain(args, subgraph, features)

        for epoch in range(epochs):
            life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1, dataset)

        # 任务结束，更新子空间/指纹
        if valid and args.method in ['safer']:
            life_model_ins.update_subspace_for_task(task, subgraph, features, train_ids)

        # ---------------- 测试阶段 ----------------
        acc_mean = []
        
        epoch_known_scores = []   # 已知类样本置信度
        epoch_unknown_scores = [] # 未知类样本置信度
        epoch_known_preds = []    # 已知类样本预测结果
        epoch_known_labels = []   # 已知类样本真实标签

        # 确定测试范围：如果是最后一个任务，只测到 task；否则测到 task + 1 (包含一个未来任务作为 Unknown)
        eval_range_end = task + 2 if task < args.n_tasks - 1 else task + 1
        

        for t in range(eval_range_end):
            # 标记：当前测试集 t 是否属于未来任务（未知类）
            is_unknown_task = (t > task)

            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(
                open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]

            taskid = 0
            current_scores = None # 用于存储当前 batch 每个节点的 score

            if args.method in ['safer']:
                pred_task, structure_score = life_model_ins.predict_task_id(subgraph, features, test_ids, 
                    tasks_seen_so_far=task + 1, 
                    return_details=True
                )
                taskid = pred_task
                
                current_scores = np.full(len(test_ids), structure_score)

            # --- 获取模型分类输出 ---
            output = life_model_ins.getpred(subgraph, features, taskid)
            
            if current_scores is None:
                probs = F.softmax(output, dim=1)
                conf, _ = torch.max(probs, dim=1)
                
                mask = np.zeros(len(features), dtype=bool)
                mask[test_ids] = True
                current_scores = conf.cpu().detach().numpy()[mask]
            mask = np.zeros(len(features), dtype=bool)
            mask[test_ids] = True
            
            preds = output.argmax(dim=1)
            preds_np = preds.cpu().detach().numpy()

            if not is_unknown_task:
                label_offset1, label_offset2 = task_manager.get_label_offset(int(taskid) - 1)[1], \
                                               task_manager.get_label_offset(int(taskid))[1]
                labels_for_acc = labels - label_offset1
            
            if is_unknown_task:
                # [Unknown 样本]
                epoch_unknown_scores.extend(current_scores)
                print(f"T{t:02d} (Unk)|", end="") 
            else:
                # [Known 样本]
                acc = evaluatewp(output, labels_for_acc, test_ids, cls_balance=args.cls_balance,
                                 ids_per_cls=ids_per_cls_test)
                acc_matrix[task][t] = round(acc * 100, 2)
                acc_mean.append(acc)
                print(f"T{t:02d} {acc * 100:.2f}|", end="")
                
               
                epoch_known_scores.extend(current_scores)
                
                #收集 OSCR 需要的预测和标签
                epoch_known_preds.extend(preds_np[mask])
                if isinstance(labels_for_acc, torch.Tensor):
                    local_labels = labels_for_acc.cpu().detach().numpy()
                else:
                    local_labels = labels_for_acc
                epoch_known_labels.extend(local_labels[mask])

        acc_mean_val = round(np.mean(acc_mean) * 100, 2)
        print(f" acc_mean: {acc_mean_val}", end="")

        if len(epoch_unknown_scores) > 0:    
            y_true_auc = np.concatenate([np.ones(len(epoch_known_scores)), np.zeros(len(epoch_unknown_scores))])
            y_scores_auc = np.concatenate([epoch_known_scores, epoch_unknown_scores])
            
            try:
                auroc = roc_auc_score(y_true_auc, y_scores_auc)
                
                oscr = compute_oscr(epoch_unknown_scores, epoch_known_scores, epoch_known_preds, epoch_known_labels)
                
                history_auc.append(auroc)
                history_oscr.append(oscr)
                print(f" | AUC: {auroc*100:.2f} | OSCR: {oscr*100:.2f}")
            except Exception as e:
                print(f" | Metric Err: {e}")
        else:
            print(" (No Open Set Metrics - Last Task)")
        

    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(life_model_ins, f)
                
    print('AP: ', acc_mean_val) 
    if len(history_auc) > 0:
        print('Avg AUC: ', round(np.mean(history_auc) * 100, 2))
        print('Avg OSCR: ', round(np.mean(history_oscr) * 100, 2))

    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    
    return acc_mean_val, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if not valid:
        life_model_ins = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)

    n_cls_so_far = 0
    history_auc = []
    history_oscr = []

    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb')
        )
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task - 1)[1]

        # ---------------- 训练阶段 ----------------
        if task == 0 and valid and args.method in ['safer']:
            life_model_ins.pretrain(args, subgraph, features, batch_size=args.batch_size)

        for epoch in range(epochs):
            life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1, dataset)
            torch.cuda.empty_cache() 

        if valid and args.method in ['safer']:
            life_model_ins.update_subspace_for_task(task, subgraph, features, train_ids)

        acc_mean = []
        epoch_known_scores = []
        epoch_unknown_scores = []
        epoch_known_preds = []
        epoch_known_labels = []
        eval_range_end = task + 2 if task < args.n_tasks - 1 else task + 1

        for t in range(eval_range_end):
            is_unknown_task = (t > task)

            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(
                open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb')
            )
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

            taskid = 0
            current_scores = None

            if args.method in ['safer']:
                pred_task, structure_score = life_model_ins.predict_task_id(
                    subgraph, features, test_ids, 
                    tasks_seen_so_far=task + 1, 
                    return_details=True
                )
                taskid = pred_task
                current_scores = np.full(len(test_ids), structure_score)

            output = life_model_ins.getpred(subgraph, features, taskid)
            
            if current_scores is None:
                probs = F.softmax(output, dim=1)
                conf, _ = torch.max(probs, dim=1)
                mask = np.zeros(len(features), dtype=bool)
                mask[test_ids] = True
                current_scores = conf.cpu().detach().numpy()[mask]

            preds = output.argmax(dim=1)
            preds_np = preds.cpu().detach().numpy()
            mask = np.zeros(len(features), dtype=bool)
            mask[test_ids] = True

            if not is_unknown_task:
                label_offset1 = task_manager.get_label_offset(int(taskid) - 1)[1]
                labels_for_acc = labels - label_offset1
                
                acc = evaluatewp(output, labels_for_acc, test_ids, cls_balance=False, ids_per_cls=ids_per_cls_test)
                acc_matrix[task][t] = round(acc * 100, 2)
                acc_mean.append(acc)
                print(f"T{t:02d} {acc * 100:.2f}|", end="")

                epoch_known_scores.extend(current_scores)
                epoch_known_preds.extend(preds_np[mask])
                if isinstance(labels_for_acc, torch.Tensor):
                    local_labels = labels_for_acc.cpu().detach().numpy()
                else:
                    local_labels = labels_for_acc
                epoch_known_labels.extend(local_labels[mask])

            else:
                epoch_unknown_scores.extend(current_scores)
                print(f"T{t:02d} (Unk)|", end="")

        acc_mean_val = round(np.mean(acc_mean) * 100, 2)
        print(f" acc_mean(ID acc): {acc_mean_val}", end="")        

        if len(epoch_unknown_scores) > 0:
            y_true_auc = np.concatenate([np.ones(len(epoch_known_scores)), np.zeros(len(epoch_unknown_scores))])
            y_scores_auc = np.concatenate([epoch_known_scores, epoch_unknown_scores])
            try:
                auroc = roc_auc_score(y_true_auc, y_scores_auc)
                oscr = compute_oscr(epoch_unknown_scores, epoch_known_scores, epoch_known_preds, epoch_known_labels)
                history_auc.append(auroc)
                history_oscr.append(oscr)
                print(f" | AUC: {auroc*100:.2f} | OSCR: {oscr*100:.2f}")
            except Exception as e:
                print(f" | Metric Err: {e}")
        
        print() 
        
    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(life_model_ins, f)

    print('AP: ', acc_mean_val)
    if len(history_auc) > 0:
        print('Avg AUC: ', round(np.mean(history_auc) * 100, 2))
        print('Avg OSCR: ', round(np.mean(history_oscr) * 100, 2))

    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    
    return acc_mean_val, mean_backward, acc_matrix

    