import numpy as np
import torch
import os
import json
from sklearn.metrics import roc_auc_score, roc_curve
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_recall_at_k(query_feats: np.ndarray, gallery_feats: np.ndarray,
                        query_labels: np.ndarray, gallery_labels: np.ndarray,
                        ks: List[int] = [1, 5, 10]) -> Dict[int, float]:
    num_query = query_feats.shape[0]
    num_gallery = gallery_feats.shape[0]
    
    sim = query_feats @ gallery_feats.T
    indices = np.argsort(-sim, axis=1)
    
    results = {k: 0.0 for k in ks}
    
    for i in range(num_query):
        q_label = query_labels[i]
        for k in ks:
            top_k_indices = indices[i, :k]
            top_k_labels = gallery_labels[top_k_indices]
            if q_label in top_k_labels:
                results[k] += 1.0
    
    for k in ks:
        results[k] /= num_query
    
    return results


def compute_map_macro(query_feats: np.ndarray, gallery_feats: np.ndarray,
                      query_labels: np.ndarray, gallery_labels: np.ndarray) -> float:
    num_query = query_feats.shape[0]
    num_gallery = gallery_feats.shape[0]
    unique_labels = np.unique(query_labels)
    
    sim = query_feats @ gallery_feats.T
    indices = np.argsort(-sim, axis=1)
    
    aps = []
    for i in range(num_query):
        q_label = query_labels[i]
        relevant = (gallery_labels == q_label).astype(np.float32)
        if relevant.sum() == 0:
            continue
        
        ranked_relevant = relevant[indices[i]]
        cumsum = np.cumsum(ranked_relevant)
        precision = cumsum / (np.arange(num_gallery) + 1)
        ap = (precision * ranked_relevant).sum() / relevant.sum()
        aps.append(ap)
    
    if len(aps) == 0:
        return 0.0
    
    label_aps = {}
    for i, label in enumerate(query_labels):
        if label not in label_aps:
            label_aps[label] = []
        label_aps[label].append(aps[i])
    
    macro_ap = np.mean([np.mean(v) for v in label_aps.values()])
    return macro_ap


def compute_ood_metrics(query_feats: np.ndarray, gallery_feats: np.ndarray,
                        query_labels: np.ndarray, gallery_labels: np.ndarray,
                        seen_classes: List[int]) -> Dict[str, Optional[float]]:
    unseen_classes = [c for c in np.unique(query_labels) if c not in seen_classes]
    
    if len(unseen_classes) == 0:
        return {
            'Recall@1_Seen': None,
            'Recall@1_Unseen': None,
            'AUROC': None,
            'FPR@TPR95': None
        }
    
    num_query = query_feats.shape[0]
    num_gallery = gallery_feats.shape[0]
    
    sim = query_feats @ gallery_feats.T
    
    is_seen = np.array([q in seen_classes for q in query_labels])
    is_unseen = ~is_seen
    
    recall_seen = None
    if is_seen.any():
        seen_indices = np.where(is_seen)[0]
        correct = 0
        for i in seen_indices:
            q_label = query_labels[i]
            best_idx = np.argmax(sim[i])
            if gallery_labels[best_idx] == q_label:
                correct += 1
        recall_seen = correct / len(seen_indices)
    
    recall_unseen = None
    if is_unseen.any():
        unseen_indices = np.where(is_unseen)[0]
        correct = 0
        for i in unseen_indices:
            q_label = query_labels[i]
            best_idx = np.argmax(sim[i])
            if gallery_labels[best_idx] == q_label:
                correct += 1
        recall_unseen = correct / len(unseen_indices)
    
    seen_gallery_mask = np.array([g in seen_classes for g in gallery_labels])
    if seen_gallery_mask.any():
        sim_to_seen = sim[:, seen_gallery_mask]
        max_sim_to_seen = np.max(sim_to_seen, axis=1)
    else:
        max_sim_to_seen = np.zeros(num_query)
    
    y_true = is_unseen.astype(int)
    y_score = -max_sim_to_seen
    
    auroc = None
    fpr95 = None
    
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        tpr95_idx = np.where(tpr >= 0.95)[0]
        if len(tpr95_idx) > 0:
            fpr95 = fpr[tpr95_idx[0]]
        else:
            fpr95 = 1.0
    
    return {
        'Recall@1_Seen': recall_seen,
        'Recall@1_Unseen': recall_unseen,
        'AUROC': auroc,
        'FPR@TPR95': fpr95
    }


def compute_lifelong_metrics(task_maps: List[List[float]]) -> Tuple[float, float, float]:
    num_tasks = len(task_maps)
    if num_tasks <= 1:
        return 0.0, 0.0, task_maps[-1][-1] if task_maps else 0.0
    
    final_task_maps = task_maps[-1]
    
    forgetting = 0.0
    for t in range(num_tasks - 1):
        max_map_t = max(task_maps[t][:t+1]) if t < len(task_maps[t]) else 0
        final_map_t = final_task_maps[t] if t < len(final_task_maps) else 0
        forgetting += max(0, max_map_t - final_map_t)
    forgetting /= (num_tasks - 1)
    
    plasticity = np.mean(final_task_maps) if len(final_task_maps) > 0 else 0.0
    
    overall = plasticity - forgetting
    
    return plasticity, forgetting, overall


def evaluate_retrieval(query_feats: torch.Tensor, gallery_feats: torch.Tensor,
                       query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                       seen_classes: List[int] = None, ks: List[int] = [1, 5, 10]) -> Dict:
    query_feats = query_feats.cpu().numpy() if isinstance(query_feats, torch.Tensor) else query_feats
    gallery_feats = gallery_feats.cpu().numpy() if isinstance(gallery_feats, torch.Tensor) else gallery_feats
    query_labels = query_labels.cpu().numpy() if isinstance(query_labels, torch.Tensor) else query_labels
    gallery_labels = gallery_labels.cpu().numpy() if isinstance(gallery_labels, torch.Tensor) else gallery_labels
    
    query_feats = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    gallery_feats = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
    
    recall_results = compute_recall_at_k(query_feats, gallery_feats, query_labels, gallery_labels, ks)
    map_macro = compute_map_macro(query_feats, gallery_feats, query_labels, gallery_labels)
    
    results = {
        'R@1': recall_results.get(1, 0.0),
        'R@5': recall_results.get(5, 0.0),
        'R@10': recall_results.get(10, 0.0),
        'mAP': map_macro
    }
    
    if seen_classes is not None:
        ood_results = compute_ood_metrics(query_feats, gallery_feats, query_labels, gallery_labels, seen_classes)
        results.update(ood_results)
    
    return results


class MetricsLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history = []
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, 'results.csv')
        self.json_path = os.path.join(save_dir, 'history.json')
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall\n')
    
    def log(self, task_id: int, num_classes: int, metrics: Dict, 
            task_maps: List[List[float]] = None, cnn_top1: float = 0.0, nme_top1: float = 0.0):
        if task_maps is not None:
            plasticity, forgetting, overall = compute_lifelong_metrics(task_maps)
        else:
            plasticity, forgetting, overall = 0.0, 0.0, 0.0
        
        row = {
            'task': task_id,
            'numclass': num_classes,
            'cnn_top1': cnn_top1,
            'nme_top1': nme_top1,
            'R@1': metrics.get('R@1', 0.0),
            'R@5': metrics.get('R@5', 0.0),
            'R@10': metrics.get('R@10', 0.0),
            'mAP': metrics.get('mAP', 0.0),
            'AUROC': metrics.get('AUROC', ''),
            'FPR95': metrics.get('FPR@TPR95', ''),
            'Plasticity': plasticity,
            'Forgetting': forgetting,
            'Overall': overall
        }
        
        self.history.append(row)
        
        with open(self.csv_path, 'a') as f:
            f.write(f"{task_id},{num_classes},{cnn_top1:.4f},{nme_top1:.4f},"
                    f"{row['R@1']:.4f},{row['R@5']:.4f},{row['R@10']:.4f},{row['mAP']:.4f},"
                    f"{row['AUROC'] if row['AUROC'] is not None else ''},"
                    f"{row['FPR95'] if row['FPR95'] is not None else ''},"
                    f"{plasticity:.4f},{forgetting:.4f},{overall:.4f}\n")
        
        import json
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return plasticity, forgetting, overall