import unittest
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    compute_recall_at_k, compute_map_macro, compute_ood_metrics,
    compute_lifelong_metrics, evaluate_retrieval, MetricsLogger
)


class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
    
    def test_perfect_retrieval(self):
        n = 100
        d = 128
        query_feats = np.random.randn(n, d).astype(np.float32)
        query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
        gallery_feats = query_feats.copy()
        np.random.shuffle(gallery_feats)
        
        query_labels = np.arange(n)
        gallery_labels = np.arange(n)
        np.random.shuffle(gallery_labels)
        
        for i in range(n):
            idx = np.where(gallery_labels == query_labels[i])[0]
            if len(idx) > 0:
                gallery_feats[idx[0]] = query_feats[i]
        
        results = compute_recall_at_k(query_feats, gallery_feats, query_labels, gallery_labels, [1, 5, 10])
        self.assertAlmostEqual(results[1], 1.0, places=2)
        self.assertAlmostEqual(results[5], 1.0, places=2)
        self.assertAlmostEqual(results[10], 1.0, places=2)
        
        map_score = compute_map_macro(query_feats, gallery_feats, query_labels, gallery_labels)
        self.assertAlmostEqual(map_score, 1.0, places=2)
    
    def test_ood_metrics_perfect_separation(self):
        n_seen = 50
        n_unseen = 50
        d = 128
        
        seen_feats = np.random.randn(n_seen, d).astype(np.float32)
        seen_feats = seen_feats / np.linalg.norm(seen_feats, axis=1, keepdims=True)
        
        unseen_feats = -seen_feats[:n_unseen]  # Opposite direction for perfect separation
        unseen_feats = unseen_feats / np.linalg.norm(unseen_feats, axis=1, keepdims=True)
        
        query_feats = np.vstack([seen_feats, unseen_feats])
        gallery_feats = query_feats.copy()
        
        seen_labels = list(range(n_seen))
        unseen_labels = list(range(n_seen, n_seen + n_unseen))
        query_labels = np.array(seen_labels + unseen_labels)
        gallery_labels = query_labels.copy()
        
        seen_classes = seen_labels
        results = compute_ood_metrics(query_feats, gallery_feats, query_labels, gallery_labels, seen_classes)
        
        self.assertAlmostEqual(results['Recall@1_Seen'], 1.0, places=2)
        self.assertAlmostEqual(results['Recall@1_Unseen'], 1.0, places=2)
        self.assertAlmostEqual(results['AUROC'], 1.0, places=2)
        self.assertAlmostEqual(results['FPR@TPR95'], 0.0, places=2)
    
    def test_ood_all_seen(self):
        n = 50
        d = 128
        query_feats = np.random.randn(n, d).astype(np.float32)
        query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
        gallery_feats = query_feats.copy()
        
        query_labels = np.arange(n)
        gallery_labels = np.arange(n)
        
        seen_classes = list(range(n))
        results = compute_ood_metrics(query_feats, gallery_feats, query_labels, gallery_labels, seen_classes)
        
        self.assertIsNone(results['AUROC'])
        self.assertIsNone(results['FPR@TPR95'])
        self.assertIsNone(results['Recall@1_Unseen'])
    
    def test_lifelong_metrics(self):
        task_maps = [
            [0.8],      # Task 0: only task 0
            [0.75, 0.8], # Task 1: task 0 dropped, task 1 good
            [0.7, 0.78, 0.82], # Task 2: task 0 dropped more, task 1 dropped, task 2 good
        ]
        
        plasticity, forgetting, overall = compute_lifelong_metrics(task_maps)
        
        self.assertGreater(plasticity, 0)
        self.assertGreaterEqual(forgetting, 0)
        self.assertAlmostEqual(overall, plasticity - forgetting, places=4)
    
    def test_lifelong_single_task(self):
        task_maps = [[0.85]]
        plasticity, forgetting, overall = compute_lifelong_metrics(task_maps)
        
        self.assertEqual(plasticity, 0.0)
        self.assertEqual(forgetting, 0.0)
        self.assertEqual(overall, 0.85)
    
    def test_evaluate_retrieval(self):
        n = 50
        d = 256
        query_feats = torch.randn(n, d)
        query_feats = F.normalize(query_feats, p=2, dim=1)
        gallery_feats = query_feats.clone()
        
        query_labels = torch.arange(n)
        gallery_labels = torch.arange(n)
        
        results = evaluate_retrieval(query_feats, gallery_feats, query_labels, gallery_labels, 
                                     seen_classes=list(range(25)))
        
        self.assertAlmostEqual(results['R@1'], 1.0, places=2)
        self.assertAlmostEqual(results['mAP'], 1.0, places=2)


if __name__ == '__main__':
    import torch.nn.functional as F
    unittest.main()