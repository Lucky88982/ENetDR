"""Training, validation threshold selection and test evaluation."""
import os
import json
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import FocalLoss
from .utils import logger

class GraphTrainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        self.crit_ent = FocalLoss(alpha=0.6, gamma=2.0).to(self.device)

        self.crit_doc_state = nn.CrossEntropyLoss()
        self.crit_pat_state = nn.CrossEntropyLoss()

        self.best_score = -1.0

    def train(self, epochs):
        for epoch in range(epochs):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                ent_logits, doc_state_logits, pat_state_logits = self.model(batch)

                ent_y = (batch.doc_y != 0).long()   

                if ent_y.numel() == 0:
                    continue
                loss_ent = self.crit_ent(ent_logits, ent_y)

                ent_mask = (ent_y == 1)

                loss_doc = torch.tensor(0.0, device=self.device)
                loss_pat = torch.tensor(0.0, device=self.device)

                if ent_mask.any():
                    
                    doc_y_shift = batch.doc_y[ent_mask] - 1
                    pat_y_shift = batch.pat_y[ent_mask] - 1

                    loss_doc = self.crit_doc_state(doc_state_logits[ent_mask], doc_y_shift)
                    loss_pat = self.crit_pat_state(pat_state_logits[ent_mask], pat_y_shift)

                loss = loss_ent + loss_doc + loss_pat

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ent': f"{loss_ent.item():.4f}",
                    'doc': f"{loss_doc.item():.4f}",
                    'pat': f"{loss_pat.item():.4f}",
                })

            logger.info(f"Epoch {epoch + 1} Evaluation on DEV set:")
            current_score = self.evaluate(self.dev_loader, search_threshold=True)

            if current_score >= self.best_score:

                self.best_score = current_score
                if not os.path.exists(self.args.output_dir):
                    os.makedirs(self.args.output_dir)

                save_path = os.path.join(self.args.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                with open(os.path.join(self.args.output_dir, "best_threshold.json"), "w") as f:
                    json.dump({"ent_th": float(self.args.ent_th), "score": self.best_score}, f)
                logger.info(f">>> New Best Model Saved! Score: {self.best_score:.4f}, Threshold: {self.args.ent_th:.2f}")

        logger.info("\nTraining Finished. Loading best model for FINAL TEST...")

        best_model_path = os.path.join(self.args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
            with open(os.path.join(self.args.output_dir, "best_threshold.json")) as f:
                self.args.ent_th = json.load(f)["ent_th"]
            logger.info("Loaded best model and its validation threshold successfully.")
        else:
            logger.warning("Best model file not found! Testing with LAST model state.")

        logger.info(f"Evaluating on TEST set with optimized threshold: {self.args.ent_th:.2f}")
        self.evaluate(self.test_loader, search_threshold=False)

    def evaluate(self, loader, search_threshold=False):
        self.model.eval()

        all_probs = []
        all_labels = []

        dialogue_stats = defaultdict(lambda: {
            'ent': {'tp': 0, 'fp': 0, 'fn': 0},
            'doc': {'tp': 0, 'fp': 0, 'fn': 0},
            'pat': {'tp': 0, 'fp': 0, 'fn': 0},
            'strict': {'tp': 0, 'fp': 0, 'fn': 0}
        })
        id2ent = {v: k for k, v in loader.dataset.ent2id.items()}
        key2aliases = loader.dataset.matcher.key2aliases
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                ent_logits, doc_state_logits, pat_state_logits = self.model(batch)

                ent_prob = F.softmax(ent_logits, dim=1)[:, 1] 

                if search_threshold:
                    cur_labels = (batch.doc_y != 0).long()
                    all_probs.extend(ent_prob.cpu().tolist())
                    all_labels.extend(cur_labels.cpu().tolist())

                ent_pred = (ent_prob >= self.args.ent_th).long()
                
                d_pred = torch.zeros_like(ent_pred)
                p_pred = torch.zeros_like(ent_pred)

                mask = (ent_pred == 1)
                if mask.any():
                    d_pred[mask] = doc_state_logits[mask].argmax(dim=1) + 1
                    p_pred[mask] = pat_state_logits[mask].argmax(dim=1) + 1

                conflict_mask = (d_pred == 0) & (p_pred == 0) & (ent_pred == 1)
                if conflict_mask.any():
                    ent_pred[conflict_mask] = 0
                    d_pred[conflict_mask] = 0
                    p_pred[conflict_mask] = 0

                d_true = batch.doc_y
                p_true = batch.pat_y
                cand_batch_ids = batch.batch[batch.candidate_mask]
                num_graphs = batch.num_graphs

                for i in range(num_graphs):
                    mask = (cand_batch_ids == i)

                    cur_did = int(batch.dialogue_id[i].item())
                    cur_widx = int(batch.window_idx[i].item())
                    cur_eid = batch.cand_eid[mask]

                    cur_d_pred = d_pred[mask]
                    cur_p_pred = p_pred[mask]
                    cur_d_true = d_true[mask]
                    cur_p_true = p_true[mask]

                    cur_prob = ent_prob[mask]

                    window_text = loader.dataset.text_map.get((cur_did, cur_widx), "").lower()
                    
                    for j in range(len(cur_d_pred)):
                        if cur_d_pred[j].item() != 0: 
                            eid = cur_eid[j].item()
                            cat, ent_name = id2ent[eid]
                            
                            valid_words = key2aliases[(cat, ent_name)]

                            text_matched = any(str(word).lower() in window_text for word in valid_words if word)

                            prob = cur_prob[j].item()

                            if not text_matched and prob < 0.6:
                                cur_d_pred[j] = 0
                                cur_p_pred[j] = 0
                    
                    pred_ent = set([e.item() for e, s in zip(cur_eid, cur_d_pred) if s!=0])
                    pred_doc = set([(e.item(), s.item()) for e, s in zip(cur_eid, cur_d_pred) if s!=0])
                    pred_pat = set([(e.item(), s.item()) for e, s in zip(cur_eid, cur_p_pred) if s!=0])

                    gold_ent = set([e.item() for e, s in zip(cur_eid, cur_d_true) if s!=0])
                    gold_doc = set([(e.item(), s.item()) for e, s in zip(cur_eid, cur_d_true) if s!=0])
                    gold_pat = set([(e.item(), s.item()) for e, s in zip(cur_eid, cur_p_true) if s!=0])

                    pred_strict = set()
                    for j in range(len(cur_d_pred)):
                        if cur_d_pred[j].item() != 0:
                            item = (cur_eid[j].item(), cur_d_pred[j].item(), cur_p_pred[j].item())
                            pred_strict.add(item)
                    
                    gold_strict = set()
                    for j in range(len(cur_d_true)):
                        if cur_d_true[j].item() != 0:
                            item = (cur_eid[j].item(), cur_d_true[j].item(), cur_p_true[j].item())
                            gold_strict.add(item)

                    if not getattr(self.args, 'legacy_candidate_gold', False):
                        gold = loader.dataset.gold_lookup[(cur_did, cur_widx)]
                        
                        pred_ent = {id2ent[e] for e in pred_ent}
                        pred_doc = {(id2ent[e], r) for e, r in pred_doc}
                        pred_pat = {(id2ent[e], r) for e, r in pred_pat}
                        pred_strict = {(id2ent[e], dr, pr) for e, dr, pr in pred_strict}
                        gold_ent = set(gold)
                        gold_doc = {(k, loader.dataset.doc_map.get(v['doc'], loader.dataset.doc_map['未知']))
                                    for k, v in gold.items()}
                        gold_pat = {(k, loader.dataset.pat_map.get(v['pat'], loader.dataset.pat_map['未知']))
                                    for k, v in gold.items()}
                        gold_strict = {(k, loader.dataset.doc_map.get(v['doc'], loader.dataset.doc_map['未知']),
                                        loader.dataset.pat_map.get(v['pat'], loader.dataset.pat_map['未知']))
                                       for k, v in gold.items()}
                    stats = dialogue_stats[cur_did]
                    for t, p_s, g_s in [
                        ('ent', pred_ent, gold_ent),
                        ('doc', pred_doc, gold_doc),
                        ('pat', pred_pat, gold_pat),
                        ('strict', pred_strict, gold_strict)
                    ]:
                        stats[t]['tp'] += len(p_s & g_s)
                        stats[t]['fp'] += len(p_s - g_s)
                        stats[t]['fn'] += len(g_s - p_s)

        if search_threshold and len(all_probs) > 0:
            best_th = 0.3
            best_score = -1.0
            
            thresholds = np.arange(0.2, 0.95, 0.05) 
            np_probs = np.array(all_probs)
            np_labels = np.array(all_labels)
            
            for th in thresholds:
                np_preds = (np_probs >= th).astype(int)
                
                tp = np.sum((np_preds == 1) & (np_labels == 1))
                fp = np.sum((np_preds == 1) & (np_labels == 0))
                fn = np.sum((np_preds == 0) & (np_labels == 1))
                
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)

                f1 = 2 * p * r / (p + r + 1e-9)

                score = f1 

                if score > best_score:
                    best_score = score
                    best_th = th
            
            logger.info(f"[Auto-Threshold] Found Best Threshold: {best_th:.2f} (candidate F1, before lexical filter: {best_score:.4f})")
            self.args.ent_th = float(best_th)
            
            return self.evaluate(loader, search_threshold=False)

        def calc_prf(tp, fp, fn):
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            return p, r, f1

        tasks_list = ['ent', 'doc', 'pat', 'strict']
        win_global = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in tasks_list}
        for _, tasks in dialogue_stats.items():
            for t in tasks_list:
                win_global[t]['tp'] += tasks[t]['tp']
                win_global[t]['fp'] += tasks[t]['fp']
                win_global[t]['fn'] += tasks[t]['fn']

        win_metrics = {t: calc_prf(win_global[t]['tp'], win_global[t]['fp'], win_global[t]['fn']) for t in tasks_list}
        dia_scores = {t: [] for t in tasks_list}
        for _, tasks in dialogue_stats.items():
            for t in tasks_list:
                
                tp, fp, fn = tasks[t]['tp'], tasks[t]['fp'], tasks[t]['fn']
                if tp + fp + fn == 0:
                    dia_scores[t].append((1.0, 1.0, 1.0))
                else:
                    dia_scores[t].append(calc_prf(tp, fp, fn))

        dia_metrics = {}
        for t in tasks_list:
            if len(dia_scores[t]) == 0:
                dia_metrics[t] = (0.0, 0.0, 0.0)
            else:
                dia_metrics[t] = (
                    sum(x[0] for x in dia_scores[t]) / len(dia_scores[t]),
                    sum(x[1] for x in dia_scores[t]) / len(dia_scores[t]),
                    sum(x[2] for x in dia_scores[t]) / len(dia_scores[t]),
                )
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Threshold used: {self.args.ent_th:.2f}")
        logger.info(f"{'Level':<15} | {'Task':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        logger.info("-" * 80)

        p, r, f = win_metrics['ent']
        logger.info(f"{'Window (Micro)':<15} | {'Entity':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = win_metrics['doc']
        logger.info(f"{'':<15} | {'Doctor':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = win_metrics['pat']
        logger.info(f"{'':<15} | {'Patient':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = win_metrics['strict']
        logger.info(f"{'':<15} | {'Strict (All)':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")

        logger.info("-" * 80)

        p, r, f = dia_metrics['ent']
        logger.info(f"{'Dialogue(Macro)':<15} | {'Entity':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = dia_metrics['doc']
        logger.info(f"{'':<15} | {'Doctor':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = dia_metrics['pat']
        logger.info(f"{'':<15} | {'Patient':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        p, r, f = dia_metrics['strict']
        logger.info(f"{'':<15} | {'Strict (All)':<15} | {p:.4f}     | {r:.4f}     | {f:.4f}")
        
        logger.info("=" * 80 + "\n")

        combined_score = (
            0.3 * win_metrics['ent'][2] + 
            0.2 * win_metrics['doc'][2] + 
            0.2 * win_metrics['pat'][2] +
            0.3 * win_metrics['strict'][2]
        )
        
        return combined_score
