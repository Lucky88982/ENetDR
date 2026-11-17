from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import re
from .utils import logger

class DenseRetriever:
    def __init__(self, ontology: Dict, bert_model, tokenizer, device, max_len=32):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len  

        self.candidates = []
        self.candidate_texts = []

        ignore_keys = ["状态", "医生状态", "病人状态", "病人态度"]

        for cat, ent_map in ontology.items():
            if cat in ignore_keys:
                continue

            for ent_name, aliases in ent_map.items():
                
                alias_str = ""
                if isinstance(aliases, list) and len(aliases) > 0:
                    
                    aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()] 
                    
                    alias_str = " ".join(sorted(set(aliases)))

                text = f"{cat}:{ent_name} {alias_str}".strip()

                self.candidates.append({'cat': cat, 'ent': ent_name})
                self.candidate_texts.append(text)

        self.key2idx = {}
        for i, c in enumerate(self.candidates):
            self.key2idx[(c["cat"], c["ent"])] = i

        self.candidate_embs = self._precompute_embs()
        logger.info(f"[Retriever] Loaded {len(self.candidates)} standard entities.")

    def _precompute_embs(self):
        self.bert_model.eval()
        all_embs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(self.candidate_texts), batch_size):
                batch_text = self.candidate_texts[i: i + batch_size]
                inputs = self.tokenizer(batch_text, return_tensors='pt', padding=True,
                                        truncation=True, max_length=self.max_len).to(self.device)   
                out = self.bert_model(**inputs)
                
                emb = out.last_hidden_state[:, 0, :]  
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)  

    def search(self, context_text: str, top_k: int = 20, return_all_scores=False):
        """
        输入对话上下文，返回 Top-K 候选实体的索引与相似度分数
        """
        self.bert_model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                context_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            out = self.bert_model(**inputs)
            ctx_emb = out.last_hidden_state[:, 0, :].cpu()  

        ctx = F.normalize(ctx_emb, p=2, dim=1)                 
        cands = F.normalize(self.candidate_embs, p=2, dim=1)   
        scores = torch.matmul(ctx, cands.T).squeeze(0)         

        k = min(top_k, scores.size(0))
        topk_scores, topk_idxs = torch.topk(scores, k)
        if return_all_scores:
            return topk_idxs.numpy(), topk_scores.numpy(), scores
        return topk_idxs.numpy(), topk_scores.numpy()

    def get_candidate_info(self, idx):
        return self.candidates[idx], self.candidate_embs[idx]

class KeywordMatcher:
    def __init__(self, ontology: Dict):
        ignore_keys = ["状态", "医生状态", "病人状态", "病人态度"]

        self.key2aliases = defaultdict(set)

        for cat, ent_map in ontology.items():
            if cat in ignore_keys:
                continue
            for ent_name, aliases in ent_map.items():
                ent_name = str(ent_name).strip()
                if ent_name:
                    self.key2aliases[(cat, ent_name)].add(ent_name)

                if isinstance(aliases, list):
                    for a in aliases:
                        if a is None:
                            continue
                        a = str(a).strip()
                        if a:
                            self.key2aliases[(cat, ent_name)].add(a)

        alias_items = []
        for key, als in self.key2aliases.items():
            for a in als:
                alias_items.append((a, key))
        alias_items.sort(key=lambda x: len(x[0]), reverse=True)

        self.alias_items = alias_items

        self._compiled = [(re.compile(re.escape(a)), key) for a, key in self.alias_items]

    def match_text(self, text: str) -> set:
        if text is None:
            return set()
        t = str(text)
        if not t:
            return set()

        hits = set()
        for pat, key in self._compiled:
            if pat.search(t):
                hits.add(key)
        return hits

    def match_utterances(self, utterances: List[str]) -> set:
        hits = set()
        for u in utterances:
            hits |= self.match_text(u)
        return hits

def build_queries_utterance_level(utterances: List[str]) -> List[str]:
    qs = []
    for u in utterances:
        if u is None:
            continue
        u = str(u).strip()
        if not u:
            continue
        u = u.replace("：", ":")

        if u.startswith("患者:"):
            q = "[PAT] " + u.split(":", 1)[1].strip()
        elif u.startswith("医生:"):
            q = "[DOC] " + u.split(":", 1)[1].strip()
        else:
            q = "[UNK] " + u

        if q:
            qs.append(q)
    return qs

def hybrid_retrieve_parts(
    retriever: DenseRetriever,
    matcher: KeywordMatcher,
    utterances: List[str],
    top_k_final: int,
    top_k_per_utt: int = 20,
    return_features: bool = False
):
    forced_keys = matcher.match_utterances(utterances)
    forced_idxs = []
    for key in sorted(forced_keys):
        if key in retriever.key2idx:
            forced_idxs.append(retriever.key2idx[key])
    
    forced_idxs = list(dict.fromkeys(forced_idxs))
    forced_set = set(forced_idxs)

    queries = build_queries_utterance_level(utterances)
    best = {}  
    
    all_best = torch.full((len(retriever.candidates),), -1.0)
    if queries:
        for q in queries:
            idxs, scores, full_scores = retriever.search(
                q, top_k=top_k_per_utt, return_all_scores=True)
            all_best = torch.maximum(all_best, full_scores.cpu())
            for idx, sc in zip(idxs, scores):
                idx = int(idx)
                sc = float(sc)
                if (idx not in best) or (sc > best[idx]):
                    best[idx] = sc

    dense_sorted = sorted(best.items(), key=lambda x: x[1], reverse=True)

    dense_fill = []
    for idx, _ in dense_sorted:
        if idx in forced_set:
            continue
        dense_fill.append(idx)
        
        if len(dense_fill) >= top_k_final:
            break

    if return_features:
        n = len(retriever.candidates)
        features = torch.zeros(n, 5)
        if n:
            features[:, 2] = all_best.clamp(-1, 1)
            if queries:
                order = torch.argsort(all_best, descending=True, stable=True)
                ranks = torch.empty(n, dtype=torch.float)
                ranks[order] = torch.arange(n, dtype=torch.float)
                features[:, 3] = 1.0 - ranks / max(n - 1, 1)
            if forced_idxs:
                features[forced_idxs, 0] = 1
            if best:
                features[list(best), 1] = 1
            features[:, 4] = features[:, 0] * features[:, 1]
        return forced_idxs, dense_fill, features
    return forced_idxs, dense_fill
