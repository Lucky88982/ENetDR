"""Window datasets, token evidence and heterogeneous graph construction."""
import random
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from .retrieval import hybrid_retrieve_parts
from .utils import logger

class EvidenceData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "token_owner":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

class GraphDialogueDataset:
    def __init__(self, data, ontology, retriever, matcher,tokenizer, bert_model,
                 top_k=15, max_len=256, top_k_per_utt=20,inject_gt=False,inject_gt_prob=1.0,inject_gt_until_epoch=-1, use_token_evidence=True, token_cache_size=1024):
        self.use_token_evidence = use_token_evidence
        self.token_cache_size = max(0, int(token_cache_size))
        self.token_cache = OrderedDict()
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.retriever = retriever
        self.top_k = top_k
        self.max_len = max_len
        self.inject_gt = inject_gt
        self.inject_gt_prob = float(inject_gt_prob)
        self.inject_gt_until_epoch = int(inject_gt_until_epoch)
        self.current_epoch = 0  

        orig_doc = ontology.get("医生状态", ["阳性", "阴性", "医生阳性", "医生阴性", "未知"])
        self.doc_map = {'NA': 0}
        for i, s in enumerate(orig_doc):
            self.doc_map[s] = i + 1

        orig_pat = ontology.get("病人状态", ["接受", "拒绝", "犹豫", "询问替代方法", "未知"])
        self.pat_map = {'NA': 0}
        for i, s in enumerate(orig_pat):
            self.pat_map[s] = i + 1
        
        self.ent2id = {}
        eid = 0
        ignore_keys = ["状态", "医生状态", "病人状态", "病人态度"]
        for cat, ent_map in ontology.items():
            if cat in ignore_keys:
                continue
            for ent_name in ent_map.keys():
                key = (cat, ent_name)
                if key not in self.ent2id:
                    self.ent2id[key] = eid
                    eid += 1
        self.matcher = matcher
        self.top_k_per_utt = top_k_per_utt

        self.samples = self._process_data(data)
        self.gold_lookup = {(s['dialogue_id'], s['window_idx']): s['gt_map']
                            for s in self.samples}
        self.cache_utt_embs = {}  
    def _utt_speaker_type(self, u: str) -> int:
        if u is None:
            return 0
        s = str(u).strip().replace("：", ":")
        if s.startswith("患者:"):
            return 0
        if s.startswith("医生:"):
            return 1
        
        return 0
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
    def _normalize_entity(self, ent_str):
        
        return ent_str.strip()
    def _get_logic_features(self, text):
        if not text:
            return torch.zeros(4)

        neg_keywords = ["不", "没", "非", "无", "未", "别", "否"]
        
        query_keywords = ["吗", "？", "?", "什么", "咋", "怎么", "几", "多", "哪"]
        
        affirm_keywords = ["好", "恩", "嗯", "行", "对", "是", "可以", "谢谢", "ok", "OK"]
        
        suggest_keywords = ["建议", "吃点", "查一下", "去", "复查", "注意", "忌", "少吃"]
        
        feat = [0.0, 0.0, 0.0, 0.0]

        if any(k in text for k in neg_keywords): feat[0] = 1.0
        if any(k in text for k in query_keywords): feat[1] = 1.0
        if any(k in text for k in affirm_keywords): feat[2] = 1.0
        if any(k in text for k in suggest_keywords): feat[3] = 1.0
        
        return torch.tensor(feat, dtype=torch.float)
    def _normalize_utt_with_speaker(self, u: str) -> str:
        if u is None:
            return ""
        u = str(u).strip()
        if not u:
            return ""

        u = u.replace("：", ":")

        if u.startswith("患者:"):
            return "[PAT] " + u[len("患者:"):].strip()
        if u.startswith("医生:"):
            return "[DOC] " + u[len("医生:"):].strip()

        return "[UNK] " + u

    def _process_data(self, data):
        processed = []
        dialogue_id = 0
        self.text_map = {}
        def iter_data(d):
            nonlocal dialogue_id
            for item in d:
                if isinstance(item, list):
                    cur_did = dialogue_id
                    dialogue_id += 1
                    for widx, win in enumerate(item):
                        yield cur_did, widx, win
                else:
                    cur_did = dialogue_id
                    dialogue_id += 1
                    yield cur_did, 0, item

        logger.info("Processing data and retrieving candidates...")
        for did, widx, item in tqdm(iter_data(data)):
            utterances = item.get('utterances', [])
            if not utterances:
                continue
            self.text_map[(did, widx)] = "".join(utterances)

            gt_map = {}
            gt_order = []  
            labels = item.get('label', [])
            for label_str in labels:
                try:
                    parts = label_str.replace('：', ':').split('-')
                    if len(parts) < 2:
                        continue
                    head = parts[0]
                    if ':' not in head:
                        continue
                    cat, ent = head.split(':', 1)

                    doc_s = '未知'
                    pat_s = '未知'
                    for p in parts[1:]:
                        if '医生' in p:
                            doc_s = p.split(':', 1)[1]
                        if '患者' in p or '病人' in p:
                            pat_s = p.split(':', 1)[1]

                    key = (cat, ent)
                    gt_map[key] = {'doc': doc_s, 'pat': pat_s}
                    gt_order.append(key)
                except:
                    continue

            forced_idxs, dense_fill, retrieval_features = hybrid_retrieve_parts(
                retriever=self.retriever,
                matcher=self.matcher,
                utterances=utterances,
                top_k_final=self.top_k,
                top_k_per_utt=self.top_k_per_utt, return_features=True
            )

            processed.append({
                'dialogue_id': did,
                'window_idx': widx,
                'utterances': utterances,
                'gt_map': gt_map,
                'gt_order': gt_order,
                'forced_idxs': forced_idxs,
                'dense_fill': dense_fill,
                
                'retrieval_features': {
                    i: retrieval_features[i].clone()
                    for i in (set(forced_idxs) | set(dense_fill) |
                              {self.retriever.key2idx[k] for k in gt_order
                               if k in self.retriever.key2idx})
                },
            })

        return processed
    def _get_utt_emb(self, text):
        if text in self.cache_utt_embs:
            return self.cache_utt_embs[text]
        
        device = self.retriever.device
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            out = self.bert_model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu()[0]
        self.cache_utt_embs[text] = emb
        return emb

    def _get_token_states(self, text):
        
        if text in self.token_cache:
            self.token_cache.move_to_end(text)
            return self.token_cache[text]
        self.bert_model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                    max_length=self.max_len,
                                    return_special_tokens_mask=True)
            special = inputs.pop('special_tokens_mask')[0].bool()
            valid = inputs['attention_mask'][0].bool() & ~special
            
            if not valid.any():
                valid[0] = True
            inputs = {k: v.to(self.retriever.device) for k, v in inputs.items()}
            states = self.bert_model(**inputs).last_hidden_state[0].cpu()
            states = states[valid].to(torch.float16).contiguous()
        if self.token_cache_size:
            self.token_cache[text] = states
            while len(self.token_cache) > self.token_cache_size:
                self.token_cache.popitem(last=False)
        return states

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        utterances = sample['utterances']
        gt_map = sample['gt_map']
        gt_order = sample['gt_order']
        forced_idxs = sample['forced_idxs']
        dense_fill = sample['dense_fill']

        inject_now = False
        if self.inject_gt and (len(gt_map) > 0):
            epoch_ok = (self.inject_gt_until_epoch < 0) or (self.current_epoch < self.inject_gt_until_epoch)
            prob_ok = (random.random() < self.inject_gt_prob)
            inject_now = epoch_ok and prob_ok

        final_idxs = []
        seen = set()

        def push(idx_):
            if idx_ not in seen:
                final_idxs.append(idx_)
                seen.add(idx_)

        if inject_now:
            for key in gt_order:
                if key in self.retriever.key2idx:
                    push(self.retriever.key2idx[key])

        for fi in forced_idxs:
            push(int(fi))

        for di in dense_fill:
            push(int(di))

        if len(final_idxs) > self.top_k:
            final_idxs = final_idxs[:self.top_k]

        candidates_data = []
        for idx_ in final_idxs:
            info, emb = self.retriever.get_candidate_info(int(idx_))
            key = (info['cat'], info['ent'])

            if key in gt_map:
                d_lbl = self.doc_map.get(gt_map[key]['doc'], self.doc_map.get('未知', 0))
                p_lbl = self.pat_map.get(gt_map[key]['pat'], self.pat_map.get('未知', 0))
            else:
                d_lbl = 0
                p_lbl = 0

            candidates_data.append({
                'cat': info['cat'],
                'ent': info['ent'],
                'emb': emb,
                'doc_label': d_lbl,
                'pat_label': p_lbl,
                'is_gt': key in gt_map
            })

        num_u = len(utterances)
        num_c = len(candidates_data)

        u_embs = torch.stack([self._get_utt_emb(self._normalize_utt_with_speaker(u)) for u in utterances])
        c_embs = torch.stack([c['emb'] for c in candidates_data]) if num_c > 0 else torch.zeros(0, self.retriever.candidate_embs.size(1))

        u_logic = torch.stack([self._get_logic_features(u) for u in utterances]) 

        if num_c > 0:
            c_logic = torch.zeros(num_c, 4)

            u_final = torch.cat([u_embs, u_logic], dim=-1)
            c_final = torch.cat([c_embs, c_logic], dim=-1)
            x = torch.cat([u_final, c_final], dim=0)
        else:
            u_final = torch.cat([u_embs, u_logic], dim=-1)
            x = u_final

        edge_src, edge_dst = [], []
        edge_attrs = []  

        for i in range(num_u - 1):
            
            edge_src.append(i); edge_dst.append(i + 1)
            edge_attrs.append([0.0, 1.0]) 

            edge_src.append(i + 1); edge_dst.append(i)
            edge_attrs.append([0.0, 1.0]) 
        utt_types_list = [self._utt_speaker_type(u) for u in utterances]  

        SPEAKER_WINDOW = 3  

        for i in range(num_u):
            links_count = 0  
            for j in range(i + 1, num_u):
                
                if utt_types_list[i] != utt_types_list[j]:
                    continue

                edge_src.append(i); edge_dst.append(j)
                edge_attrs.append([2.0, 1.0]) 

                edge_src.append(j); edge_dst.append(i)
                edge_attrs.append([2.0, 1.0])

                links_count += 1

                if links_count >= SPEAKER_WINDOW:
                    break
        
        if num_c > 0:
            
            u_norm = F.normalize(u_embs, p=2, dim=1)
            c_norm = F.normalize(c_embs, p=2, dim=1)
            sim_matrix = torch.mm(u_norm, c_norm.T)
            
            SIM_THRESHOLD = 0.25 
            c_connected = [False] * num_c

            for i in range(num_u):
                
                u_type = utt_types_list[i] 
                
                interact_type = 3.0 if u_type == 1 else 4.0
                
                for j in range(num_c):
                    score = sim_matrix[i, j].item()
                    if score < SIM_THRESHOLD: continue 

                    u_idx = i
                    c_idx = num_u + j

                    edge_src.append(u_idx); edge_dst.append(c_idx)
                    edge_attrs.append([interact_type, score]) 
                    
                    edge_src.append(c_idx); edge_dst.append(u_idx)
                    edge_attrs.append([interact_type, score]) 
                    
                    c_connected[j] = True

            for j in range(num_c):
                if not c_connected[j]:
                    best_u = torch.argmax(sim_matrix[:, j]).item()
                    best_score = sim_matrix[best_u, j].item()

                    u_type = utt_types_list[best_u]
                    interact_type = 3.0 if u_type == 1 else 4.0

                    u_idx = best_u
                    c_idx = num_u + j
                    
                    edge_src.append(u_idx); edge_dst.append(c_idx)
                    edge_attrs.append([interact_type, best_score])
                    
                    edge_src.append(c_idx); edge_dst.append(u_idx)
                    edge_attrs.append([interact_type, best_score])

        if len(edge_src) == 0:
             
             edge_index = torch.empty((2, 0), dtype=torch.long)
             edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
             edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
             edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        doc_labels = torch.tensor([c['doc_label'] for c in candidates_data], dtype=torch.long)
        pat_labels = torch.tensor([c['pat_label'] for c in candidates_data], dtype=torch.long)

        candidate_mask = torch.zeros(num_u + num_c, dtype=torch.bool)
        candidate_mask[num_u:] = True

        cand_eid = torch.tensor(
            [self.ent2id[(c['cat'], c['ent'])] for c in candidates_data],
            dtype=torch.long
        )

        utt_types = torch.tensor([self._utt_speaker_type(u) for u in utterances], dtype=torch.long)
        cand_types = torch.full((num_c,), 2, dtype=torch.long)
        node_type = torch.cat([utt_types, cand_types], dim=0) 

        retrieval_features = (torch.stack([sample['retrieval_features'][i] for i in final_idxs])
                              if final_idxs else torch.empty(0, 5))
        extras = {}
        if self.use_token_evidence:
            
            states = []
            owners = []
            positions = []
            for uidx, u in enumerate(utterances):
                content = str(u or '').strip().replace('：', ':')
                if content.startswith(('患者:', '医生:')):
                    content = content.split(':', 1)[1].strip()
                h = self._get_token_states(content)
                states.append(h)
                owners.append(torch.full((len(h),), uidx, dtype=torch.long))
                positions.append(torch.arange(len(h)))
            extras = dict(token_states=torch.cat(states), token_owner=torch.cat(owners),
                          token_pos=torch.cat(positions))
        return EvidenceData(
            retrieval_features=retrieval_features,
            **extras,
            x=x,  
            edge_index=edge_index,
            edge_attr=edge_attr,
            doc_y=doc_labels,
            pat_y=pat_labels,
            candidate_mask=candidate_mask,
            num_u=num_u,
            num_c=num_c,
            dialogue_id=sample['dialogue_id'],
            window_idx=sample['window_idx'],
            cand_eid=cand_eid,
            node_type=node_type
        )
