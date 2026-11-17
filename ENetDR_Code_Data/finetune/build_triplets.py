import logging

import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import os
from sentence_transformers import SentenceTransformer, models

import json
import random

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class TripletBuilder:
    def __init__(self, ontology_path, data_paths, bert_model_name, output_path):
        self.ontology_path = ontology_path
        self.data_paths = data_paths
        self.bert_model_name = bert_model_name
        self.output_path = output_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

        logging.info(f"Loading base model for hard negative mining: {self.bert_model_name}")

        try:
            
            self.retriever_model = SentenceTransformer(self.bert_model_name, device=self.device)
        except TypeError:
            logging.info("Standard loading failed. Building model manually from components...")

            word_embedding_model = models.Transformer(self.bert_model_name, max_seq_length=512)

            hidden_dim = word_embedding_model.get_word_embedding_dimension()
            pooling_model = models.Pooling(
                hidden_dim,
                pooling_mode_cls_token=True, 
                pooling_mode_mean_tokens=False
            )
            
            self.retriever_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    def _build_candidate_pool(self):
        """
        构建一个包含所有实体及其别名的候选池，用于检索
        """
        candidates = []
        candidate_texts = []
        ignore_keys = ["状态", "医生状态", "病人状态", "病人态度"]

        for cat, ent_map in self.ontology.items():
            if cat in ignore_keys: continue
            for ent_name, aliases in ent_map.items():
                alias_str = " ".join(sorted(set([a.strip() for a in aliases if a and a.strip()])))
                text = f"{cat}:{ent_name} {alias_str}".strip()
                candidates.append((cat, ent_name))
                candidate_texts.append(text)
        
        logging.info(f"Building embeddings for {len(candidates)} candidates...")
        candidate_embs = self.retriever_model.encode(
            candidate_texts, 
            batch_size=128, 
            show_progress_bar=True, 
            convert_to_tensor=True, 
            device=self.device
        )
        
        return candidates, candidate_embs

    def _normalize_utt(self, u):
        """统一 utterance 格式，带说话人标记"""
        if not u: return ""
        u = str(u).strip().replace("：", ":")
        if u.startswith("患者:"): return "[PAT] " + u.split(":", 1)[1].strip()
        if u.startswith("医生:"): return "[DOC] " + u.split(":", 1)[1].strip()
        return "[UNK] " + u

    def build(self, top_k_for_neg=200):
        candidates, candidate_embs = self._build_candidate_pool()
        
        triplets = []
        
        for data_path in self.data_paths:
            logging.info(f"Processing data from: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for dialogue in tqdm(data):
                for window in dialogue:
                    utterances = window.get('utterances', [])
                    labels = window.get('label', [])
                    
                    if not labels: continue

                    context_text = " ".join([self._normalize_utt(u) for u in utterances if u])
                    if not context_text: continue

                    context_emb = self.retriever_model.encode(
                        context_text, 
                        convert_to_tensor=True, 
                        device=self.device
                    )

                    scores = cos_sim(context_emb, candidate_embs)[0]

                    top_k = min(top_k_for_neg, len(candidates))
                    top_results = torch.topk(scores, k=top_k)
                    
                    top_indices = top_results.indices.cpu().tolist()

                    for label_str in labels:
                        try:
                            
                            entity_part = label_str.split('-')[0]
                            cat, ent = entity_part.split(':')
                            positive_key = (cat, ent)

                            pos_aliases = self.ontology.get(cat, {}).get(ent, [])
                            pos_alias_str = " ".join(sorted(set([a.strip() for a in pos_aliases if a and a.strip()])))
                            positive_text = f"{cat}:{ent} {pos_alias_str}".strip()

                            pos_chars = set(ent)

                            hard_negative = None
                            for idx in top_indices:
                                cand_cat, cand_ent = candidates[idx]
                                if (cand_cat, cand_ent) == positive_key:
                                    continue

                                cand_chars = set(cand_ent)
                                overlap_len = len(pos_chars.intersection(cand_chars))
                                
                                if cand_cat == cat and overlap_len >= 2:
                                    continue

                                cand_aliases = self.ontology.get(cand_cat, {}).get(cand_ent, [])
                                cand_alias_str = " ".join(sorted(set([a.strip() for a in cand_aliases if a and a.strip()])))
                                hard_negative = f"{cand_cat}:{cand_ent} {cand_alias_str}".strip()
                                break

                            if not hard_negative:
                                all_cats = list(self.ontology.keys())
                                
                                safe_cats = [c for c in all_cats if c != cat and c not in ["状态", "医生状态", "病人状态", "病人态度"]]
                                if not safe_cats: safe_cats = all_cats
                                
                                rand_cat = random.choice(safe_cats)
                                rand_ent = random.choice(list(self.ontology[rand_cat].keys()))
                                rand_aliases = self.ontology.get(rand_cat, {}).get(rand_ent, [])
                                rand_alias_str = " ".join(sorted(set([a.strip() for a in rand_aliases if a and a.strip()])))
                                hard_negative = f"{rand_cat}:{rand_ent} {rand_alias_str}".strip()

                            instruction = "为这个句子生成表示以用于检索相关文章："
                            query_text = instruction + context_text
                            
                            triplets.append({
                                'query': query_text,
                                'pos': positive_text,
                                'neg': hard_negative
                            })

                        except (ValueError, IndexError):
                            continue

        logging.info(f"Generated {len(triplets)} triplets. Saving to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for t in triplets:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        logging.info("Done.")

if __name__ == "__main__":

    ONTOLOGY_FILE = "./data/example_dict.json"

    DATA_FILES = [
        "./data/chunyu_train.json"
    ]

    BERT_MODEL = "./bge-base-zh-v1.5"

    OUTPUT_FILE = "./data/train_bge_new.jsonl"

    builder = TripletBuilder(
        ontology_path=ONTOLOGY_FILE,
        data_paths=DATA_FILES,
        bert_model_name=BERT_MODEL,
        output_path=OUTPUT_FILE
    )
    builder.build()

   