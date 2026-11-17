import json

import json
import logging
import os
import math
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sklearn.model_selection import train_test_split
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'object'):
    np.object = object

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def train_retriever_advanced():

    model_name = "./bge-base-zh-v1.5" 
    
    train_data_path = "./data/train_bge_new.jsonl"
    output_path = "./checkpoints/medical_retriever_new"

    batch_size = 64
    num_epochs = 5

    use_hard_negatives = True 

    logger.info(f"Loading SOTA base model: {model_name}...")
    
    try:
        
        model = SentenceTransformer(model_name)
    except TypeError:
        logger.info("Standard loading failed. Building model manually...")
        
        word_embedding_model = models.Transformer(model_name, max_seq_length=512)
        hidden_dim = word_embedding_model.get_word_embedding_dimension()

        pooling_model = models.Pooling(
            hidden_dim,
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False
        )
        
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logger.info("Loading and processing data...")
    all_examples = []
    
    if not os.path.exists(train_data_path):
        logger.error(f"Data file not found: {train_data_path}")
        return

    cnt = 0
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            
            query = data.get('query', '')
            pos = data.get('pos', '')
            neg = data.get('neg', '')

            if use_hard_negatives and neg:
                texts = [query, pos, neg]
            else:
                texts = [query, pos]
                
            all_examples.append(InputExample(texts=texts))
            cnt += 1

    logger.info(f"Total raw examples: {cnt}")
    
    if cnt == 0:
        logger.error("No training data found!")
        return

    train_samples, dev_samples = train_test_split(all_examples, test_size=0.1, random_state=42)
    logger.info(f"Train size: {len(train_samples)}, Dev size: {len(dev_samples)}")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(
        model=model, 
        scale=20.0
    )

    logger.info("Constructing evaluator...")
    
    dev_queries = {}  
    dev_corpus = {}   
    dev_rel_docs = {} 
    
    for idx, ex in enumerate(dev_samples):
        qid = str(idx)
        cid = f"doc_{idx}"
        
        dev_queries[qid] = ex.texts[0] 
        dev_corpus[cid] = ex.texts[1]  
        dev_rel_docs[qid] = {cid}      

        if len(ex.texts) > 2:
             neg_cid = f"neg_{idx}"
             dev_corpus[neg_cid] = ex.texts[2]

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=dev_queries,
        corpus=dev_corpus,
        relevant_docs=dev_rel_docs,
        name='medical-dev',
        show_progress_bar=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10]
    )

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=int(len(train_dataloader) * 0.5), 
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={'lr': 2e-5},
        weight_decay=0.01
    )
    
    logger.info(f"Training finished. Best model saved to {output_path}")

if __name__ == "__main__":
    train_retriever_advanced()

  