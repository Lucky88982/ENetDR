"""Command-line entry point for full-model training."""
import os
import json
import argparse
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch_geometric.loader import DataLoader as GeoDataLoader
from .utils import setup_file_logger
from .retrieval import DenseRetriever, KeywordMatcher
from .data import GraphDialogueDataset
from .model import LLM_GNN_Model
from .trainer import GraphTrainer
from .utils import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/drnet_extensions")  
    
    parser.add_argument("--bert_path", type=str, default="./checkpoints/medical_retriever")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=15, help="Retrieval Top K")
    parser.add_argument("--inject_gt_prob", type=float, default=0.9,
                    help="Probabilistically inject GT during training (0~1).")
    parser.add_argument("--inject_gt_until_epoch", type=int, default=-1,
                    help="Inject GT only when epoch < this value; -1 means always.")
    parser.add_argument("--ent_th", type=float, default=0.3)
    parser.add_argument('--disable_token_evidence', action='store_true')
    parser.add_argument('--disable_retrieval_gate', action='store_true')
    parser.add_argument('--legacy_graph', action='store_true')
    parser.add_argument('--legacy_candidate_gold', action='store_true')
    parser.add_argument('--token_max_len', type=int, default=256)
    parser.add_argument('--token_cache_size', type=int, default=1024)
    parser.add_argument('--edge_chunk_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.top_k < 1 or args.token_max_len < 2:
        parser.error('top_k must be >= 1 and token_max_len >= 2')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    setup_file_logger(args.output_dir)

    with open(os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_dir, "example_dict.json"), "r", encoding="utf-8") as f:
        ontology = json.load(f)

    def load_data(name):
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}, using empty list.")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    train_data = load_data("chunyu_train.json")
    dev_data = load_data("chunyu_dev.json")  
    test_data = load_data("chunyu_test.json")

    if not train_data or not dev_data or not test_data:
        raise ValueError('Train, dev and test data must all be nonempty; check --data_dir.')

    logger.info("Initializing BERT & Retriever...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    bert_model = BertModel.from_pretrained(args.bert_path).to(device)

    bert_model.eval()
    bert_model.requires_grad_(False)
    retriever = DenseRetriever(ontology, bert_model, tokenizer, device)
    matcher = KeywordMatcher(ontology)
    evidence_options = dict(max_len=args.token_max_len,
                            use_token_evidence=not (args.disable_token_evidence or args.legacy_graph),
                            token_cache_size=args.token_cache_size)
    
    logger.info("Building Datasets (Retrieving candidates)...")
    train_ds = GraphDialogueDataset(
    train_data, ontology, retriever, matcher, tokenizer, bert_model,
    top_k=args.top_k, top_k_per_utt=20,
    inject_gt=True,
    inject_gt_prob=args.inject_gt_prob,
    inject_gt_until_epoch=args.inject_gt_until_epoch, **evidence_options
    )

    dev_ds   = GraphDialogueDataset(dev_data, ontology, retriever, matcher, tokenizer, bert_model,
                                    top_k=args.top_k, top_k_per_utt=20, inject_gt=False, **evidence_options)
    test_ds  = GraphDialogueDataset(test_data, ontology, retriever, matcher, tokenizer, bert_model,
                                    top_k=args.top_k, top_k_per_utt=20, inject_gt=False, **evidence_options)

    train_loader = GeoDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = GeoDataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)  
    test_loader = GeoDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    num_doc = len(train_ds.doc_map)
    num_pat = len(train_ds.pat_map)
    model = LLM_GNN_Model(input_dim=bert_model.config.hidden_size,
                          num_doc_cls=num_doc, num_pat_cls=num_pat,
                          use_token_evidence=not args.disable_token_evidence,
                          use_retrieval_gate=not args.disable_retrieval_gate,
                          edge_chunk_size=args.edge_chunk_size,
                          legacy_graph=args.legacy_graph).to(device)

    trainer = GraphTrainer(model, train_loader, dev_loader, test_loader, device, args)
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()
