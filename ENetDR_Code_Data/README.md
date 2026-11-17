# ENetDR: Code and Dataset

ENetDR performs candidate entity retrieval followed by entity relevance classification
and separate doctor-side and patient-side relation prediction from Chinese medical dialogues.
This package contains the core implementation and extended MIE dataset.

## Directory structure
```text
ENetDR_Code_Data/
  enetdr/
    __init__.py
    retrieval.py       # Rule matching and semantic retrieval
    data.py            # Window data and heterogeneous graph construction
    evidence.py        # Hierarchical attention and retrieval-conditioned updates
    model.py           # Graph encoder and prediction heads
    losses.py          # Focal loss
    trainer.py         # Training, threshold selection and evaluation
    train.py           # Command-line training entry point
    utils.py           # Logging
  finetune/
    build_triplets.py   # Generate retrieval training triplets
    bert_fine_tune.py   # Fine-tune the BGE retriever
  data/
    chunyu_train.json
    chunyu_dev.json
    chunyu_test.json
    example_dict.json
  requirements.txt
  README.md
```

## Environment and pretrained model
Install PyTorch appropriate for your machine and CUDA version, followed by the dependencies:
```bash
python -m pip install -r requirements.txt
```
A CUDA-enabled GPU is recommended for training.Place the complete `BAAI/bge-base-zh-v1.5` pretrained model and tokenizer in
`./bge-base-zh-v1.5/`. 
## 1. Generate retrieval training samples
```bash
python finetune/build_triplets.py
```
Inputs: `data/chunyu_train.json`, `data/example_dict.json`, and `bge-base-zh-v1.5/`.
Output: `data/train_bge_new.jsonl`, with `query`, `pos`, and `neg` fields.
The script uses training data only and overwrites that output if rerun.

## 2. Fine-tune the retriever
```bash
python -c "from pathlib import Path; Path('checkpoints').mkdir(exist_ok=True)"
python finetune/bert_fine_tune.py
```

## 3. Train and evaluate ENetDR
```bash
python -m enetdr.train
```

For available arguments:
```bash
python -m enetdr.train --help
```

## Dataset format
| Split | Dialogues | Windows |
| --- | ---: | ---: |
| Training | 880 | 14,290 |
| Development | 110 | 1,820 |
| Test | 110 | 1,771 |

Each dialogue JSON file is a list of dialogues. Each dialogue is a list of window
objects containing `utterances` and `label`. Utterances are Chinese strings prefixed
with speaker identity; empty strings represent padding. Preserve dialogue/window
order and padding positions. Labels specify the entity category, normalized entity,
and doctor-side and patient-side relations.
`example_dict.json` defines categories, canonical entities, aliases and relation labels.
Preserve the relation-label mapping order when training or loading checkpoints.

Original dialogue source: https://github.com/nlpir2020/MIE-ACL-2020
