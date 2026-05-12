"""
prepare_data.py — Pull member/non-member texts, tokenize, and save balanced splits.

Run_3 changes vs Run_2:
  - Multi-dataset: accepts --dataset (any of 9: 6 MIMIR + wikitext103 + agnews + xsum)
  - MIMIR domains: 1000 members + 1000 non-members, ngram_13_0.8 split
  - NLP benchmarks: 10000 members + 10000 non-members (Fu et al. 2024 protocol)
  - Outputs go to data/{DATASET}/ instead of data/
  - Sequences shorter than MAX_LENGTH tokens are filtered out (not padded blindly)
  - max_length stays 256

MIMIR dataset structure (iamgroot42/mimir, config=DATASET, split=MIMIR_SPLIT):
  Each row: {member: str, nonmember: str, member_neighbors: list, nonmember_neighbors: list}

NLP benchmark protocol (Fu et al. 2024):
  train split → member texts  (fine-tuned on)
  test  split → non-member texts (never seen during fine-tuning)

Outputs:
  data/{DATASET}/members.pt     — {input_ids [N,256], attention_mask [N,256], texts [N]}
  data/{DATASET}/nonmembers.pt  — same structure

Requires:
  HF_TOKEN env var set (MIMIR is gated at iamgroot42/mimir)
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config


def tokenize_and_filter(texts: list, tokenizer, max_length: int = 256,
                        desc: str = "", min_length: int = 16):
    """
    Tokenize texts, truncate to max_length, keep those with >= min_length tokens.
    For MIMIR domains min_length=max_length (strict). For NLP datasets min_length=16.
    Returns (input_ids [M, L], attention_mask [M, L], kept_texts [M]).
    """
    batch_size = 64
    all_ids, all_mask, kept_texts = [], [], []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"  tokenizing {desc}"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            list(batch),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"]          # [B, L]
        mask = enc["attention_mask"]     # [B, L]

        # Keep sequences with >= min_length real tokens
        long_enough = mask.sum(dim=1) >= min_length
        if long_enough.sum() == 0:
            continue
        ids  = ids[long_enough]
        mask = mask[long_enough]
        kept = [t for t, keep in zip(batch, long_enough.tolist()) if keep]

        all_ids.append(ids)
        all_mask.append(mask)
        kept_texts.extend(kept)

    if not all_ids:
        return torch.zeros(0, max_length, dtype=torch.long), \
               torch.zeros(0, max_length, dtype=torch.long), []

    return (
        torch.cat(all_ids,  dim=0),
        torch.cat(all_mask, dim=0),
        kept_texts,
    )


def load_mimir(dataset_name: str, n: int, hf_token: str):
    """Load member/non-member texts from MIMIR."""
    # Map internal name → HuggingFace config name (e.g. wikipedia → wikipedia_(en))
    hf_name = config.MIMIR_HF_NAME.get(dataset_name, dataset_name)
    print(f"Loading MIMIR '{dataset_name}' (HF config='{hf_name}') split='{config.MIMIR_SPLIT}'...")
    ds = load_dataset(
        config.MIMIR_HF_REPO,
        hf_name,
        split=config.MIMIR_SPLIT,
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"  Total rows: {len(ds)}")
    n_use = min(n, len(ds))
    ds = ds.select(range(n_use))
    member_texts    = list(ds["member"])
    nonmember_texts = list(ds["nonmember"])
    print(f"  Using {n_use} rows → {len(member_texts)} members + {len(nonmember_texts)} non-members")
    return member_texts, nonmember_texts


def load_nlp(dataset_name: str, n: int):
    """Load member/non-member texts for NLP benchmarks (Fu et al. 2024 protocol)."""
    cfg = config.NLP_HF_CONFIGS[dataset_name]
    text_col = cfg["text_col"]

    kwargs = {"path": cfg["path"]}
    if cfg["name"]:
        kwargs["name"] = cfg["name"]

    print(f"Loading NLP dataset '{dataset_name}' ({cfg['path']})...")

    member_ds = load_dataset(**kwargs, split=cfg["member_split"], trust_remote_code=True)
    nonmember_ds = load_dataset(**kwargs, split=cfg["nonmember_split"], trust_remote_code=True)

    # For AG News and XSum, concatenate 'text'/'document' across all labels/splits
    m_texts  = [str(row[text_col]) for row in member_ds    if row[text_col]]
    nm_texts = [str(row[text_col]) for row in nonmember_ds if row[text_col]]

    # Cap to n
    m_texts  = m_texts[:n]
    nm_texts = nm_texts[:n]
    print(f"  Member texts:     {len(m_texts)} (from {cfg['member_split']} split)")
    print(f"  Non-member texts: {len(nm_texts)} (from {cfg['nonmember_split']} split)")
    return m_texts, nm_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,
                        choices=config.ALL_DATASETS,
                        help="Dataset to prepare (one of 9)")
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--out_base",   default="data",
                        help="Base output directory; files go to out_base/DATASET/")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN env var not set. Export it before running.")

    login(token=hf_token, add_to_git_credential=False)

    out_dir = os.path.join(args.out_base, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Determine sample count
    if args.dataset in config.MIMIR_DATASETS:
        n = config.MIMIR_N
    else:
        n = config.NLP_N

    # ----------------------------------------------------------------
    # Load raw texts
    # ----------------------------------------------------------------
    if args.dataset in config.MIMIR_DATASETS:
        member_texts, nonmember_texts = load_mimir(args.dataset, n, hf_token)
    else:
        member_texts, nonmember_texts = load_nlp(args.dataset, n)

    # ----------------------------------------------------------------
    # Load tokenizer
    # ----------------------------------------------------------------
    model_path = os.environ.get("MODEL_PATH", config.MODEL_PATH)
    print(f"\nLoading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ----------------------------------------------------------------
    # Tokenize + filter
    # ----------------------------------------------------------------
    # MIMIR: keep all samples (just truncate+pad to max_length — no filtering).
    # NLP: keep anything >= 16 tokens.
    min_len = 1 if args.dataset in config.MIMIR_DATASETS else 16

    for texts, split_name in [(member_texts, "members"), (nonmember_texts, "nonmembers")]:
        print(f"\nTokenizing {split_name} (max_length={args.max_length}, min_length={min_len})...")
        ids, mask, kept = tokenize_and_filter(texts, tokenizer, args.max_length, split_name, min_length=min_len)
        print(f"  Kept {len(kept)}/{len(texts)} sequences (min_length={min_len})")

        if len(kept) == 0:
            sys.exit(f"ERROR: no {split_name} sequences survived the length filter. "
                     "Check the dataset or lower --max_length.")

        out_path = os.path.join(out_dir, f"{split_name}.pt")
        torch.save({
            "input_ids":      ids,
            "attention_mask": mask,
            "texts":          kept,
        }, out_path)
        print(f"  Saved {out_path} — shape: {ids.shape}")

    print(f"\nDone. Data saved to {out_dir}/")


if __name__ == "__main__":
    main()
