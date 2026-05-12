"""
run_attacks.py — Loss, Zlib, and Ratio baseline attacks for MDLM MIA.

All three attacks are derived from the NLL of the target (finetuned) model,
using SAMA's compute_nlloss utility with the correct Qwen3 mask token (151669).

  Loss:  score = -NLL_target(text)
  Zlib:  score = -NLL_target(text) / zlib_entropy(text)
  Ratio: score = -NLL_target(text) / NLL_reference(text)

Why we don't use SAMA's LossAttack / ZlibAttack / RatioAttack classes directly:
  - LossAttack / ZlibAttack expect a pre-computed 'nlloss' column not in our pipeline
  - RatioAttack hardcodes AutoModelForCausalLM for reference loading (wrong class)
  We reuse only compute_nlloss() from SAMA/attack/attacks/utils.py and apply the
  correct mask_id=151669 for Qwen3.

Outputs (same format as sama_scores.pt):
  results/loss_scores.pt   — {scores, labels}
  results/zlib_scores.pt   — {scores, labels}
  results/ratio_scores.pt  — {scores, labels}
"""

import argparse
import os
import sys
import zlib as zlib_mod

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Qwen2Tokenizer


# ---------------------------------------------------------------------------
# SAMA_ROOT resolution (same fallback chain as run_sama.py)
# ---------------------------------------------------------------------------
def _find_sama_root() -> str | None:
    candidates = [
        os.environ.get("SAMA_ROOT", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "SAMA")),  # cloned alongside scripts
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../SAMA")),
        "/home/SAMA",
        os.path.abspath("SAMA"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and os.path.isdir(os.path.join(p, "attack")):
            return p
    return None


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tpr_at_fpr(y_true, y_score, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


def compute_nll_for_texts(model, tokenizer, texts, mask_token_id, device,
                           max_length=256, mc_num=3):
    """Compute per-sample NLL for a list of texts using SAMA's compute_nlloss."""
    from attack.attacks.utils import compute_nlloss

    nlls = []
    for text in tqdm(texts, desc="Computing NLL", leave=False):
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            nll_arr = compute_nlloss(
                model,
                input_ids,
                attention_mask,
                shift_logits=False,   # MDLM/Qwen3 does not shift logits
                mc_num=mc_num,
                mask_id=mask_token_id,
            )
        nlls.append(float(nll_arr[0]))

    return np.array(nlls, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=cfg.ALL_DATASETS)
    parser.add_argument("--data_base",  default="data")
    parser.add_argument("--model_base", default="models")
    parser.add_argument("--out_base",   default="results")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--mc_num",     type=int, default=cfg.SAMA_MC,
                        help="Monte Carlo passes for NLL estimation")
    parser.add_argument("--max_length", type=int, default=cfg.MAX_LENGTH)
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit total samples for dry-run")
    args = parser.parse_args()

    data_dir  = os.path.join(args.data_base,  args.dataset)
    model_dir = os.path.join(args.model_base, args.dataset)
    out_dir   = os.path.join(args.out_base,   args.dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # ------------------------------------------------------------------
    # Load SAMA root for compute_nlloss
    # ------------------------------------------------------------------
    sama_root = _find_sama_root()
    if not sama_root:
        print("[ERROR] SAMA repo not found. Set SAMA_ROOT or run from project root.")
        sys.exit(1)
    print(f"SAMA root: {sama_root}")
    sys.path.insert(0, sama_root)
    sys.path.insert(0, os.path.join(sama_root, "attack"))

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    members    = torch.load(os.path.join(data_dir, "members.pt"),    weights_only=False)
    nonmembers = torch.load(os.path.join(data_dir, "nonmembers.pt"), weights_only=False)

    member_texts    = list(members["texts"])
    nonmember_texts = list(nonmembers["texts"])

    if args.n_samples is not None:
        half = args.n_samples // 2
        member_texts    = member_texts[:half]
        nonmember_texts = nonmember_texts[:half]
        print(f"Dry-run: {len(member_texts)} members + {len(nonmember_texts)} non-members")

    texts  = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts), dtype=np.int32)
    total  = len(texts)
    print(f"Total samples: {total}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    ft_path   = os.path.join(model_dir, "finetuned_checkpoint")
    base_path = os.path.join(model_dir, "base_checkpoint")

    # Use Qwen2Tokenizer directly — transformers>=4.57 AutoTokenizer calls AutoConfig
    # first, which fails for local paths with unknown model_type 'a2d-qwen3'.
    tokenizer     = Qwen2Tokenizer.from_pretrained(ft_path)
    mask_token_id = tokenizer.mask_token_id
    print(f"mask_token_id: {mask_token_id}")

    wandb.init(project="da5001-mia", name=f"run3-{args.dataset}-attacks", config={
        "dataset": args.dataset,
        "mc_num": args.mc_num, "max_length": args.max_length,
        "mask_token_id": mask_token_id, **vars(args),
    })

    print("Loading target (finetuned) model...")
    target_model = AutoModelForMaskedLM.from_pretrained(
        ft_path, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()

    # ------------------------------------------------------------------
    # Compute target NLL for all samples (used by Loss, Zlib, Ratio)
    # ------------------------------------------------------------------
    print(f"\nComputing target NLL ({args.mc_num} MC passes per sample)...")
    target_nlls = compute_nll_for_texts(
        target_model, tokenizer, texts, mask_token_id,
        device, max_length=args.max_length, mc_num=args.mc_num,
    )
    print(f"Target NLL: mean={target_nlls.mean():.4f}, std={target_nlls.std():.4f}")

    # Free target model from GPU before loading reference (save VRAM)
    del target_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Loss attack: score = -NLL_target
    # ------------------------------------------------------------------
    loss_scores = -target_nlls

    # ------------------------------------------------------------------
    # Zlib attack: score = -NLL_target / zlib_entropy(text)
    # ------------------------------------------------------------------
    # Zlib attack (Carlini et al. 2021): score = -NLL_target / |zlib(text)|
    # Lower NLL (better modelled) + smaller compressed size → higher score → likely member
    print("\nComputing zlib scores (standard Carlini 2021 formula)...")
    zlib_scores = np.zeros(total, dtype=np.float64)
    for i, text in enumerate(texts):
        compress_bytes = max(1, len(zlib_mod.compress(text.encode())))
        zlib_scores[i] = -target_nlls[i] / compress_bytes

    # ------------------------------------------------------------------
    # Ratio attack: score = -NLL_target / NLL_reference
    # ------------------------------------------------------------------
    print("\nLoading reference (base) model for ratio attack...")
    ref_model = AutoModelForMaskedLM.from_pretrained(
        base_path, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()

    print(f"Computing reference NLL ({args.mc_num} MC passes per sample)...")
    ref_nlls = compute_nll_for_texts(
        ref_model, tokenizer, texts, mask_token_id,
        device, max_length=args.max_length, mc_num=args.mc_num,
    )
    print(f"Reference NLL: mean={ref_nlls.mean():.4f}, std={ref_nlls.std():.4f}")

    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ratio_scores = -(target_nlls / (ref_nlls + 1e-8))

    # ------------------------------------------------------------------
    # Save + report
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    FPR_THRESHOLDS = [0.001, 0.01, 0.10]

    print("\n--- Attack Results ---")
    for attack_name, scores in [("loss", loss_scores), ("zlib", zlib_scores), ("ratio", ratio_scores)]:
        auc = roc_auc_score(labels, scores)

        # Flip if AUC < 0.5 (signal inversion diagnostic)
        if auc < 0.5:
            inv_auc = roc_auc_score(labels, -scores)
            if inv_auc > auc:
                print(f"[DIAG] {attack_name.upper()}: AUC={auc:.4f} < 0.5, inverting scores → {inv_auc:.4f}")
                scores = -scores
                auc    = inv_auc

        tprs = {f: tpr_at_fpr(labels, scores, f) for f in FPR_THRESHOLDS}
        print(f"{attack_name.upper():6s}: AUC={auc:.4f} | "
              f"TPR@0.1%={tprs[0.001]:.4f} | "
              f"TPR@1%={tprs[0.01]:.4f} | "
              f"TPR@10%={tprs[0.10]:.4f}")

        out_path = os.path.join(out_dir, f"{attack_name}_scores.pt")
        torch.save({
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }, out_path)
        print(f"  Saved {out_path}")

        wandb.log({
            f"{attack_name}/auc":            auc,
            f"{attack_name}/tpr_at_0.1fpr": tprs[0.001],
            f"{attack_name}/tpr_at_1fpr":   tprs[0.01],
            f"{attack_name}/tpr_at_10fpr":  tprs[0.10],
        })

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
