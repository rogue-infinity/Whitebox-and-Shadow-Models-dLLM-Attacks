"""
run_sama.py — SAMA baseline using official codebase (Stry233/SAMA, MIT licensed).

Credit: SAMA algorithm from "Membership Inference Attacks Against Fine-tuned
Diffusion Language Models" (arXiv 2601.20125, ICLR'26, MIT licensed).
Repository: https://github.com/Stry233/SAMA

Integration strategy:
  SAMA's ModelManager uses AutoModel.from_pretrained, but Qwen3-0.6B MDLM is
  registered as AutoModelForMaskedLM. Fix: monkey-patch ModelManager.init_model
  to use AutoModelForMaskedLM before importing SamaAttack. Mask ID and
  shift_logits are passed directly in the config dict so SAMA never falls back
  to its LLaDA defaults.

SAMA_ROOT resolution (local and JarvisLabs compatible):
  1. SAMA_ROOT env var (set by run_pipeline.sh on JarvisLabs)
  2. ../../../SAMA relative to this file (local Project/SAMA layout)
  3. /home/SAMA (JarvisLabs persistent path after git clone)
  4. ./SAMA in the current working directory

Diagnostic: after scoring, if AUC < 0.5, tries score inversion (1 - score)
and reports whether it helps. Run_1 had AUC=0.37 due to over-memorisation;
with normalised training this should not fire in Run_2.
"""

import argparse
import os
import sys

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Qwen2Tokenizer
import types

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


# ---------------------------------------------------------------------------
# SAMA_ROOT resolution
# ---------------------------------------------------------------------------
def _find_sama_root() -> str | None:
    candidates = [
        os.environ.get("SAMA_ROOT", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "SAMA")),  # cloned alongside scripts
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../SAMA")),
        "/home/SAMA",          # JarvisLabs persistent path
        os.path.abspath("SAMA"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and os.path.isdir(os.path.join(p, "attack")):
            return p
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tpr_at_fpr(y_true, y_score, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, choices=cfg.ALL_DATASETS)
    parser.add_argument("--data_base", default="data")
    parser.add_argument("--model_base",default="models")
    parser.add_argument("--out_base",  default="results")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit total samples for a dry-run")
    parser.add_argument("--T",         type=int, default=cfg.T_STEPS,
                        help="Masking steps")
    parser.add_argument("--n_subsets", type=int, default=cfg.SAMA_N_SUBSETS)
    parser.add_argument("--batch_size_sama", type=int, default=4,
                        help="Batch size passed to SAMA official implementation")
    args = parser.parse_args()

    data_dir  = os.path.join(args.data_base,  args.dataset)
    model_dir = os.path.join(args.model_base, args.dataset)
    out_dir   = os.path.join(args.out_base,   args.dataset)

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    ft_path   = os.path.join(model_dir, "finetuned_checkpoint")
    base_path = os.path.join(model_dir, "base_checkpoint")

    # Load data
    members    = torch.load(os.path.join(data_dir, "members.pt"),    weights_only=True)
    nonmembers = torch.load(os.path.join(data_dir, "nonmembers.pt"), weights_only=True)

    member_texts    = list(members["texts"])
    nonmember_texts = list(nonmembers["texts"])

    n_mem    = members["input_ids"].shape[0]
    n_nonmem = nonmembers["input_ids"].shape[0]
    labels   = np.array([1] * n_mem + [0] * n_nonmem, dtype=np.int32)

    if args.n_samples is not None:
        half = args.n_samples // 2
        member_texts    = member_texts[:min(half, n_mem)]
        nonmember_texts = nonmember_texts[:min(half, n_nonmem)]
        labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts), dtype=np.int32)
        print(f"Dry-run: {len(member_texts)} members + {len(nonmember_texts)} non-members")

    combined_texts  = member_texts + nonmember_texts
    combined_labels = labels.tolist()
    total = len(combined_texts)
    print(f"Total samples: {total}")

    # Pre-load tokenizer to get mask_token_id before SAMA init
    # Use Qwen2Tokenizer directly — transformers>=4.57 AutoTokenizer calls AutoConfig
    # first, which fails for local paths with unknown model_type 'a2d-qwen3'.
    # tokenizer_config.json declares tokenizer_class=Qwen2Tokenizer explicitly.
    print("Loading tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(ft_path)
    mask_token_id = tokenizer.mask_token_id
    print(f"mask_token_id: {mask_token_id}")

    wandb.init(project="da5001-mia", name=f"run3-{args.dataset}-sama", config={
        "dataset": args.dataset,
        "T": args.T, "n_subsets": args.n_subsets,
        "alpha_min": cfg.ALPHA_MIN, "alpha_max": cfg.ALPHA_MAX,
        "mask_token_id": mask_token_id,
        **vars(args),
    })

    # ------------------------------------------------------------------
    # Official SAMA codebase — no fallback
    # ------------------------------------------------------------------
    sama_root = _find_sama_root()
    if not sama_root:
        print("[ERROR] SAMA repo not found at any expected location.")
        print("  Expected: ../../../SAMA (local) or /home/SAMA (JarvisLabs)")
        print("  Set SAMA_ROOT env var to override, or ensure run_pipeline.sh cloned it.")
        sys.exit(1)

    print(f"Found SAMA repo at: {sama_root}")
    sys.path.insert(0, sama_root)
    # sama.py does `from attacks import AbstractAttack` (bare name),
    # which resolves only if SAMA/attack/ is also on sys.path.
    sys.path.insert(0, os.path.join(sama_root, "attack"))

    # Patch (1): inject init_model into attack.run BEFORE importing SamaAttack.
    # sama.py does `from attack.run import init_model` but attack/run.py only
    # defines MIARunner — no module-level init_model exists, causing ImportError.
    # attack/run.py also imports `tabulate` at the top level; if tabulate is
    # missing, `import attack.run` itself raises ImportError before we can inject.
    # Fix: create a stub module in sys.modules under the 'attack.run' key first,
    # then let sama.py's `from attack.run import init_model` pick up the stub.
    def _init_model_patched(model_path, tokenizer_name, device_arg, lora=None):
        # Use Qwen2Tokenizer directly — AutoTokenizer fails for 'a2d-qwen3' in transformers>=4.57
        tok = Qwen2Tokenizer.from_pretrained(model_path)
        mdl = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        ).to(device_arg)
        return mdl, tok, device_arg

    if "attack.run" not in sys.modules:
        _stub = types.ModuleType("attack.run")
        _stub.init_model = _init_model_patched
        sys.modules["attack.run"] = _stub
        print("[COMPAT] Injected stub attack.run with init_model")
    else:
        sys.modules["attack.run"].init_model = _init_model_patched
        print("[COMPAT] Patched existing attack.run.init_model")

    # Patch (2): also patch ModelManager in case it is used directly elsewhere
    import attack.misc.models as _models_mod
    _models_mod.ModelManager.init_model = staticmethod(_init_model_patched)
    print("[COMPAT] Patched ModelManager.init_model → AutoModelForMaskedLM")

    try:
        from attack.attacks.sama import SamaAttack
        from datasets import Dataset as HFDataset
    except Exception as e:
        print(f"[ERROR] Failed to import SamaAttack: {e}")
        sys.exit(1)

    print("Loading target model (finetuned) for SAMA...")
    target_model = AutoModelForMaskedLM.from_pretrained(
        ft_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    sama_config = {
        "reference_model_path": os.path.abspath(base_path),
        "reference_device":     str(device),
        "steps":                args.T,
        "batch_size":           args.batch_size_sama,
        "max_length":           cfg.MAX_LENGTH,
        "subset_size":          cfg.SAMA_M_TOKENS,
        "num_subsets":          args.n_subsets,
        "min_mask_frac":        cfg.ALPHA_MIN,
        "max_mask_frac":        cfg.ALPHA_MAX,
        "l_schedule":           "linear",
        "seed":                 args.seed,
        "save_metadata":        False,
        # Pass mask params directly so SAMA skips get_model_nll_params() defaults
        "model_mask_id":        mask_token_id,
        "model_shift_logits":   False,   # MDLM (Qwen3) does not shift logits
    }

    print("Instantiating SamaAttack (will load reference/base model)...")
    try:
        attack_obj = SamaAttack("sama", target_model, tokenizer, sama_config, device)
    except Exception as e:
        print(f"[ERROR] SamaAttack instantiation failed: {e}")
        sys.exit(1)

    # Post-init fix for ref_mask_id:
    # sama_config["model_mask_id"] correctly sets target_mask_id, but SAMA always
    # determines ref_mask_id via get_model_nll_params(ref_model), which defaults to
    # 126336 (LLaDA) for unknown model types like 'a2d-qwen3'. Both our models use
    # the same Qwen3 tokenizer, so ref_mask_id must equal mask_token_id too.
    if hasattr(attack_obj, "ref_mask_id") and attack_obj.ref_mask_id != mask_token_id:
        print(f"[COMPAT] Correcting ref_mask_id: {attack_obj.ref_mask_id} → {mask_token_id}")
        attack_obj.ref_mask_id = mask_token_id

    # Build HuggingFace Dataset — SAMA expects "text" and "label" columns
    hf_ds = HFDataset.from_dict({"text": combined_texts, "label": combined_labels})

    print(f"Running official SamaAttack on {total} samples...")
    try:
        result_ds = attack_obj.run(hf_ds)
    except Exception as e:
        print(f"[ERROR] SamaAttack.run() failed: {e}")
        sys.exit(1)

    # Score column: SamaAttack adds column named self.name = "sama"
    score_col = None
    for col in ("sama", "score", "sama_score", "membership_score"):
        if col in result_ds.column_names:
            score_col = col
            break
    if score_col is None:
        score_col = result_ds.column_names[-1]
    print(f"Using score column: '{score_col}'")

    scores = np.array(result_ds[score_col], dtype=np.float64)
    print("[OK] Official SAMA run complete.")

    # ------------------------------------------------------------------
    # Diagnostic: detect score inversion
    # ------------------------------------------------------------------
    auc    = roc_auc_score(labels, scores)
    tpr_01 = tpr_at_fpr(labels, scores, 0.001)
    tpr_1  = tpr_at_fpr(labels, scores, 0.01)
    tpr_10 = tpr_at_fpr(labels, scores, 0.10)

    if auc < 0.5:
        inv_auc = roc_auc_score(labels, 1.0 - scores)
        print(f"\n[DIAG] SAMA AUC={auc:.4f} < 0.5 (possible signal inversion).")
        print(f"[DIAG] Inverted AUC={inv_auc:.4f}.", end=" ")
        if inv_auc > auc:
            print("Using 1-score (scores inverted).")
            scores = 1.0 - scores
            auc    = inv_auc
            tpr_01 = tpr_at_fpr(labels, scores, 0.001)
            tpr_1  = tpr_at_fpr(labels, scores, 0.01)
            tpr_10 = tpr_at_fpr(labels, scores, 0.10)
        else:
            print("Inversion did not help — check model training convergence.")
            print("[DIAG] Possible causes: under-training, wrong mask_id, MIMIR split issue.")

    print(f"\nSAMA Results:")
    print(f"  AUC:          {auc:.4f}")
    print(f"  TPR@0.1%FPR:  {tpr_01:.4f}")
    print(f"  TPR@1%FPR:    {tpr_1:.4f}")
    print(f"  TPR@10%FPR:   {tpr_10:.4f}")

    wandb.log({
        "sama/auc":             auc,
        "sama/tpr_at_0.1fpr":  tpr_01,
        "sama/tpr_at_1fpr":    tpr_1,
        "sama/tpr_at_10fpr":   tpr_10,
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sama_scores.pt")
    torch.save({
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }, out_path)
    print(f"Saved {out_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
