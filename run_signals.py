"""
run_signals.py — Extract white-box MetricsBundle feature vectors for all ~2000 samples.

Run_3: T=4 timesteps, α∈[5%,50%], K=8 mask configs, GRAD_T=2 gradient timesteps.
  - 1000 members + 1000 non-members per MIMIR dataset

For each sample:
  1. extract_metrics() on finetuned model  → feature vector [11*T + 1] = [45 for T=4]
  2. Cross-model cosine sim (finetuned vs base hidden_norms) → scalar → append → [46]

Outputs:
  results/{DS}/X.pt           — feature matrix [N, 46]
  results/{DS}/y.pt           — labels [N]  (1=member, 0=non-member)
  results/{DS}/X_per_group.pt — dict: group_name → tensor slice [N, group_size]
  results/{DS}/signal_names.pt — list of 46 feature name strings
  results/{DS}/raw_signals.pt  — per-sample raw trajectory tensors stacked [N, T] or [N, GRAD_T]
"""

import argparse
import os
import sys

import torch
import wandb
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Qwen2Tokenizer
from tqdm import tqdm

# Import from project
sys.path.insert(0, os.path.dirname(__file__))
from mdlm_metrics_extractor import extract_metrics
import config as cfg

# extract_metrics settings (from central config)
T          = cfg.T_STEPS       # 4
K          = cfg.SIG_K         # 8
GRAD_T     = cfg.SIG_GRAD_T    # 2
ALPHA_MIN  = cfg.ALPHA_MIN     # 0.05
ALPHA_MAX  = cfg.ALPHA_MAX     # 0.50
SEED       = 42
MAX_LENGTH = cfg.MAX_LENGTH    # 256

# Feature group slices: 11 groups × T + 1 (elbo_var) + 1 (cross_model_cos) = 11*T+2 = 46 for T=4
# Note: elbo_var is T+1 index (single scalar), dldt starts at T+1
GROUPS = {
    "elbo_traj":         slice(0,         T),
    "elbo_var":          slice(T,         T + 1),
    "dldt":              slice(T + 1,     2 * T + 1),
    "d2ldt2":            slice(2 * T + 1, 3 * T + 1),
    "pred_entropy":      slice(3 * T + 1, 4 * T + 1),
    "mask_consistency":  slice(4 * T + 1, 5 * T + 1),
    "hidden_norms":      slice(5 * T + 1, 6 * T + 1),
    "hidden_cosine":     slice(6 * T + 1, 7 * T + 1),
    "attn_entropy":      slice(7 * T + 1, 8 * T + 1),
    "attn_crosslayer":   slice(8 * T + 1, 9 * T + 1),
    "attn_barycenter":   slice(9 * T + 1, 10 * T + 1),
    "attn_perturbation": slice(10 * T + 1, 11 * T + 1),
    "cross_model_cos":   slice(11 * T + 1, 11 * T + 2),
}


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_eager(path: str, dtype, device):
    model = AutoModelForMaskedLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    return model


def cross_model_cosine_sim(ft_bundle, base_bundle) -> float:
    """Cosine similarity between mean hidden norms of finetuned vs base model."""
    hn_ft   = ft_bundle.hidden_norms
    hn_base = base_bundle.hidden_norms

    if hn_ft is None or hn_base is None:
        return 0.0

    v_ft   = hn_ft.mean(dim=0)
    v_base = hn_base.mean(dim=0)

    min_layers = min(v_ft.shape[0], v_base.shape[0])
    v_ft   = v_ft[:min_layers].float()
    v_base = v_base[:min_layers].float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v_ft.unsqueeze(0), v_base.unsqueeze(0)
    ).item()
    return cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=cfg.ALL_DATASETS)
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit samples per class for dry-run")
    parser.add_argument("--data_base",  default="data")
    parser.add_argument("--model_base", default="models")
    parser.add_argument("--out_base",   default="results")
    args = parser.parse_args()

    data_dir  = os.path.join(args.data_base,  args.dataset)
    model_dir = os.path.join(args.model_base, args.dataset)
    out_dir   = os.path.join(args.out_base,   args.dataset)

    wandb.init(project="da5001-mia", name=f"run3-{args.dataset}-signals", config={
        "dataset": args.dataset,
        "T": T, "K": K, "grad_timesteps": GRAD_T,
        "alpha_min": ALPHA_MIN, "alpha_max": ALPHA_MAX,
        "max_length": MAX_LENGTH, **vars(args),
    })

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    ft_path   = os.path.join(model_dir, "finetuned_checkpoint")
    base_path = os.path.join(model_dir, "base_checkpoint")

    print("Loading finetuned model (eager attn)...")
    ft_model = load_model_eager(ft_path, dtype, device)

    print("Loading base model (eager attn)...")
    base_model = load_model_eager(base_path, dtype, device)

    # Use Qwen2Tokenizer directly — transformers>=4.57 AutoTokenizer calls AutoConfig
    # first, which fails for local paths with unknown model_type 'a2d-qwen3'.
    tokenizer = Qwen2Tokenizer.from_pretrained(ft_path)

    # Load data
    members    = torch.load(os.path.join(data_dir, "members.pt"),    weights_only=False)
    nonmembers = torch.load(os.path.join(data_dir, "nonmembers.pt"), weights_only=False)

    member_texts    = members["texts"]
    nonmember_texts = nonmembers["texts"]

    if args.n_samples is not None:
        member_texts    = member_texts[: args.n_samples]
        nonmember_texts = nonmember_texts[: args.n_samples]

    all_texts = list(member_texts) + list(nonmember_texts)
    labels    = [1] * len(member_texts) + [0] * len(nonmember_texts)
    total     = len(all_texts)
    print(f"Extracting signals from {total} samples ({len(member_texts)} members, {len(nonmember_texts)} non-members)...")

    # Determine if extract_metrics accepts max_length
    import inspect
    _em_sig = inspect.signature(extract_metrics)
    _em_accepts_max_length = "max_length" in _em_sig.parameters

    feature_vectors = []
    failed = 0

    for i, text in enumerate(tqdm(all_texts, desc="Extracting signals")):
        try:
            em_kwargs = dict(
                n_timesteps=T, n_mask_configs=K,
                grad_timesteps=GRAD_T, capture_attentions=True, seed=SEED,
                alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
            )
            if _em_accepts_max_length:
                em_kwargs["max_length"] = MAX_LENGTH

            # Primary extraction on finetuned model
            bundle_ft = extract_metrics(ft_model, tokenizer, text, **em_kwargs)
            fv = bundle_ft.to_feature_vector()  # [11*T + 1] = [45 for T=4]

            # Cross-model feature: base model with lighter settings
            base_kwargs = dict(
                n_timesteps=T, n_mask_configs=4,
                grad_timesteps=0, capture_attentions=False, seed=SEED,
                alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
            )
            if _em_accepts_max_length:
                base_kwargs["max_length"] = MAX_LENGTH

            bundle_base = extract_metrics(base_model, tokenizer, text, **base_kwargs)
            cos_sim = cross_model_cosine_sim(bundle_ft, bundle_base)
            cos_sim_t = torch.tensor([cos_sim], dtype=fv.dtype)

            fv_full = torch.cat([fv, cos_sim_t], dim=0)  # [46 for T=4]
            feature_vectors.append(fv_full)

            wandb.log({"signals/fv_norm": fv_full.norm().item(), "signals/sample": i + 1})

        except Exception as e:
            print(f"  WARNING: sample {i} failed: {e}")
            failed += 1
            feat_dim = 11 * T + 1 + 1  # 45+1=46 for T=4
            feature_vectors.append(torch.zeros(feat_dim))

    print(f"\nExtraction complete. Failed: {failed}/{total}")

    X = torch.stack(feature_vectors)  # [N, 46 for T=4]
    y = torch.tensor(labels, dtype=torch.long)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(X, os.path.join(out_dir, "X.pt"))
    torch.save(y, os.path.join(out_dir, "y.pt"))
    print(f"Saved X.pt {X.shape}, y.pt {y.shape}")

    # ----------------------------------------------------------------
    # Ablation study outputs
    # ----------------------------------------------------------------

    # 1. X_per_group.pt — group slices for leave-one-out ablation
    X_per_group = {name: X[:, slc].clone() for name, slc in GROUPS.items()}
    torch.save(X_per_group, os.path.join(out_dir, "X_per_group.pt"))
    print(f"Saved X_per_group.pt ({len(GROUPS)} groups)")

    # 2. signal_names.pt — human-readable feature names matching X columns
    signal_names = []
    alphas = torch.linspace(ALPHA_MIN, ALPHA_MAX, T).tolist()
    for t in range(T):
        signal_names.append(f"elbo_t{t}_a{alphas[t]:.2f}")
    signal_names.append("elbo_var")
    for t in range(T):
        signal_names.append(f"dldt_t{t}")
    for t in range(T):
        signal_names.append(f"d2ldt2_t{t}")
    for t in range(T):
        signal_names.append(f"pred_entropy_t{t}")
    for t in range(T):
        signal_names.append(f"mask_consistency_t{t}")
    for t in range(T):
        signal_names.append(f"hidden_norms_t{t}")
    for t in range(T):
        signal_names.append(f"hidden_cosine_t{t}")
    for t in range(T):
        signal_names.append(f"attn_entropy_t{t}")
    for t in range(T):
        signal_names.append(f"attn_crosslayer_t{t}")
    for t in range(T):
        signal_names.append(f"attn_barycenter_t{t}")
    for t in range(T):
        signal_names.append(f"attn_perturbation_t{t}")
    signal_names.append("cross_model_cos")
    assert len(signal_names) == X.shape[1], \
        f"signal_names len {len(signal_names)} != X cols {X.shape[1]}"
    torch.save(signal_names, os.path.join(out_dir, "signal_names.pt"))
    print(f"Saved signal_names.pt ({len(signal_names)} names)")

    # 3. raw_signals.pt — per-sample raw bundle fields stacked as [N, T] or [N, GRAD_T]
    #    Collected during extraction from bundle_ft fields. Re-slice from X for groups
    #    that map directly, and note which were skipped.
    raw_signals = {
        "elbo_per_t":           X[:, GROUPS["elbo_traj"]].clone(),    # [N, T]
        "elbo_var":             X[:, GROUPS["elbo_var"]].clone(),      # [N, 1]
        "dldt":                 X[:, GROUPS["dldt"]].clone(),          # [N, T]
        "d2ldt2":               X[:, GROUPS["d2ldt2"]].clone(),        # [N, T]
        "pred_entropy_per_t":   X[:, GROUPS["pred_entropy"]].clone(),  # [N, T]
        "mask_consistency_per_t": X[:, GROUPS["mask_consistency"]].clone(),  # [N, T]
        "hidden_norms_per_t":   X[:, GROUPS["hidden_norms"]].clone(),  # [N, T]
        "hidden_cosine_per_t":  X[:, GROUPS["hidden_cosine"]].clone(), # [N, T]
        "attn_entropy_per_t":   X[:, GROUPS["attn_entropy"]].clone(),  # [N, T]
        "attn_crosslayer_per_t": X[:, GROUPS["attn_crosslayer"]].clone(),  # [N, T]
        "attn_barycenter_per_t": X[:, GROUPS["attn_barycenter"]].clone(),  # [N, T]
        "attn_perturbation_per_t": X[:, GROUPS["attn_perturbation"]].clone(),  # [N, T]
        "cross_model_cos":      X[:, GROUPS["cross_model_cos"]].clone(),  # [N, 1]
        "labels":               y.clone(),
        "T":                    T,
        "alphas":               torch.tensor(alphas),
    }
    torch.save(raw_signals, os.path.join(out_dir, "raw_signals.pt"))
    print(f"Saved raw_signals.pt ({len(raw_signals)} keys)")

    wandb.log({"signals/total": total, "signals/failed": failed, "signals/feat_dim": X.shape[1]})
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
