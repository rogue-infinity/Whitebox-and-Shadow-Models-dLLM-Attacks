"""
run_shadow_mia.py — Shadow Model MIA orchestrator for Run_4_Shadow_MIA.

Threat model: attacker has NO access to the target model's fine-tuning data.
Instead, K=3 shadow models are fine-tuned on surrogate data from a disjoint
MIMIR split (ngram_13_0.2). Classifiers trained on shadow features are then
transferred to the target model's features (Run3 X.pt) without retraining.

Five conditions (A–E) compare oracle vs shadow transfer across different
feature subsets, with B vs E as the primary ablation (46-dim vs 30-dim pruned).

Phases:
  1  prepare_shadow_data    — load MIMIR ngram_13_0.2, partition into K chunks
  2  train_shadow_models    — fine-tune K shadow models sequentially
  3  extract_shadow_features — extract 46-dim features while model is in GPU memory
  4  train_and_transfer     — 5 conditions + distribution analysis + SHAP
  5  aggregate_results      — save transfer_results.pt, log final wandb tables

Usage:
    python run_shadow_mia.py --dataset hackernews
    python run_shadow_mia.py --dataset arxiv --start_phase 4   # resume
    python run_shadow_mia.py --dataset arxiv --smoke            # quick sanity check

Run across all 6 domains in parallel on separate L4 instances:
    See run_shadow_launcher.sh
"""

import argparse
import gc
import inspect
import math
import os
import random
import sys
import time

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
from transformers import Qwen2Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from mdlm_metrics_extractor import extract_metrics

TIME_EPSILON = 1e-3   # ε_min for masking time schedule

# All MIMIR domains — Phase 1 pools member texts from ALL of these so that
# even small domains (e.g. hackernews with ~35 rows) get enough shadow samples.
ALL_MIMIR_DOMAINS = ["arxiv", "github", "hackernews", "pubmed_central", "wikipedia", "pile_cc"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_eager(path: str, dtype, device):
    """Load MDLM model with eager attention so attention weights are captured."""
    return AutoModelForMaskedLM.from_pretrained(
        path,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",   # required — flash/sdpa returns None for attentions
    ).to(device).eval()


def mdlm_loss(model, input_ids, attention_mask, mask_token_id, device):
    """MDLM ELBO loss (identical to finetune.py — do not modify)."""
    B, L = input_ids.shape
    t = torch.empty(B, device=device).uniform_(TIME_EPSILON, 1.0)
    p_mask = t.unsqueeze(1).expand(B, L)
    rand = torch.rand(B, L, device=device)
    mask_positions = (rand < p_mask) & (attention_mask == 1)
    if mask_positions.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    noised = input_ids.clone()
    noised[mask_positions] = mask_token_id
    logits = model(input_ids=noised, attention_mask=attention_mask).logits
    return torch.nn.functional.cross_entropy(
        logits[mask_positions], input_ids[mask_positions], reduction="mean"
    )


def cross_model_cosine_sim(bundle_ft, bundle_base) -> float:
    """Cosine similarity between mean hidden norms of finetuned vs base model.
    Copied from run_signals.py — must remain identical."""
    hn_ft   = bundle_ft.hidden_norms
    hn_base = bundle_base.hidden_norms
    if hn_ft is None or hn_base is None:
        return 0.0
    v_ft   = hn_ft.mean(dim=0).float()
    v_base = hn_base.mean(dim=0).float()
    n = min(v_ft.shape[0], v_base.shape[0])
    return torch.nn.functional.cosine_similarity(
        v_ft[:n].unsqueeze(0), v_base[:n].unsqueeze(0)
    ).item()


def build_signal_names(T: int) -> list:
    """Generate the 46 feature names in the same order as run_signals.py."""
    alphas = torch.linspace(cfg.ALPHA_MIN, cfg.ALPHA_MAX, T).tolist()
    names = []
    for t in range(T):
        names.append(f"elbo_t{t}_a{alphas[t]:.2f}")
    names.append("elbo_var")
    for prefix in ["dldt", "d2ldt2", "pred_entropy", "mask_consistency",
                   "hidden_norms", "hidden_cosine", "attn_entropy",
                   "attn_crosslayer", "attn_barycenter", "attn_perturbation"]:
        for t in range(T):
            names.append(f"{prefix}_t{t}")
    names.append("cross_model_cos")
    return names


def free_gpu(model):
    """Delete model and free GPU memory before loading the next shadow."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_free = torch.cuda.mem_get_info()[0] / 1e9
        wandb.log({"memory/gpu_free_gb": mem_free})


# ---------------------------------------------------------------------------
# Dataset loading helper — supports datasets <3.0 (remote) and >=3.0 (local)
# ---------------------------------------------------------------------------

def _load_mimir_members(repo: str, hf_name: str, split: str, hf_token: str) -> list:
    """
    Load member texts from iamgroot42/mimir.
    datasets >=3.0 dropped support for custom loading scripts (mimir.py).
    Primary path: standard load_dataset (works on remote with datasets<3.0).
    Fallback: download parquet files directly via huggingface_hub and read with pandas.
    """
    # Primary: standard datasets API.
    # datasets <3.0  (JarvisLabs remote): needs trust_remote_code=True
    # datasets >=3.0 (local, newer):      loading scripts fully dropped → parquet fallback
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, hf_name, split=split,
                          token=hf_token or True, trust_remote_code=True)
        return [row["member"] for row in ds if isinstance(row.get("member"), str)]
    except (RuntimeError, Exception) as e:
        err = str(e)
        if "loading script" not in err and "no longer supported" not in err and \
           "Dataset scripts are no longer supported" not in err:
            raise
        print(f"  [INFO] datasets loading-script API unavailable — falling back to direct parquet download.")

    # Fallback: download parquet files directly via huggingface_hub
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    parquet_files = [
        f for f in list_repo_files(repo, repo_type="dataset", token=hf_token or True)
        if hf_name in f and split in f and f.endswith(".parquet")
    ]

    if not parquet_files:
        # Try without split in path (some repos store all splits in one file per config)
        parquet_files = [
            f for f in list_repo_files(repo, repo_type="dataset", token=hf_token or True)
            if hf_name in f and f.endswith(".parquet")
        ]

    if not parquet_files:
        raise RuntimeError(
            f"Could not find parquet files for {repo}/{hf_name}/{split}. "
            f"Check dataset structure on HuggingFace Hub."
        )

    all_texts = []
    for pf in sorted(parquet_files):
        local = hf_hub_download(repo_id=repo, filename=pf,
                                repo_type="dataset", token=hf_token or True)
        df = pd.read_parquet(local)
        if "member" in df.columns:
            all_texts.extend(df["member"].dropna().astype(str).tolist())

    print(f"  Loaded {len(all_texts)} member texts from {len(parquet_files)} parquet file(s)")
    return all_texts


# ---------------------------------------------------------------------------
# Phase 1 — Surrogate Data Construction
# ---------------------------------------------------------------------------

def phase1_prepare_shadow_data(dataset: str, out_dir: str, smoke: bool, hf_token: str):
    """
    Load MIMIR ngram_13_0.2 split (disjoint from Run3's ngram_13_0.8).
    Partition into K=3 non-overlapping member chunks (600 each).
    Non-members for shadow k: 400 samples sampled from the other two chunks.
    Saves: {out_dir}/surrogate_data.pt
    """
    save_path = os.path.join(out_dir, "surrogate_data.pt")
    if os.path.exists(save_path):
        wandb.log({"phase1/skipped": True})
        print(f"Phase 1: surrogate_data.pt already exists, skipping.")
        return

    # Pool member texts from ALL 6 MIMIR domains so that even small domains
    # (e.g. hackernews with ~35 rows in this split) don't starve the shadow classifier.
    # This implements a domain-agnostic shadow attacker — stronger threat model.
    print(f"Phase 1: Pooling ngram_13_0.2 members from ALL {len(ALL_MIMIR_DOMAINS)} domains...")
    member_texts = []
    for d in ALL_MIMIR_DOMAINS:
        hf_name_d = cfg.MIMIR_HF_NAME[d]
        try:
            texts = _load_mimir_members(cfg.MIMIR_HF_REPO, hf_name_d,
                                        cfg.SHADOW_MIMIR_SPLIT, hf_token)
            member_texts.extend(texts)
            print(f"  {d}: +{len(texts)} → total {len(member_texts)}")
            wandb.log({f"phase1/domain_{d}_count": len(texts)})
        except Exception as e:
            print(f"  [WARN] Could not load {d}: {e}")

    random.seed(cfg.SHADOW_BASE_SEED)
    random.shuffle(member_texts)

    n_needed = cfg.N_SHADOW * cfg.SHADOW_CHUNK_SIZE   # 3 * 600 = 1800
    if smoke:
        n_needed = cfg.N_SHADOW * 20   # 60 samples total in smoke mode

    if len(member_texts) < n_needed:
        new_chunk = len(member_texts) // cfg.N_SHADOW
        msg = (f"Pooled ngram_13_0.2 has only {len(member_texts)} member rows "
               f"(need {n_needed}). Reducing chunk size {cfg.SHADOW_CHUNK_SIZE}→{new_chunk}.")
        print(f"WARNING: {msg}")
        wandb.alert(title="Pooled surrogate shortage", text=msg, level=wandb.AlertLevel.WARN)
        chunk_size = new_chunk
    else:
        chunk_size = cfg.SHADOW_CHUNK_SIZE if not smoke else 20

    nonmember_size = cfg.SHADOW_NONMEMBER_SIZE if not smoke else 15

    # Tokenize with Qwen2Tokenizer — never AutoTokenizer on local paths (Issue 11)
    tokenizer = Qwen2Tokenizer.from_pretrained(os.environ.get("MODEL_PATH", cfg.MODEL_PATH), trust_remote_code=True)

    def tokenize(texts):
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=cfg.MAX_LENGTH, return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    # Partition 3 disjoint chunks for member sets
    chunks_ids, chunks_mask, chunks_texts = [], [], []
    for k in range(cfg.N_SHADOW):
        start = k * chunk_size
        end   = start + chunk_size
        batch = member_texts[start:end]
        ids, mask = tokenize(batch)
        chunks_ids.append(ids)
        chunks_mask.append(mask)
        chunks_texts.append(batch)

    # Verify no overlap between chunks (deterministic slices are non-overlapping by construction)
    assert len(set(chunks_texts[0]) & set(chunks_texts[1])) == 0, "Chunk 0/1 overlap"
    assert len(set(chunks_texts[0]) & set(chunks_texts[2])) == 0, "Chunk 0/2 overlap"
    assert len(set(chunks_texts[1]) & set(chunks_texts[2])) == 0, "Chunk 1/2 overlap"

    rng = np.random.default_rng(cfg.SHADOW_BASE_SEED)

    member_sets, nonmember_sets, label_sets = [], [], []
    for k in range(cfg.N_SHADOW):
        # Members for shadow k: its own chunk
        mem_ids  = chunks_ids[k]
        mem_mask = chunks_mask[k]

        # Non-members: 400 sampled from the OTHER two chunks (combined pool)
        other_idx = [j for j in range(cfg.N_SHADOW) if j != k]
        pool_ids  = torch.cat([chunks_ids[j]  for j in other_idx], dim=0)
        pool_mask = torch.cat([chunks_mask[j] for j in other_idx], dim=0)

        n_nonmem = min(nonmember_size, pool_ids.shape[0])
        sel = rng.choice(pool_ids.shape[0], size=n_nonmem, replace=False)
        nonmem_ids  = pool_ids[sel]
        nonmem_mask = pool_mask[sel]

        # Stack: members first, then non-members
        all_ids  = torch.cat([mem_ids,  nonmem_ids],  dim=0)
        all_mask = torch.cat([mem_mask, nonmem_mask], dim=0)
        labels   = torch.cat([
            torch.ones(mem_ids.shape[0],  dtype=torch.long),
            torch.zeros(nonmem_ids.shape[0], dtype=torch.long),
        ])

        member_sets.append({"input_ids": mem_ids,  "attention_mask": mem_mask})
        nonmember_sets.append({"input_ids": nonmem_ids, "attention_mask": nonmem_mask})
        label_sets.append(labels)

    torch.save({
        "chunks_ids":      chunks_ids,        # list of K tensors [chunk_size, 256]
        "chunks_mask":     chunks_mask,
        "member_sets":     member_sets,        # list of K dicts {input_ids, attention_mask}
        "nonmember_sets":  nonmember_sets,
        "labels":          label_sets,         # list of K tensors [chunk_size + nonmember_size]
        "chunk_size":      chunk_size,
        "nonmember_size":  nonmember_size,
        "pooled_domains":  ALL_MIMIR_DOMAINS,  # all 6 domains pooled for shadow training
    }, save_path)

    wandb.log({
        "phase1/chunk_size":     chunk_size,
        "phase1/nonmember_size": nonmember_size,
        "phase1/total_members":  chunk_size * cfg.N_SHADOW,
        "phase1/done":           True,
    })
    print(f"Phase 1 done. Saved surrogate_data.pt  "
          f"({chunk_size} members + {nonmember_size} non-members per shadow)")


# ---------------------------------------------------------------------------
# Phase 2 + 3 — Fine-tune shadow k and extract features (combined to avoid
#               reloading the model; extraction happens before deletion)
# ---------------------------------------------------------------------------

def finetune_and_extract_shadow(
    k: int,
    dataset: str,
    surrogate_data: dict,
    base_model,            # base model already loaded (for cross_model_cos + ELBO gap)
    out_dir: str,
    device,
    dtype,
    smoke: bool,
):
    """
    Fine-tune shadow model k on its member chunk, then immediately extract
    46-dim features for all 1000 samples before freeing GPU memory.

    Returns: (X_k [N,46], y_k [N]) where N = chunk_size + nonmember_size
    """
    feat_path = os.path.join(out_dir, f"shadow_k{k}_features.pt")
    if os.path.exists(feat_path):
        print(f"  Shadow {k}: features already exist, loading.")
        saved = torch.load(feat_path, weights_only=False)
        return saved["X"], saved["y"]

    seed = cfg.SHADOW_BASE_SEED + k
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"\n{'='*60}\n Shadow {k} — seed={seed}\n{'='*60}")

    # ----------------------------------------------------------------
    # Load shadow model (base weights, eager attn for feature extraction)
    # ----------------------------------------------------------------
    print(f"  Loading base model for shadow {k}...")
    shadow_model = AutoModelForMaskedLM.from_pretrained(
        os.environ.get("MODEL_PATH", cfg.MODEL_PATH),
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",   # needed for attention features in Phase 3
    ).to(device)
    shadow_model.train()

    tokenizer = Qwen2Tokenizer.from_pretrained(os.environ.get("MODEL_PATH", cfg.MODEL_PATH), trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 151669   # Qwen3-0.6B-diffusion-mdlm-v0.1 hardcoded fallback

    # ----------------------------------------------------------------
    # Phase 2 — Fine-tune on member chunk
    # ----------------------------------------------------------------
    mem_data = surrogate_data["member_sets"][k]
    input_ids      = mem_data["input_ids"].to(device)
    attention_mask = mem_data["attention_mask"].to(device)
    N_train = input_ids.shape[0]

    if hasattr(shadow_model, "gradient_checkpointing_enable"):
        shadow_model.gradient_checkpointing_enable()

    from torch.optim import AdamW
    optimizer = AdamW(
        shadow_model.parameters(),
        lr=cfg.FT_LR, weight_decay=cfg.FT_WD,
        betas=(0.9, 0.999), eps=1e-8,
    )

    n_epochs    = 2 if smoke else cfg.FT_EPOCHS
    max_steps   = 4 if smoke else None
    global_step = 0
    done        = False

    for epoch in range(n_epochs):
        if done:
            break
        indices   = list(range(N_train))
        random.shuffle(indices)
        n_batches = math.ceil(N_train / cfg.FT_BATCH_SIZE)
        epoch_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"  Shadow {k} Epoch {epoch+1}/{n_epochs}")
        for step in pbar:
            batch_idx = indices[step * cfg.FT_BATCH_SIZE : (step+1) * cfg.FT_BATCH_SIZE]
            ids  = input_ids[batch_idx]
            mask = attention_mask[batch_idx]

            optimizer.zero_grad()
            loss = mdlm_loss(shadow_model, ids, mask, mask_token_id, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow_model.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_loss  += lv
            global_step += 1
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({f"shadow_k{k}/train_loss": lv, f"shadow_k{k}/step": global_step})

            if max_steps and global_step >= max_steps:
                done = True
                break

        mean_loss = epoch_loss / max(1, n_batches if not done else step + 1)
        wandb.log({f"shadow_k{k}/epoch_loss": mean_loss, f"shadow_k{k}/epoch": epoch + 1})

    # ----------------------------------------------------------------
    # Sanity check: ELBO gap on member set (base vs shadow)
    # ----------------------------------------------------------------
    shadow_model.eval()
    with torch.no_grad():
        n_gap_batches = min(4, math.ceil(N_train / cfg.FT_BATCH_SIZE))
        gap_losses_base, gap_losses_ft = [], []
        for step in range(n_gap_batches):
            batch_idx = list(range(step * cfg.FT_BATCH_SIZE,
                                   min((step+1) * cfg.FT_BATCH_SIZE, N_train)))
            ids  = input_ids[batch_idx]
            mask = attention_mask[batch_idx]

            # Fine-tuned loss
            l_ft = mdlm_loss(shadow_model, ids, mask, mask_token_id, device)
            gap_losses_ft.append(l_ft.item())

            # Base model loss (already loaded, kept in eval mode)
            l_base = mdlm_loss(base_model, ids, mask, mask_token_id, device)
            gap_losses_base.append(l_base.item())

    elbo_gap = float(np.mean(gap_losses_base) - np.mean(gap_losses_ft))
    wandb.log({f"shadow_k{k}/elbo_gap_members": elbo_gap})
    print(f"  Shadow {k} ELBO gap: {elbo_gap:.4f} nats")

    if elbo_gap < cfg.SHADOW_MIN_ELBO_GAP:
        wandb.alert(
            title=f"Low ELBO gap: shadow {k}",
            text=f"Shadow {k} ELBO gap = {elbo_gap:.4f} nats < {cfg.SHADOW_MIN_ELBO_GAP}. "
                 f"Fine-tuning may not have memorized well. Continuing anyway.",
            level=wandb.AlertLevel.WARN,
        )

    # Free optimizer memory before feature extraction
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Phase 3 — Extract 46-dim features for all 1000 samples
    # ----------------------------------------------------------------
    print(f"  Extracting features for shadow {k}...")

    mem_data    = surrogate_data["member_sets"][k]
    nonmem_data = surrogate_data["nonmember_sets"][k]
    labels_k    = surrogate_data["labels"][k]

    # Build text list: members first, then non-members
    # We decode input_ids back to text for extract_metrics (which takes raw text)
    def ids_to_texts(input_ids_tensor):
        texts = []
        for row in input_ids_tensor:
            # Strip padding before decoding
            ids = row.tolist()
            pad_id = tokenizer.pad_token_id or 0
            ids = [i for i in ids if i != pad_id]
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    member_texts    = ids_to_texts(mem_data["input_ids"])
    nonmember_texts = ids_to_texts(nonmem_data["input_ids"])
    all_texts = member_texts + nonmember_texts
    total     = len(all_texts)

    # Check if extract_metrics accepts max_length parameter
    _em_sig            = inspect.signature(extract_metrics)
    _accepts_max_length = "max_length" in _em_sig.parameters

    em_kwargs = dict(
        n_timesteps=cfg.T_STEPS,
        n_mask_configs=cfg.SIG_K,
        grad_timesteps=cfg.SIG_GRAD_T,
        capture_attentions=True,     # required for attn_entropy / attn_crosslayer etc.
        seed=cfg.SHADOW_BASE_SEED,   # fixed — reproducible features
        alpha_min=cfg.ALPHA_MIN,
        alpha_max=cfg.ALPHA_MAX,
    )
    base_em_kwargs = dict(
        n_timesteps=cfg.T_STEPS,
        n_mask_configs=4,            # fewer configs for base model (faster)
        grad_timesteps=0,
        capture_attentions=False,
        seed=cfg.SHADOW_BASE_SEED,
        alpha_min=cfg.ALPHA_MIN,
        alpha_max=cfg.ALPHA_MAX,
    )
    if _accepts_max_length:
        em_kwargs["max_length"]      = cfg.MAX_LENGTH
        base_em_kwargs["max_length"] = cfg.MAX_LENGTH

    feature_vectors = []
    failed = 0
    feat_dim = 11 * cfg.T_STEPS + 2   # = 46 for T=4

    t0 = time.time()
    for i, text in enumerate(tqdm(all_texts, desc=f"  Shadow {k} features")):
        try:
            bundle_ft   = extract_metrics(shadow_model, tokenizer, text, **em_kwargs)
            fv          = bundle_ft.to_feature_vector()    # [45 for T=4]

            bundle_base = extract_metrics(base_model, tokenizer, text, **base_em_kwargs)
            cos_sim     = cross_model_cosine_sim(bundle_ft, bundle_base)
            fv_full     = torch.cat([fv, torch.tensor([cos_sim], dtype=fv.dtype)])  # [46]

            feature_vectors.append(fv_full.cpu())

            if i % 50 == 0:
                elapsed = time.time() - t0
                wandb.log({
                    f"shadow_k{k}/signals_done": i + 1,
                    f"shadow_k{k}/signals_total": total,
                    f"shadow_k{k}/elapsed_s": elapsed,
                    f"shadow_k{k}/fv_norm": fv_full.norm().item(),
                })

        except Exception as e:
            print(f"  WARNING: shadow {k} sample {i} failed: {e}")
            failed += 1
            feature_vectors.append(torch.zeros(feat_dim))

    X_k = torch.stack(feature_vectors)   # [N, 46]
    y_k = labels_k.clone()

    wandb.log({
        f"shadow_k{k}/feat_shape":       list(X_k.shape),
        f"shadow_k{k}/extract_failed":   failed,
        f"shadow_k{k}/extract_total":    total,
        f"shadow_k{k}/extract_elapsed_s": time.time() - t0,
    })

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X": X_k, "y": y_k}, feat_path)
    print(f"  Shadow {k}: saved features {X_k.shape}, failed={failed}/{total}")

    # ----------------------------------------------------------------
    # Delete shadow model and free GPU before next shadow
    # ----------------------------------------------------------------
    free_gpu(shadow_model)
    return X_k, y_k


# ---------------------------------------------------------------------------
# Phase 4 + 5 — Classifier training, distribution analysis, SHAP, transfer
# ---------------------------------------------------------------------------

def phase4_train_and_transfer(
    dataset: str,
    out_dir: str,
    run3_results: str,
    smoke: bool,
):
    """
    Train 5 conditions on pooled shadow features, evaluate on target (Run3) features.
    Also runs full distribution analysis (KDE, Hellinger, KS-test) on all 46 features
    for both shadow and target populations.
    Saves: {out_dir}/transfer_results.pt
    """
    from shadow_classifier import train_all_conditions, run_distribution_analysis

    # ----------------------------------------------------------------
    # Load pooled shadow features
    # ----------------------------------------------------------------
    shadow_feat_path = os.path.join(out_dir, "shadow_features.pt")
    sf = torch.load(shadow_feat_path, weights_only=False)
    X_shadow = sf["X_46"].numpy().astype(np.float32)   # [3000, 46]
    y_shadow = sf["y"].numpy()

    # ----------------------------------------------------------------
    # Load target features from Run3 — never re-extract
    # ----------------------------------------------------------------
    X_target = torch.load(
        os.path.join(run3_results, dataset, "X.pt"), weights_only=True
    ).numpy().astype(np.float32)   # [2000, 46]
    y_target = torch.load(
        os.path.join(run3_results, dataset, "y.pt"), weights_only=True
    ).numpy()

    # Load Run3 oracle AUC (Condition A) — correct key is metrics_xgb.auc_mean
    clf_r3 = torch.load(
        os.path.join(run3_results, dataset, "classifier_results.pt"),
        weights_only=False,
    )
    oracle_auc = clf_r3["metrics_xgb"]["auc_mean"]

    # Load Run3 solo AUC per signal group for group-transfer comparison
    ablation_r3 = torch.load(
        os.path.join(run3_results, dataset, "ablation_results.pt"),
        weights_only=False,
    )
    run3_solo_auc = {g: ablation_r3[g]["solo_auc"] for g in ablation_r3}

    # Load signal names from Run3 (or rebuild if absent)
    sig_names_path = os.path.join(run3_results, dataset, "signal_names.pt")
    if os.path.exists(sig_names_path):
        signal_names = torch.load(sig_names_path, weights_only=False)
    else:
        signal_names = build_signal_names(cfg.T_STEPS)

    print(f"\nPhase 4: shadow {X_shadow.shape}, target {X_target.shape}")
    print(f"  Oracle AUC (Run3): {oracle_auc:.4f}")

    # ----------------------------------------------------------------
    # Distribution analysis — all 46 features, shadow + target populations
    # ----------------------------------------------------------------
    print("  Running distribution analysis (shadow features)...")
    shadow_dist = run_distribution_analysis(X_shadow, y_shadow, signal_names,
                                             population="shadow", smoke=smoke)

    print("  Running distribution analysis (target features)...")
    target_dist = run_distribution_analysis(X_target, y_target, signal_names,
                                             population="target", smoke=smoke)

    # ----------------------------------------------------------------
    # Train all 5 conditions + group transfer
    # ----------------------------------------------------------------
    print("  Training 5 conditions...")
    condition_results, group_results = train_all_conditions(
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        X_target=X_target,
        y_target=y_target,
        oracle_auc=oracle_auc,
        run3_solo_auc=run3_solo_auc,
        signal_names=signal_names,
        smoke=smoke,
    )

    # ----------------------------------------------------------------
    # Log final condition comparison table to wandb
    # ----------------------------------------------------------------
    columns = ["condition", "n_features", "auc", "auc_lo", "auc_hi",
               "tpr_0.001", "tpr_0.01", "tpr_0.10", "gap_from_oracle"]
    table = wandb.Table(columns=columns)
    for cname, res in condition_results.items():
        m = res["metrics"]
        gap = oracle_auc - m.get("auc_mean", m.get("auc", 0.0))
        table.add_data(
            cname, res["n_features"],
            m.get("auc_mean", m.get("auc", 0.0)),
            m.get("auc_lo", 0.0), m.get("auc_hi", 0.0),
            m.get("tpr_at_0.001", 0.0),
            m.get("tpr_at_0.01",  0.0),
            m.get("tpr_at_0.10",  m.get("tpr_at_0.1", 0.0)),
            gap,
        )
        auc_val = m.get("auc_mean", m.get("auc", 0.0))
        wandb.log({
            f"transfer/{cname}/auc":           auc_val,
            f"transfer/{cname}/auc_lo":        m.get("auc_lo", 0.0),
            f"transfer/{cname}/auc_hi":        m.get("auc_hi", 0.0),
            f"transfer/{cname}/tpr_at_0001":   m.get("tpr_at_0.001", 0.0),
            f"transfer/{cname}/tpr_at_001":    m.get("tpr_at_0.01",  0.0),
            f"transfer/{cname}/tpr_at_010":    m.get("tpr_at_0.10",  m.get("tpr_at_0.1", 0.0)),
            f"transfer/{cname}/gap_from_oracle": gap,
            f"transfer/{cname}/n_features":    res["n_features"],
        })

    wandb.log({"transfer/condition_table": table})

    # B vs E primary ablation delta
    if "B_full46" in condition_results and "E_pruned30" in condition_results:
        auc_B = condition_results["B_full46"]["metrics"].get("auc_mean",
                condition_results["B_full46"]["metrics"].get("auc", 0.0))
        auc_E = condition_results["E_pruned30"]["metrics"].get("auc_mean",
                condition_results["E_pruned30"]["metrics"].get("auc", 0.0))
        wandb.log({
            "transfer/b_vs_e_delta":          auc_B - auc_E,
            "transfer/b_vs_e_B_wins":         int(auc_B > auc_E),
            "transfer/auc_oracle":            oracle_auc,
            "transfer/auc_B_full46":          auc_B,
            "transfer/auc_E_pruned30":        auc_E,
        })

    # Group transfer table
    group_cols = ["group", "shadow_transfer_auc", "run3_solo_auc", "degradation"]
    group_table = wandb.Table(columns=group_cols)
    for gname, gauc in group_results.items():
        solo = run3_solo_auc.get(gname, float("nan"))
        group_table.add_data(gname, gauc, solo, solo - gauc)
        wandb.log({
            f"group_transfer/{gname}/shadow_auc":    gauc,
            f"group_transfer/{gname}/run3_solo_auc": solo,
            f"group_transfer/{gname}/degradation":   solo - gauc,
        })
    wandb.log({"group_transfer/table": group_table})

    # ----------------------------------------------------------------
    # Save everything
    # ----------------------------------------------------------------
    torch.save({
        "dataset":          dataset,
        "conditions":       condition_results,
        "group_transfer":   group_results,
        "oracle_auc":       oracle_auc,
        "run3_solo_auc":    run3_solo_auc,
        "signal_names":     signal_names,
        "shadow_dist":      shadow_dist,
        "target_dist":      target_dist,
    }, os.path.join(out_dir, "transfer_results.pt"))
    print(f"  Saved transfer_results.pt")

    return condition_results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run4 Shadow Model MIA")
    parser.add_argument("--dataset",      required=True, choices=cfg.MIMIR_DATASETS)
    parser.add_argument("--start_phase",  type=int, default=1, choices=[1, 2, 3, 4],
                        help="Resume from this phase (1=full run)")
    parser.add_argument("--run3_results", default=cfg.RUN3_RESULTS_DEFAULT,
                        help="Path to Run3 results directory containing target X.pt, y.pt")
    parser.add_argument("--smoke",        action="store_true",
                        help="Smoke test: very few samples, 2 epochs, quick sanity check")
    args = parser.parse_args()

    dataset  = args.dataset
    out_dir  = os.path.join(cfg.SHADOW_RESULTS_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")

    # ----------------------------------------------------------------
    # wandb init for orchestrator
    # ----------------------------------------------------------------
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=f"{cfg.WANDB_RUN_PREFIX}-{dataset}-shadow",
        config={
            "dataset":        dataset,
            "n_shadow":       cfg.N_SHADOW,
            "shadow_split":   cfg.SHADOW_MIMIR_SPLIT,
            "target_split":   cfg.MIMIR_SPLIT,
            "smoke":          args.smoke,
            "start_phase":    args.start_phase,
            "T_steps":        cfg.T_STEPS,
            "alpha_min":      cfg.ALPHA_MIN,
            "alpha_max":      cfg.ALPHA_MAX,
            "K_mask_configs": cfg.SIG_K,
            "FT_epochs":      cfg.FT_EPOCHS,
            "FT_lr":          cfg.FT_LR,
            "FT_wd":          cfg.FT_WD,
            "FT_batch_size":  cfg.FT_BATCH_SIZE,
            "seed_base":      cfg.SHADOW_BASE_SEED,
            "conditions":     list(cfg.SHADOW_CONDITIONS.keys()),
        },
        tags=["run4", "shadow-mia", dataset] + (["smoke"] if args.smoke else []),
    )

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    wandb.log({"device": str(device), "dtype": str(dtype)})
    print(f"Device: {device}, dtype: {dtype}")

    # ----------------------------------------------------------------
    # Phase 1 — Surrogate data
    # ----------------------------------------------------------------
    if args.start_phase <= 1:
        print("\n[Phase 1] Preparing surrogate data...")
        phase1_prepare_shadow_data(dataset, out_dir, args.smoke, hf_token)

    surrogate_data = torch.load(
        os.path.join(out_dir, "surrogate_data.pt"), weights_only=False
    )

    # ----------------------------------------------------------------
    # Load base model ONCE — used for ELBO gap + cross_model_cos across all shadows
    # ----------------------------------------------------------------
    print("\nLoading base model (shared across all shadow extractions)...")
    base_model = load_model_eager(os.environ.get("MODEL_PATH", cfg.MODEL_PATH), dtype, device)
    base_model.eval()

    # ----------------------------------------------------------------
    # Phases 2 + 3 — Fine-tune and extract features for each shadow k
    # ----------------------------------------------------------------
    all_X, all_y = [], []

    for k in range(cfg.N_SHADOW):
        feat_path = os.path.join(out_dir, f"shadow_k{k}_features.pt")

        if args.start_phase <= 3 or not os.path.exists(feat_path):
            X_k, y_k = finetune_and_extract_shadow(
                k=k,
                dataset=dataset,
                surrogate_data=surrogate_data,
                base_model=base_model,
                out_dir=out_dir,
                device=device,
                dtype=dtype,
                smoke=args.smoke,
            )
        else:
            print(f"\nShadow {k}: loading cached features from {feat_path}")
            saved = torch.load(feat_path, weights_only=False)
            X_k, y_k = saved["X"], saved["y"]

        all_X.append(X_k)
        all_y.append(y_k)

    # Free base model after all extractions complete
    print("\nFreeing base model from GPU...")
    free_gpu(base_model)

    # ----------------------------------------------------------------
    # Pool shadow features across K=3 models
    # ----------------------------------------------------------------
    X_shadow_46 = torch.cat(all_X, dim=0)   # [3000, 46]
    y_shadow    = torch.cat(all_y, dim=0)   # [3000]

    shadow_feat_path = os.path.join(out_dir, "shadow_features.pt")
    torch.save({
        "X_46":  X_shadow_46,
        "X_8":   X_shadow_46[:, list(range(0, 4)) + list(range(13, 17))],  # [3000, 8]
        "X_attn": X_shadow_46[:, list(range(29, 45))],                      # [3000, 16]
        "X_30":  X_shadow_46[:, list(range(0, 29)) + [45]],                 # [3000, 30]
        "y":     y_shadow,
        "n_shadow": cfg.N_SHADOW,
        "signal_names": build_signal_names(cfg.T_STEPS),
    }, shadow_feat_path)

    wandb.log({
        "shadow_pool/X_shape": list(X_shadow_46.shape),
        "shadow_pool/y_balance": float(y_shadow.float().mean().item()),
    })
    print(f"\nPooled shadow features: {X_shadow_46.shape}, balance={y_shadow.float().mean():.2%}")

    # ----------------------------------------------------------------
    # Phase 4 — Classifier training + distribution analysis + transfer
    # ----------------------------------------------------------------
    if args.start_phase <= 4:
        print("\n[Phase 4] Classifier training and transfer evaluation...")
        phase4_train_and_transfer(
            dataset=dataset,
            out_dir=out_dir,
            run3_results=args.run3_results,
            smoke=args.smoke,
        )

    wandb.finish()
    print(f"\nRun4 complete for domain: {dataset}")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
