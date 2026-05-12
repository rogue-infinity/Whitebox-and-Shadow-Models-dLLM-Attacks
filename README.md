# White-box and Shadow Model Membership Inference Attacks on Diffusion LLMs

[![Read the Technical Report](https://img.shields.io/badge/Technical%20Report-Read%20PDF-red?style=for-the-badge&logo=adobeacrobatreader)](https://github.com/rogue-infinity/Whitebox-and-Shadow-Models-dLLM-Attacks/blob/main/Technical_Report.pdf)

This repository contains the code for reproducing the membership inference attack (MIA) experiments on **Masked Diffusion Language Models (MDLMs)** reported in our study. Two complementary attack strategies are implemented:

1. **White-box trajectory attack** — fine-tune the target MDLM on a held-out corpus, extract a 46-dimensional ELBO-trajectory feature vector per text, and train an XGBoost/MLP classifier. Evaluated on the [MIMIR](https://github.com/iamgroot42/mimir) benchmark (6 domains).
2. **Shadow model transfer attack** — train shadow MDLMs on disjoint surrogate data, use their features to train a classifier with *no target membership labels*, then transfer it directly to the target domain.

---

## Results

### Experiment 1 — White-box vs. baselines (6 MIMIR domains, mean ± 95 % CI)

| Method | Access | Mean AUC | TPR @ 1 % FPR | TPR @ 0.1 % FPR |
|--------|--------|----------|---------------|-----------------|
| Loss | black-box | 0.669 | 4.5 % | — |
| Zlib | black-box | 0.688 | 6.2 % | — |
| Ratio (FT/base NLL) | black-box | 0.666 | — | — |
| SAMA (Chen et al.) | grey-box | 0.816 | 6.6 % | — |
| **XGBoost (ours)** | **white-box** | **0.878** | **24.7 %** | **10.2 %** |
| **MLP (ours)** | **white-box** | **0.882** | **25.2 %** | **10.8 %** |

### Experiment 2 — Shadow model transfer (mean across 6 domains)

| Condition | Features | Mean AUC | TPR @ 1 % FPR |
|-----------|----------|----------|---------------|
| A — Oracle (white-box upper bound) | 46-dim | 0.878 | 27.4 % |
| **B — Shadow transfer (ours)** | 46-dim | **0.858** | **19.5 %** |
| C — ELBO + entropy only | 8-dim | 0.843 | 19.2 % |
| D — Attention only (negative control) | 16-dim | 0.525 | 1.6 % |
| E — Pruned (no attention) | 30-dim | 0.853 | 20.0 % |

**Key take-aways:**
- The shadow transfer (B) recovers **~98 %** of oracle white-box performance with no target membership labels.
- ELBO trajectory features transfer across domains; attention features do not (condition D collapses to random).

---

## Background

### Masked Diffusion Language Models (MDLMs)

MDLMs are generative models that learn to reconstruct randomly masked tokens. At training time, a masking ratio `t ~ Uniform(ε, 1)` is sampled and each token is independently replaced with `[MASK]` with probability `t`. The model is trained to minimise the ELBO (cross-entropy on masked positions). We use **`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`** (600 M parameters, Qwen3 backbone) as the target model.

### The MIA Signal

Fine-tuning on a set of member texts causes the model to memorise them: the ELBO for member texts decreases faster and more steeply across masking levels than for unseen texts. The ELBO gap `ΔL = L_base(x) − L_ft(x)` is the core signal.

### 46-Dimensional Feature Vector

Extracted at `T = 4` masking levels `α ∈ {0.05, 0.20, 0.35, 0.50}`:

| Group | Dims | Description |
|-------|------|-------------|
| `elbo_traj` | 4 | ELBO at each α |
| `elbo_var` | 1 | Variance across `K = 8` mask configs |
| `dldt` | 4 | First derivative ∂L/∂α |
| `d2ldt2` | 4 | Second derivative ∂²L/∂α² |
| `pred_entropy` | 4 | Mean token prediction entropy |
| `mask_consistency` | 4 | Argmax stability across two independent masks |
| `hidden_norms` | 4 | ℓ₂ norm of last-layer hidden states |
| `hidden_cosine` | 4 | Cosine similarity of hidden states to α = 0 |
| `attn_entropy` | 4 | Mean attention head entropy |
| `attn_crosslayer` | 4 | Cross-layer attention correlation |
| `attn_barycenter` | 4 | Attention distribution barycentric centre |
| `attn_perturbation` | 4 | Attention sensitivity to masking |
| `cross_model_cos` | 1 | FT vs. base model hidden cosine similarity |
| **Total** | **46** | |

### SAMA Baseline

SAMA (Chen et al., ICLR 2026) is a grey-box attack that performs progressive cumulative masking over `T = 4` timesteps with `N = 128` random token subsets and aggregates NLL scores with harmonic weights. It requires only scalar NLL outputs (no internal model access).

---

## Prerequisites

- **Python 3.10+**
- **CUDA GPU with ≥ 16 GB VRAM** (tested on NVIDIA L4 24 GB and A100 40 GB)
- A [HuggingFace](https://huggingface.co) account — the MIMIR dataset (`iamgroot42/mimir`) requires a gated access token
- (Optional) A [Weights & Biases](https://wandb.ai) account for experiment tracking

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/rogue-infinity/Whitebox-and-Shadow-Models-dLLM-Attacks.git
cd Whitebox-and-Shadow-Models-dLLM-Attacks
```

### 2. Clone external dependencies into the same folder

```bash
# SAMA — official MIA implementation (MIT license)
git clone https://github.com/Stry233/SAMA.git

# dLLM — diffusion LLM training / inference utilities
git clone https://github.com/ZHZisZZ/dllm.git
pip install -e ./dllm
```

### 3. Install Python dependencies

```bash
# Install PyTorch first — match your CUDA version:
# https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the rest:
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
export HF_TOKEN=<your_huggingface_token>       # required: access MIMIR dataset
export WANDB_API_KEY=<your_wandb_api_key>      # optional: experiment tracking
```

> **Never commit these values.** Use a `.env` file (already in `.gitignore`) or a secrets manager.

The model (`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`) is downloaded automatically from HuggingFace Hub on first run. To use a local copy, set:

```bash
export MODEL_PATH=/path/to/local/model
```

---

## Repository Layout

```
.
├── config.py                    # All hyperparameters (single source of truth)
├── prepare_data.py              # Stage 1 — load & tokenise MIMIR data
├── finetune.py                  # Stage 2 — AdamW fine-tune MDLM on member texts
├── verify_memorization.py       # Stage 3 — sanity-check ELBO gap
├── run_sama.py                  # Stage 4 — SAMA grey-box baseline
├── run_attacks.py               # Stage 5 — Loss / Zlib / Ratio black-box baselines
├── mdlm_metrics_extractor.py    # Core — 46-dim feature extraction (used by stages 6 & 4)
├── run_signals.py               # Stage 6 — extract white-box features for all samples
├── train_classifier.py          # Stage 7 — XGBoost / LightGBM / MLP, 5-fold CV
├── benchmark.py                 # Stage 8 — AUC, ablation, LaTeX tables
├── analysis.py                  # Publication plots (ROC, ablation heat-map, UMAP)
│
├── run_shadow_mia.py            # Exp 2 — shadow MIA master script (4 phases)
├── shadow_classifier.py         # Exp 2 — conditions A–E + SHAP analysis
├── aggregate_domains.py         # Exp 2 — cross-domain result aggregation
│
├── run_pipeline.sh              # Run Experiment 1 end-to-end for one domain
├── run_shadow_launcher.sh       # Run Experiment 2 (shadow MIA) per domain
│
├── requirements.txt
└── .gitignore
```

---

## Running Experiment 1 — White-box Attack

Run all 8 stages for a single MIMIR domain:

```bash
bash run_pipeline.sh arxiv
```

Available domains: `arxiv`, `github`, `hackernews`, `pile_cc`, `pubmed_central`, `wikipedia`

To run all 6 domains sequentially:

```bash
for domain in arxiv github hackernews pile_cc pubmed_central wikipedia; do
    bash run_pipeline.sh "$domain"
done
```

Or run individual stages manually:

```bash
python prepare_data.py --dataset arxiv          # Stage 1
python finetune.py --dataset arxiv              # Stage 2
python verify_memorization.py --dataset arxiv   # Stage 3
python run_sama.py --dataset arxiv              # Stage 4
python run_attacks.py --dataset arxiv           # Stage 5
python run_signals.py --dataset arxiv           # Stage 6  (~2–8 h on L4)
python train_classifier.py --dataset arxiv      # Stage 7
python benchmark.py --dataset arxiv             # Stage 8
python analysis.py                              # Generate plots (all domains)
```

**Output per domain** (written to `results/<domain>/`):

| File | Contents |
|------|----------|
| `X.pt` | Feature matrix `[N, 46]` |
| `y.pt` | Labels `[N]` (1 = member) |
| `X_per_group.pt` | Dict of group slices |
| `signal_names.pt` | Human-readable feature names |
| `sama_scores.pt` | SAMA grey-box scores |
| `loss_scores.pt`, `zlib_scores.pt`, `ratio_scores.pt` | Baseline scores |
| `classifier_results.pt` | OOF probabilities + importances |
| `ablation_results.pt` | LOSO + solo AUC per group |
| `benchmark.pt` | Final AUC + TPR@FPR table with bootstrap CIs |

---

## Running Experiment 2 — Shadow Model Transfer Attack

Experiment 2 requires the `results/<domain>/` outputs from Experiment 1 (target features).

```bash
bash run_shadow_launcher.sh
```

Or run a single domain:

```bash
python run_shadow_mia.py --dataset arxiv
python shadow_classifier.py --dataset arxiv
python aggregate_domains.py          # after all 6 domains are done
```

**Output per domain** (written to `shadow_results/<domain>/`):

| File | Contents |
|------|----------|
| `surrogate_data.pt` | K non-overlapping shadow data chunks |
| `shadow_k{k}_features.pt` | Features from shadow model k |
| `shadow_features.pt` | Pooled `[3000, 46]` shadow feature matrix |
| `transfer_results.pt` | AUC + TPR@FPR for conditions A–E |
| `shap_analysis.pt` | SHAP feature importance for shadow classifier |

---

## Hyperparameters

All hyperparameters are centralised in `config.py`. Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_PATH` | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | Base MDLM |
| `MASK_TOKEN_ID` | 151669 | Qwen3 mask token |
| `MAX_LENGTH` | 256 | Tokenised text truncation |
| `FT_LR` | 1e-4 | Fine-tuning learning rate |
| `FT_WD` | 0.01 | Weight decay |
| `FT_BATCH_SIZE` | 8 | Per-GPU batch size |
| `FT_EPOCHS` | 5 | Fine-tuning epochs |
| `T_STEPS` | 4 | Masking levels (α ∈ {0.05, 0.20, 0.35, 0.50}) |
| `SIG_K` | 8 | Independent mask configs per sample |
| `SAMA_N_SUBSETS` | 128 | SAMA random token subsets |
| `SAMA_M_TOKENS` | 10 | Tokens per SAMA subset |
| `N_SHADOW` | 3 | Shadow models per domain |
| `SHADOW_CHUNK_SIZE` | 600 | Members per shadow model |
| `MIMIR_N` | 1000 | Members + non-members per domain |

---

## Ablation Findings

Leave-one-signal-out (LOSO) AUC drop when each group is removed from the full 46-dim vector (mean across 6 domains):

| Group removed | ΔAUC |
|---------------|------|
| `elbo_traj` | **−0.130** |
| `pred_entropy` | −0.021 |
| `elbo_var` | −0.012 |
| `hidden_norms` | −0.009 |
| `attn_entropy` | −0.005 |
| `attn_crosslayer` | −0.004 |
| `attn_barycenter` | −0.004 |
| `attn_perturbation` | −0.003 |

**Conclusion:** The ELBO trajectory alone accounts for the entire improvement over SAMA. Attention features add negligible signal and do not transfer across domains.

---

## Citation

If you use this code, please cite the SAMA paper and acknowledge the MIMIR benchmark:

```bibtex
@inproceedings{chen2026sama,
  title     = {Membership Inference Attacks Against Fine-tuned Diffusion Language Models},
  author    = {Chen, ...},
  booktitle = {ICLR},
  year      = {2026}
}

@article{duan2024mimir,
  title   = {Do Membership Inference Attacks Work on Large Language Models?},
  author  = {Duan, Michael and Suri, Anshuman and Mireshghallah, Niloofar and
             Evans, Sewon and Shi, Weijia and Zettlemoyer, Luke and Tsvetkov,
             Yulia and Choi, Yejin and Evans, David and Hajishirzi, Hannaneh},
  journal = {arXiv:2402.07841},
  year    = {2024}
}
```

---

## Acknowledgements

- [SAMA](https://github.com/Stry233/SAMA) — official codebase for subset-aggregated membership attacks (MIT license)
- [dLLM](https://github.com/ZHZisZZ/dllm) — diffusion LLM utilities
- [MIMIR](https://github.com/iamgroot42/mimir) — membership inference benchmark
- [Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) — target model
