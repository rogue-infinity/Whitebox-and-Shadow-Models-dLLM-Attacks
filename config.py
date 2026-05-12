"""
config.py — Central configuration for Run_4_Shadow_MIA.

Extends Run_3_Full_MIMIR config verbatim, then adds shadow-model attack
parameters at the bottom. All Run3 hyperparameters are preserved unchanged
so that feature extraction and fine-tuning are identical across runs.
"""

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
MIMIR_DATASETS = ["arxiv", "github", "hackernews", "pubmed_central", "wikipedia", "pile_cc"]
NLP_DATASETS   = ["wikitext103", "agnews", "xsum"]
ALL_DATASETS   = MIMIR_DATASETS + NLP_DATASETS

# ---------------------------------------------------------------------------
# Sample sizes
# ---------------------------------------------------------------------------
MIMIR_N = 1_000    # members + non-members per MIMIR domain
NLP_N   = 10_000   # members + non-members per NLP benchmark (Fu et al. 2024)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
MAX_LENGTH = 256   # tokens; sequences shorter than this are discarded

# ---------------------------------------------------------------------------
# MIMIR HuggingFace config
# ---------------------------------------------------------------------------
MIMIR_HF_REPO = "iamgroot42/mimir"
MIMIR_SPLIT   = "ngram_13_0.8"    # Run3 target split — do NOT use for shadow data

# HuggingFace config name for each MIMIR domain (may differ from internal key)
MIMIR_HF_NAME = {
    "arxiv":         "arxiv",
    "github":        "github",
    "hackernews":    "hackernews",
    "pubmed_central": "pubmed_central",
    "wikipedia":     "wikipedia_(en)",   # HF uses "(en)" suffix — Issue 10
    "pile_cc":       "pile_cc",
}

# ---------------------------------------------------------------------------
# NLP benchmark HuggingFace configs (unused in Run4 but kept for consistency)
# ---------------------------------------------------------------------------
NLP_HF_CONFIGS = {
    "wikitext103": {"path": "wikitext", "name": "wikitext-103-raw-v1",
                    "text_col": "text",
                    "member_split": "train", "nonmember_split": "test"},
    "agnews":      {"path": "ag_news",  "name": None,
                    "text_col": "text",
                    "member_split": "train", "nonmember_split": "test"},
    "xsum":        {"path": "EdinburghNLP/xsum", "name": None,
                    "text_col": "document",
                    "member_split": "train", "nonmember_split": "test"},
}

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters — IDENTICAL to Run3, no exceptions
# ---------------------------------------------------------------------------
FT_LR          = 1e-4
FT_WD          = 0.01
FT_BATCH_SIZE  = 8
FT_EPOCHS      = 5

# ---------------------------------------------------------------------------
# Masking schedule shared by SAMA + signal extraction
# ---------------------------------------------------------------------------
T_STEPS   = 4
ALPHA_MIN = 0.05   # 5%  — minimum masking ratio
ALPHA_MAX = 0.50   # 50% — maximum masking ratio

# ---------------------------------------------------------------------------
# SAMA attack parameters (unused in Run4, kept for import compatibility)
# ---------------------------------------------------------------------------
SAMA_N_SUBSETS = 128
SAMA_M_TOKENS  = 10
SAMA_MC        = 4

# ---------------------------------------------------------------------------
# Signal extraction — IDENTICAL to Run3
# ---------------------------------------------------------------------------
SIG_K      = 8   # mask configs for multi-mask consistency
SIG_GRAD_T = 2   # timesteps for gradient norm computation

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
MODEL_PATH = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"

# ---------------------------------------------------------------------------
# Run 4: Shadow Model Attack
# ---------------------------------------------------------------------------

# Surrogate data split — deliberately different from Run3's ngram_13_0.8
# so shadow training data is guaranteed non-overlapping with the target.
SHADOW_MIMIR_SPLIT = "ngram_13_0.2"

# Number of shadow models per domain.
# K=3 → 3×1000 = 3000 shadow training samples for the classifier.
# With 6 L4s running in parallel (one per domain), total wall time ≈ 6.5 hrs.
N_SHADOW = 3

# Per-shadow reproducibility: shadow k uses seed SHADOW_BASE_SEED + k.
SHADOW_BASE_SEED = 42   # k=0 → 42, k=1 → 43, k=2 → 44

# Surrogate data split per shadow model:
#   Each shadow gets SHADOW_CHUNK_SIZE member samples (non-overlapping chunks).
#   Non-members: SHADOW_NONMEMBER_SIZE samples drawn from the OTHER chunks.
SHADOW_CHUNK_SIZE      = 600   # members per shadow model
SHADOW_NONMEMBER_SIZE  = 400   # non-members per shadow model (total per shadow = 1000)

# Minimum ELBO gap (nats) for shadow model sanity check.
# Below this, we log a wandb.alert but do NOT abort.
SHADOW_MIN_ELBO_GAP = 0.3

# ---------------------------------------------------------------------------
# Feature index definitions (T=4 → 46 dims total)
#
# Signal groups and their index ranges (must match mdlm_metrics_extractor.py):
#   elbo_traj         [0:4]    ELBO at α ∈ {0.05, 0.20, 0.35, 0.50}
#   elbo_var          [4:5]    variance of ELBO across mask configs
#   dldt              [5:9]    first derivative ∂L/∂t
#   d2ldt2            [9:13]   second derivative ∂²L/∂t²
#   pred_entropy      [13:17]  entropy of token predictions
#   mask_consistency  [17:21]  fraction of stable argmax predictions
#   hidden_norms      [21:25]  ℓ₂ norm of last hidden state
#   hidden_cosine     [25:29]  cosine similarity of hidden states to t=0
#   attn_entropy      [29:33]  mean attention head entropy          ← low LOSO drop
#   attn_crosslayer   [33:37]  cross-layer attention correlation    ← low LOSO drop
#   attn_barycenter   [37:41]  barycenter of attention distribution ← low LOSO drop
#   attn_perturbation [41:45]  attention sensitivity to masking     ← low LOSO drop
#   cross_model_cos   [45:46]  FT vs base model hidden cosine
# ---------------------------------------------------------------------------

SHADOW_CONDITIONS = {
    # A: oracle — replay Run3 white-box classifier as upper bound
    "A_oracle":        list(range(46)),

    # B: full 46-dim transfer — primary shadow baseline
    "B_full46":        list(range(46)),

    # C: 8-dim permutation-invariant subset (ELBO trajectory + predictive entropy)
    # These are scalar aggregates invariant to attention-head permutation.
    "C_elbo_entropy8": list(range(0, 4)) + list(range(13, 17)),

    # D: attention-only control — expected to fail (~0.50 AUC) because
    # attention weights are not aligned across independently fine-tuned models.
    "D_attn16":        list(range(29, 45)),

    # E: 30-dim pruned — drops all 16 attention dims, keeps everything else.
    # PRIMARY ABLATION: if E >= B, attention dims are confirmed noise in transfer.
    "E_pruned30":      list(range(0, 29)) + [45],
}

# Signal group → slice into the 46-dim feature vector.
# Used for group-level transfer analysis (13 groups × one XGBoost each).
GROUP_SLICES = {
    "elbo_traj":         slice(0,  4),
    "elbo_var":          slice(4,  5),
    "dldt":              slice(5,  9),
    "d2ldt2":            slice(9,  13),
    "pred_entropy":      slice(13, 17),
    "mask_consistency":  slice(17, 21),
    "hidden_norms":      slice(21, 25),
    "hidden_cosine":     slice(25, 29),
    "attn_entropy":      slice(29, 33),
    "attn_crosslayer":   slice(33, 37),
    "attn_barycenter":   slice(37, 41),
    "attn_perturbation": slice(41, 45),
    "cross_model_cos":   slice(45, 46),
}

# ---------------------------------------------------------------------------
# XGBoost hyperparameters — identical to Run3 train_classifier.py
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss",
    random_state=42,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FPR_THRESHOLDS  = [0.001, 0.01, 0.10]
N_BOOTSTRAPS    = 1000
BOOTSTRAP_SEED  = 42

# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
WANDB_PROJECT    = "da5001-mia"
WANDB_RUN_PREFIX = "run4"

# ---------------------------------------------------------------------------
# Paths (relative to the code/ directory, matching JarvisLabs layout)
# ---------------------------------------------------------------------------
# On JarvisLabs the Run3 results are uploaded alongside Run4 code under run3_results/.
# Locally they live at ../Run_3_Full_MIMIR/results/.
RUN3_RESULTS_DEFAULT = "../Run_3_Full_MIMIR/results"
SHADOW_RESULTS_DIR   = "../shadow_results"
