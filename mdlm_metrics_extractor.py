"""
mdlm_metrics_extractor.py
=========================
Production-grade metrics extraction harness for MDLM / dLLM models.

Extracts every signal from the MIA signal taxonomy (see Table in the SAMA
paper + the local signal-catalogue PDF), organised by how the signal is
computed and what it measures.  AttenMIA-inspired attention signals have
been added as first-class citizens of the feature set.

  Signal category             | What we capture
  ----------------------------|--------------------------------------------------
  ELBO loss trajectory        | Per-timestep L(t) across t ∈ [0,1]
  Loss curve shape            | dL/dt, d²L/dt² (finite differences over t grid)
  ELBO variance               | Var_t[L(t)] — consistency of reconstruction
  Score / logit entropy       | H(p_θ(·|z_t)) per masked token at each t
  Multi-mask consistency      | Var of predictions across K mask configs at same t
  Hidden states               | Per-layer hidden-state norms + cosine geometry
  Gradient norms              | ‖∇_θ L(x₀, t)‖ at selected t values
  ── AttenMIA-inspired ────────────────────────────────────────────────────────
  Attention entropy           | Per-head H(A_lh) at each layer × timestep
                              |   Members → concentrated attention → low entropy
                              |   Non-members → diffuse attention → high entropy
  Cross-layer correlation     | Pearson correlation between flattened attention
                              |   maps of consecutive layers, tracked over t.
                              |   Members → stable, structured cross-layer signal
  Barycentric drift           | Centre-of-mass of the row-mean attention vector
                              |   over token positions, tracked over t.
                              |   Members → smoother drift as masking increases
  Attention perturbation      | Variance of per-head entropy across K mask configs
                              |   (the DLM analogue of AttenMIA's active attack).
                              |   Members → smaller variance (robust attention)

DLM-specific novelty vs AttenMIA (which targets AR models):
  AttenMIA gets ONE attention snapshot per sentence.
  Here we get T snapshots — one per masking ratio — producing a full
  attention trajectory across noise levels.  This is unique to DLMs and
  constitutes a novel signal dimension not available in any prior work.

All signals are captured inside T × K forward passes (no extra passes
beyond those already needed for ELBO/entropy), returning a typed
MetricsBundle dataclass that can be serialised to disk or consumed
downstream by an attack classifier.

Final classifier feature vector (after aggregation)
----------------------------------------------------
  Group                         | Dim
  ------------------------------|-----
  ELBO trajectory L(t)          |  T
  ELBO variance                 |  1
  dL/dt, d²L/dt²               |  2T
  Prediction entropy per t      |  T
  Multi-mask consistency        |  T
  Hidden state norm per t       |  T   (mean over layers)
  Hidden cosine sim per t       |  T   (mean over layers)
  Attn entropy per t            |  T   (mean over layers × heads)
  Attn cross-layer corr per t   |  T   (mean over consecutive layer pairs)
  Barycentric drift per t       |  T   (L2 norm of CoM vector)
  Attn perturbation var per t   |  T   (mean over layers × heads)
  ------------------------------|-----
  Total (T=16)                  | 11T + 1 = 177 scalars
                                |   → 31 from original plan + 80 from AttenMIA
                                |   (generic: 11*T + 1 for any T)

Usage
-----
    python mdlm_metrics_extractor.py
    python mdlm_metrics_extractor.py --text "The patient was diagnosed with..." \\
                                      --timesteps 10 --mask_configs 8 --save metrics.pt

Architecture note
-----------------
The script does NOT modify the generation loop in mdlm_qwen3_test.py.
Instead, it re-uses load_model_and_tokenizer() and select_device() from that
module, adding a thin instrumentation layer on top.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ── Import helpers from the existing test harness ─────────────────────────────
try:
    from mdlm_qwen3_test import (
        load_model_and_tokenizer,
        select_device,
        model_dtype,
    )
except ModuleNotFoundError:
    import importlib.util
    _here = Path(__file__).parent
    _spec = importlib.util.spec_from_file_location(
        "mdlm_qwen3_test", _here / "mdlm_qwen3_test.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_model_and_tokenizer = _mod.load_model_and_tokenizer
    select_device            = _mod.select_device
    model_dtype              = _mod.model_dtype


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Typed result bundle
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricsBundle:
    """
    Container for all extracted signals for a single input sequence.

    Shapes use the following abbreviations:
        T   = number of timestep samples (--timesteps)
        L   = sequence length (prompt tokens only, no padding)
        K   = number of independent mask configurations (--mask_configs)
        Nl  = number of transformer layers
        Nh  = number of attention heads
    """

    # ── Input metadata ─────────────────────────────────────────────────────
    text:          str                        # raw input string
    token_ids:     list[int]                  # tokenised input (no padding)
    seq_len:       int                        # L

    # ── ELBO loss trajectory  [T] ──────────────────────────────────────────
    timestep_grid: torch.Tensor               # t values in [0,1], shape [T]
    elbo_per_t:    torch.Tensor               # L(t) at each t,     shape [T]

    # ── Loss curve shape  [T] ─────────────────────────────────────────────
    dldt:          torch.Tensor               # dL/dt  (central finite diff) [T]
    d2ldt2:        torch.Tensor               # d²L/dt² [T]

    # ── ELBO variance (scalar) ─────────────────────────────────────────────
    elbo_variance: float                      # Var_t[L(t)]

    # ── Per-token prediction entropy  [T] ─────────────────────────────────
    # H(p_θ(·|z_t)) averaged over masked tokens at each timestep
    pred_entropy_per_t: torch.Tensor          # shape [T]
    # Full per-token, per-timestep matrix (for detailed analysis)
    pred_entropy_full:  torch.Tensor          # shape [T, L]

    # ── Multi-mask consistency  [T] ───────────────────────────────────────
    # Fraction of positions where all K configs agree on the same argmax
    mask_consistency_per_t: torch.Tensor      # shape [T]

    # ── Attention patterns  [T, Nl, L, L] ────────────────────────────────
    # Head-averaged attention maps; one snapshot per timestep
    attention_maps: Optional[torch.Tensor]    # shape [T, Nl, L, L] or None

    # ── AttenMIA: per-head attention entropy  [T, Nl, Nh] ─────────────────
    # H(A_lh) for each layer l and head h, at each timestep t.
    # Low entropy → concentrated attention → membership signal.
    # Aggregated mean stored in attn_entropy_mean_per_t [T].
    attn_entropy_full:    Optional[torch.Tensor]  # [T, Nl, Nh] or None
    attn_entropy_per_t:   Optional[torch.Tensor]  # [T] mean over (Nl, Nh)

    # ── AttenMIA: cross-layer attention correlation  [T, Nl-1] ────────────
    # Pearson r between flattened attention maps of consecutive layer pairs
    # (l, l+1), at each timestep t.  Tracks how consistently the model routes
    # information across depth as masking increases.
    # Aggregated mean stored in attn_crosslayer_mean_per_t [T].
    attn_crosslayer_corr: Optional[torch.Tensor]  # [T, Nl-1] or None
    attn_crosslayer_per_t: Optional[torch.Tensor] # [T] mean over layer pairs

    # ── AttenMIA: barycentric drift  [T, L] ───────────────────────────────
    # The "centre of mass" of the row-mean attention distribution over token
    # positions, tracked across masking ratios.  Concretely:
    #   CoM(t) = Σ_i i * ā(t,i)   where ā(t,i) = mean_l mean_h A_lh(t)[i,:]
    # Members show smoother CoM trajectories as t increases.
    # attn_barycenter_per_t stores ‖CoM(t)‖ as a scalar summary.
    attn_barycenter_per_t: Optional[torch.Tensor] # [T] or None

    # ── AttenMIA: attention perturbation variance  [T] ────────────────────
    # Variance of per-head attention entropy across K mask configurations at
    # each timestep.  This is the DLM analogue of AttenMIA's "active attack":
    # perturbing members (different mask configs) causes smaller attention
    # shifts than perturbing non-members.
    attn_perturbation_per_t: Optional[torch.Tensor]  # [T] or None

    # ── Hidden state geometry  [T, Nl] ────────────────────────────────────
    hidden_norms:      torch.Tensor           # ‖h_l‖₂ per layer, shape [T, Nl]
    hidden_cosine_sim: torch.Tensor           # cos-sim to t=0 hidden, [T, Nl]

    # ── Gradient norms  [T_grad] ──────────────────────────────────────────
    # Computed only at a small subset of timesteps (expensive)
    grad_t_grid:   torch.Tensor               # subset of timestep_grid
    grad_norms:    torch.Tensor               # ‖∇_θ L‖₂ per param group [T_grad]

    # ── Timing ────────────────────────────────────────────────────────────
    elapsed_seconds: float

    def to_feature_vector(self) -> torch.Tensor:
        """
        Flatten all signals into a single 1-D feature vector suitable for
        feeding into a downstream classifier (logistic regression, GBDT, etc.).

        Returns a CPU float32 tensor of shape [D] where D = 11*T + 1.
        Gracefully handles missing attention fields (fills with zeros).
        """
        T = self.elbo_per_t.shape[0]
        device = "cpu"

        def _safe(x: Optional[torch.Tensor], shape: tuple) -> torch.Tensor:
            if x is not None:
                return x.float().cpu().flatten()
            return torch.zeros(shape, dtype=torch.float32)

        parts = [
            self.elbo_per_t.float().cpu(),                    # T
            torch.tensor([self.elbo_variance], dtype=torch.float32),  # 1
            self.dldt.float().cpu(),                          # T
            self.d2ldt2.float().cpu(),                        # T
            self.pred_entropy_per_t.float().cpu(),            # T
            self.mask_consistency_per_t.float().cpu(),        # T
            # Hidden: mean over layers → [T]
            self.hidden_norms.float().cpu().mean(dim=-1),     # T
            self.hidden_cosine_sim.float().cpu().mean(dim=-1),# T
            # AttenMIA signals
            _safe(self.attn_entropy_per_t, (T,)),             # T
            _safe(self.attn_crosslayer_per_t, (T,)),          # T
            _safe(self.attn_barycenter_per_t, (T,)),          # T
            _safe(self.attn_perturbation_per_t, (T,)),        # T
        ]
        return torch.cat(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_masking(
    token_ids:  torch.Tensor,       # [1, L]
    mask_id:    int,
    mask_ratio: float,              # fraction of tokens to mask ∈ (0,1]
    rng:        Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Randomly mask a `mask_ratio` fraction of positions in token_ids.
    Returns a new [1, L] tensor with selected positions replaced by mask_id.
    """
    L = token_ids.size(1)
    n_mask = max(1, int(round(mask_ratio * L)))
    perm = torch.randperm(L, generator=rng, device=token_ids.device)
    mask_positions = perm[:n_mask]
    z_t = token_ids.clone()
    z_t[0, mask_positions] = mask_id
    return z_t


def _patch_model_config_for_outputs(model: AutoModelForMaskedLM) -> None:
    """
    Some custom trust_remote_code models gate hidden-state and attention
    output on config flags rather than on forward-pass kwargs.
    Patching config before the first call ensures they are always emitted.
    This is a no-op on well-behaved models.

    Also downgrades SDPA → eager attention, because SDPA never returns
    attention weight tensors (out.attentions is always None with SDPA).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    if getattr(cfg, '_attn_implementation', None) == 'sdpa':
        cfg._attn_implementation = 'eager'
    for attr in ("output_hidden_states", "output_attentions"):
        if not getattr(cfg, attr, False):
            setattr(cfg, attr, True)
    # Some architectures nest the real transformer under .model / .bert / etc.
    for child_name in ("model", "transformer", "bert", "roberta", "encoder"):
        child = getattr(model, child_name, None)
        if child is not None and hasattr(child, "config"):
            if getattr(child.config, '_attn_implementation', None) == 'sdpa':
                child.config._attn_implementation = 'eager'
            for attr in ("output_hidden_states", "output_attentions"):
                if not getattr(child.config, attr, False):
                    setattr(child.config, attr, True)


def _forward_with_hooks(
    model:             AutoModelForMaskedLM,
    input_ids:         torch.Tensor,          # [1, L]
    capture_attentions: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Run one forward pass and return:
        logits        : [1, L, V]
        hidden_states : list of Nl tensors, each [1, L, d_model]
        attentions    : list of Nl tensors, each [1, Nh, L, L], or []

    Strategy
    --------
    1. Patch model.config so custom models honour the output flags.
    2. Call model() with output_hidden_states=True / output_attentions=True.
    3. If the model still returns None for hidden_states (some trust_remote_code
       models fully ignore kwargs), fall back to PyTorch forward hooks
       registered on every nn.Module whose output matches the expected shape.
    """
    _patch_model_config_for_outputs(model)

    out = model(
        input_ids,
        output_hidden_states=True,
        output_attentions=capture_attentions,
    )
    logits = out.logits  # [1, L, V]

    needs_hook_hidden = out.hidden_states is None
    needs_hook_attn   = capture_attentions and (out.attentions is None)

    if needs_hook_hidden or needs_hook_attn:
        L = input_ids.shape[1]
        captured_hidden: list[torch.Tensor] = []
        captured_attn:   list[torch.Tensor] = []
        hooks = []

        def _make_hook(h_store: list, a_store: list):
            def _hook(module, inp, output):
                # Collect every tensor in the output — attention modules return
                # (attn_output [B,L,d], attn_weights [B,Nh,L,L], ...) so we
                # must inspect all elements, not just output[0].
                if isinstance(output, torch.Tensor):
                    candidates = [output]
                elif isinstance(output, tuple):
                    candidates = [o for o in output
                                  if isinstance(o, torch.Tensor) and o.is_floating_point()]
                else:
                    return
                for t in candidates:
                    # Hidden state: [batch, L, d_model]
                    if (needs_hook_hidden and t.ndim == 3
                            and t.shape[0] == input_ids.shape[0]
                            and t.shape[1] == L):
                        h_store.append(t.detach())
                    # Attention weights: [batch, Nh, L, L]
                    if (needs_hook_attn and t.ndim == 4
                            and t.shape[0] == input_ids.shape[0]
                            and t.shape[2] == L and t.shape[3] == L):
                        a_store.append(t.detach())
            return _hook

        for module in model.modules():
            hooks.append(module.register_forward_hook(
                _make_hook(captured_hidden, captured_attn)
            ))

        # Must pass output_attentions=True so attention weights are actually
        # computed (config patch is not always honoured by trust_remote_code models).
        model(input_ids, output_attentions=needs_hook_attn)
        for h in hooks:
            h.remove()

        def _dedup(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            seen, out_list = set(), []
            for t in tensors:
                key = (t.data_ptr(), t.shape)
                if key not in seen:
                    seen.add(key)
                    out_list.append(t)
            return out_list

        hidden_states = (
            _dedup(captured_hidden) if needs_hook_hidden
            else list(out.hidden_states)
        )
        attentions = (
            _dedup(captured_attn) if needs_hook_attn
            else (list(out.attentions) if out.attentions is not None else [])
        )
    else:
        hidden_states = list(out.hidden_states)
        attentions = (
            list(out.attentions)
            if capture_attentions and out.attentions is not None
            else []
        )

    return logits, hidden_states, attentions


# ──────────────────────────────────────────────────────────────────────────────
# Original signal helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _token_entropy(logits: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token Shannon entropy H(softmax(logits)) at masked positions.

    Args:
        logits           : [1, L, V]
        masked_positions : bool tensor [L] — True where token was masked

    Returns:
        entropies : [L] — entropy at each position (0 at non-masked positions)
    """
    probs     = F.softmax(logits[0], dim=-1)         # [L, V]
    log_probs = F.log_softmax(logits[0], dim=-1)     # [L, V]
    H = -(probs * log_probs).sum(dim=-1)             # [L]
    return H * masked_positions.float()


def _layer_norms(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute mean ‖h‖₂ over token dimension for each layer.

    Handles two tensor shapes emitted by different model variants:
        [1, L, d]  — standard HuggingFace output_hidden_states format
        [L, d]     — hook-captured tensors (no batch dimension)

    Returns:
        norms : [Nl]
    """
    norms = []
    for h in hidden_states:
        h2d = h[0] if h.ndim == 3 else h   # [L, d]
        norms.append(h2d.norm(dim=-1).mean())
    return torch.stack(norms)


def _attention_mean(attentions: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Average attention weights over heads for each layer.

    Args:
        attentions : list of Nl tensors, each [1, Nh, L, L]

    Returns:
        avg_attn : [Nl, L, L], or None if list is empty
    """
    if not attentions:
        return None
    return torch.stack([a[0].mean(dim=0) for a in attentions])  # [Nl, L, L]


# ──────────────────────────────────────────────────────────────────────────────
# AttenMIA signal helpers  (NEW)
# ──────────────────────────────────────────────────────────────────────────────

def _attention_entropy_per_head(attentions: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Compute the Shannon entropy of each attention head's distribution over
    key positions, averaged over query positions.

    For head h in layer l, the attention matrix is A_lh ∈ R^{L×L} where
    each row is a probability distribution over the L key positions.
    We compute:
        H(A_lh) = mean_{query q} [ -Σ_k A_lh[q,k] log A_lh[q,k] ]

    Low H → concentrated, decisive attention → membership signal.
    High H → diffuse, uncertain attention → non-member signal.

    This is the core per-head signal in AttenMIA, adapted here to the DLM
    setting where we compute it at each masking ratio t rather than once.

    Args:
        attentions : list of Nl tensors, each [1, Nh, L, L]

    Returns:
        head_entropies : [Nl, Nh] or None if attentions is empty
    """
    if not attentions:
        return None

    layer_entropies = []
    for attn_layer in attentions:                     # [1, Nh, L, L]
        A = attn_layer[0]                             # [Nh, L, L]
        # Clamp for numerical safety before log
        A_safe = A.clamp(min=1e-9)
        # Row-wise entropy: -Σ_k p log p, then mean over query positions
        H = -(A_safe * A_safe.log()).sum(dim=-1)      # [Nh, L]
        H_mean = H.mean(dim=-1)                       # [Nh]
        layer_entropies.append(H_mean)

    return torch.stack(layer_entropies)               # [Nl, Nh]


def _cross_layer_attention_correlation(
    attentions: list[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Compute Pearson correlation between the flattened head-averaged attention
    maps of consecutive layer pairs (l, l+1).

    Intuition (from AttenMIA's layer-wise correlation analysis):
    For member sequences, the model has learned a stable, consistent routing
    of information that persists as depth increases.  This manifests as higher
    correlation between adjacent layers' attention patterns.
    For non-members, attention is more erratic across layers.

    In the DLM setting, we compute this at each masking ratio t, producing
    a correlation *trajectory* — a signal not available in any AR-based attack.

    Args:
        attentions : list of Nl tensors, each [1, Nh, L, L]

    Returns:
        corr : [Nl-1] Pearson r for each consecutive layer pair, or None
    """
    if len(attentions) < 2:
        return None

    # Head-average each layer's attention map: [L, L]
    avg_maps = [a[0].mean(dim=0).flatten() for a in attentions]  # list of [L²]

    correlations = []
    for i in range(len(avg_maps) - 1):
        x = avg_maps[i]
        y = avg_maps[i + 1]
        # Pearson r via z-score
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = (x_c.norm() * y_c.norm()).clamp(min=1e-8)
        r = (x_c * y_c).sum() / denom
        correlations.append(r)

    return torch.stack(correlations)                  # [Nl-1]


def _attention_barycenter(attentions: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Compute the barycentric (centre-of-mass) position of the attention
    distribution over token positions.

    Concretely:
        ā(i) = mean_l mean_h mean_q A_lh[q, i]   (attention mass at key pos i)
        CoM  = Σ_i i * ā(i) / Σ_i ā(i)           (scalar: normalised position)

    Returning the scalar CoM value tracks where in the sequence the model
    is focusing its attention.  AttenMIA showed that member sequences have
    a more stable CoM that drifts predictably with masking ratio.

    In the DLM setting, we return this scalar at each t, so the full
    barycenter trajectory across masking levels is stored in [T].

    Args:
        attentions : list of Nl tensors, each [1, Nh, L, L]

    Returns:
        com_scalar : scalar tensor (centre-of-mass position), or None
    """
    if not attentions:
        return None

    # Stack all layers and heads: mean over (Nl, Nh, query) → ā ∈ R^L
    all_attn = torch.stack([a[0] for a in attentions])  # [Nl, Nh, L, L]
    # Mean over Nl, Nh, and query dimension → marginal over key position
    a_bar = all_attn.mean(dim=(0, 1, 2))                 # [L]
    a_bar = a_bar / a_bar.sum().clamp(min=1e-8)          # normalise

    L = a_bar.shape[0]
    positions = torch.arange(L, dtype=a_bar.dtype, device=a_bar.device)
    com = (positions * a_bar).sum()                      # scalar
    return com


def _attention_perturbation_variance(
    per_config_head_entropies: list[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """
    Compute the variance of per-head attention entropy across K mask
    configurations at a fixed timestep t.

    This is the DLM analogue of AttenMIA's "active attack": in AttenMIA,
    they perturb the input and measure how much the attention shifts.  Here,
    each of the K mask configurations is a distinct perturbation of the same
    sentence — so variance of the attention entropy signal across configs
    directly measures perturbation sensitivity.

    Members → small variance (stable attention across mask configs)
    Non-members → large variance (erratic attention, sensitive to masking)

    Args:
        per_config_head_entropies : list of K tensors, each [Nl, Nh] or None
                                    One per mask configuration at a fixed t.

    Returns:
        var_scalar : scalar — mean variance over (Nl, Nh), or None
    """
    valid = [e for e in per_config_head_entropies if e is not None]
    if len(valid) < 2:
        return None

    stacked = torch.stack(valid)       # [K, Nl, Nh]
    # Variance over K configurations, then mean over (Nl, Nh)
    var = stacked.var(dim=0).mean()    # scalar
    return var


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction engine
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_metrics(
    model:              AutoModelForMaskedLM,
    tokenizer:          AutoTokenizer,
    text:               str,
    n_timesteps:        int   = 16,
    n_mask_configs:     int   = 8,
    grad_timesteps:     int   = 4,
    capture_attentions: bool  = True,
    seed:               int   = 42,
    alpha_min:          float = 0.05,   # 5%  masking ratio lower bound
    alpha_max:          float = 0.50,   # 50% masking ratio upper bound
    max_length:         int   = 256,    # accepted for API compatibility; text is truncated externally
) -> MetricsBundle:
    """
    Full metrics extraction for a single input text.

    Procedure
    ---------
    1.  Tokenise the input (no padding; single sequence).
    2.  Define a uniform grid of mask ratios  t ∈ {1/T, 2/T, …, 1}.
        In MDLM the mask ratio IS the continuous time t.
    3.  For each t in the grid:
        a. Sample K independent random mask configurations.
        b. For each configuration run a forward pass capturing logits,
           hidden states, and (optionally) attention weights.
        c. Aggregate across K configurations:
           - mean cross-entropy loss → L(t)
           - mean per-token prediction entropy
           - variance of argmax predictions → multi-mask consistency
           - mean head-averaged attention maps (per layer)       [original]
           - per-head attention entropy [Nl, Nh]                 [AttenMIA]
           - variance of head entropy across K configs           [AttenMIA]
           - cross-layer attention correlation [Nl-1]            [AttenMIA]
           - barycentric CoM scalar                              [AttenMIA]
           - mean hidden state norms
    4.  Compute finite-difference derivatives of L(t) over t.
    5.  Compute gradient norms at a small subset of t values
        (requires grad; handled in a separate with-grad context).
    6.  Return MetricsBundle.

    Args
    ----
    model              : Loaded MDLM model in eval() mode.
    tokenizer          : Corresponding tokenizer.
    text               : Raw input string to analyse.
    n_timesteps        : Number of t values to evaluate (T).
    n_mask_configs     : Independent mask draws per timestep (K).
    grad_timesteps     : How many t values to compute gradients at (expensive).
    capture_attentions : If True, store AttenMIA signals and attention maps.
    seed               : RNG seed for reproducibility.

    Returns
    -------
    MetricsBundle with all signals populated.
    """
    t_start = time.time()
    device  = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    rng     = torch.Generator(device=device).manual_seed(seed)

    # ── 1. Tokenise ──────────────────────────────────────────────────────────
    token_ids = tokenizer.encode(text, return_tensors="pt").to(device)  # [1, L]
    L = token_ids.size(1)
    log.info("Input: %d tokens  |  text[:60]: %r", L, text[:60])

    # ── 2. Timestep grid ─────────────────────────────────────────────────────
    # t=0 → fully observed; t=1 → fully masked.
    # We sample T points uniformly in (0, 1] (avoid t=0 where no mask exists).
    timestep_grid = torch.linspace(alpha_min, alpha_max, n_timesteps)  # [T]

    # ── Accumulators ─────────────────────────────────────────────────────────
    elbo_per_t:             list[float]               = []
    entropy_per_t:          list[float]               = []
    entropy_full_per_t:     list[torch.Tensor]        = []   # each [L]
    consistency_per_t:      list[float]               = []
    attn_per_t:             list[Optional[torch.Tensor]] = [] # each [Nl,L,L]
    hidden_norm_per_t:      list[torch.Tensor]        = []   # each [Nl]

    # AttenMIA accumulators
    attn_entropy_per_t:      list[Optional[torch.Tensor]] = [] # each: mean scalar
    attn_entropy_full_per_t: list[Optional[torch.Tensor]] = [] # each [Nl, Nh]
    attn_crosslayer_per_t:   list[Optional[torch.Tensor]] = [] # each: mean scalar
    attn_barycenter_per_t_:  list[Optional[torch.Tensor]] = [] # each: scalar
    attn_perturbation_per_t_: list[Optional[torch.Tensor]] = []# each: scalar

    # Baseline hidden states at t≈0 (no masking) — used for cosine similarity
    with torch.no_grad():
        _, h0_states, _ = _forward_with_hooks(
            model, token_ids, capture_attentions=False
        )

    hidden_available = len(h0_states) > 0
    if not hidden_available:
        log.warning(
            "Hidden states unavailable from this model — "
            "hidden_norms and hidden_cosine_sim will be zero tensors."
        )
        h0_dirs: list[torch.Tensor] = []
    else:
        h0_dirs = [h[0].mean(dim=0) for h in h0_states]   # list of [d_model]

    cosine_sim_per_t: list[torch.Tensor] = []              # each [Nl]

    # ── 3. Main loop over timesteps ───────────────────────────────────────────
    for t_idx, t in enumerate(timestep_grid.tolist()):
        mask_ratio = t   # in MDLM, t IS the fraction of tokens masked

        # Per-config accumulators
        losses_k:            list[float]                      = []
        entropies_k:         list[torch.Tensor]               = []  # each [L]
        predictions_k:       list[torch.Tensor]               = []  # each [L]
        attn_k:              list[Optional[torch.Tensor]]     = []  # [Nl,L,L]
        hidden_norm_k:       list[torch.Tensor]               = []
        hidden_dir_k:        list[list[torch.Tensor]]         = []  # [K, Nl]

        # AttenMIA per-config accumulators
        head_entropy_k:      list[Optional[torch.Tensor]]     = []  # [Nl, Nh]
        crosslayer_corr_k:   list[Optional[torch.Tensor]]     = []  # [Nl-1]
        barycenter_k:        list[Optional[torch.Tensor]]     = []  # scalar

        for k in range(n_mask_configs):
            z_t = _apply_masking(token_ids, mask_id, mask_ratio, rng=rng)

            logits, hidden, attentions = _forward_with_hooks(
                model, z_t, capture_attentions=capture_attentions
            )  # logits: [1, L, V]

            # ── Cross-entropy loss at masked positions only ───────────────
            masked_pos = (z_t[0] == mask_id)                    # [L]  bool
            if masked_pos.sum() == 0:
                losses_k.append(0.0)
                entropies_k.append(torch.zeros(L, device=device))
                predictions_k.append(token_ids[0].clone())
                attn_k.append(None)
                head_entropy_k.append(None)
                crosslayer_corr_k.append(None)
                barycenter_k.append(None)
                if hidden:
                    hidden_norm_k.append(_layer_norms(hidden))
                    hidden_dir_k.append([h[0].mean(dim=0) for h in hidden])
                continue

            target  = token_ids[0]                              # [L]
            log_p   = F.log_softmax(logits[0], dim=-1)         # [L, V]
            ce_full = F.nll_loss(log_p, target, reduction="none")  # [L]
            ce_masked = ce_full[masked_pos].mean().item()
            losses_k.append(ce_masked)

            # ── Per-token prediction entropy ──────────────────────────────
            H = _token_entropy(logits, masked_pos)              # [L]
            entropies_k.append(H)

            # ── Argmax prediction (for consistency metric) ────────────────
            predictions_k.append(logits[0].argmax(dim=-1))     # [L]

            # ── Head-averaged attention maps (original signal) ────────────
            attn_k.append(_attention_mean(attentions))          # [Nl,L,L] or None

            # ── AttenMIA: per-head attention entropy ──────────────────────
            #   [Nl, Nh] — low entropy = concentrated = membership signal
            he = _attention_entropy_per_head(attentions)
            head_entropy_k.append(he)

            # ── AttenMIA: cross-layer correlation ─────────────────────────
            #   [Nl-1] — high r = stable routing across depth = member signal
            cl = _cross_layer_attention_correlation(attentions)
            crosslayer_corr_k.append(cl)

            # ── AttenMIA: barycentric centre-of-mass ─────────────────────
            #   scalar — tracks where the model is attending
            bary = _attention_barycenter(attentions)
            barycenter_k.append(bary)

            # ── Hidden states ─────────────────────────────────────────────
            if hidden:
                hidden_norm_k.append(_layer_norms(hidden))
                hidden_dir_k.append([h[0].mean(dim=0) for h in hidden])

        # ── Aggregate over K configs ──────────────────────────────────────
        elbo_per_t.append(float(np.mean(losses_k)))

        # Mean prediction entropy over K configs (shape [L])
        ent_mean = torch.stack(entropies_k).mean(dim=0)        # [L]
        entropy_full_per_t.append(ent_mean)
        entropy_per_t.append(ent_mean[ent_mean > 0].mean().item())

        # Multi-mask consistency: fraction of positions where all K configs
        # agree on the same argmax prediction
        preds_tensor = torch.stack(predictions_k)              # [K, L]
        mode_pred = preds_tensor.cpu().mode(dim=0).values.to(preds_tensor.device)
        agree = (preds_tensor == mode_pred.unsqueeze(0)).float().mean(dim=0)
        consistency_per_t.append(agree.mean().item())

        # Mean head-averaged attention maps over K configs
        valid_attn = [a for a in attn_k if a is not None]
        attn_per_t.append(
            torch.stack(valid_attn).mean(dim=0) if valid_attn else None
        )  # [Nl, L, L]

        # ── AttenMIA aggregation ──────────────────────────────────────────

        # (a) Per-head attention entropy: mean over K configs → [Nl, Nh]
        valid_he = [h for h in head_entropy_k if h is not None]
        if valid_he:
            mean_he = torch.stack(valid_he).mean(dim=0)        # [Nl, Nh]
            attn_entropy_full_per_t.append(mean_he)
            attn_entropy_per_t.append(mean_he.mean())          # scalar
        else:
            attn_entropy_full_per_t.append(None)
            attn_entropy_per_t.append(None)

        # (b) Perturbation variance: variance of head entropy across K configs
        #     Small = stable attention under perturbation = member signal
        attn_perturbation_per_t_.append(
            _attention_perturbation_variance(head_entropy_k)
        )

        # (c) Cross-layer correlation: mean over K configs, then mean over pairs
        valid_cl = [c for c in crosslayer_corr_k if c is not None]
        if valid_cl:
            mean_cl = torch.stack(valid_cl).mean(dim=0)        # [Nl-1]
            attn_crosslayer_per_t.append(mean_cl.mean())       # scalar
        else:
            attn_crosslayer_per_t.append(None)

        # (d) Barycenter: mean CoM scalar over K configs
        valid_bary = [b for b in barycenter_k if b is not None]
        if valid_bary:
            mean_bary = torch.stack(valid_bary).mean()         # scalar
            attn_barycenter_per_t_.append(mean_bary)
        else:
            attn_barycenter_per_t_.append(None)

        # ── Hidden state aggregation ──────────────────────────────────────
        if hidden_norm_k and all(h.numel() > 0 for h in hidden_norm_k):
            hidden_norm_per_t.append(
                torch.stack(hidden_norm_k).mean(dim=0)         # [Nl]
            )
        else:
            hidden_norm_per_t.append(torch.zeros(1, device=device))

        if hidden_available and hidden_dir_k and h0_dirs:
            Nl = len(h0_dirs)
            mean_dirs_k = [
                torch.stack([
                    hidden_dir_k[k][l]
                    for k in range(len(hidden_dir_k))
                    if l < len(hidden_dir_k[k])
                ]).mean(dim=0)
                for l in range(Nl)
            ]
            cos_sims = [
                F.cosine_similarity(h0.unsqueeze(0), ht.unsqueeze(0)).item()
                for h0, ht in zip(h0_dirs, mean_dirs_k)
            ]
            cosine_sim_per_t.append(torch.tensor(cos_sims, device=device))
        else:
            cosine_sim_per_t.append(torch.zeros(1, device=device))

        if (t_idx + 1) % max(1, n_timesteps // 5) == 0 or t_idx == 0:
            he_scalar = attn_entropy_per_t[-1]
            log.info(
                "  t=%.3f | L(t)=%.4f | H_pred=%.4f | consistency=%.4f"
                " | H_attn=%s",
                t,
                elbo_per_t[-1],
                entropy_per_t[-1],
                consistency_per_t[-1],
                f"{he_scalar.item():.4f}" if he_scalar is not None else "n/a",
            )

    # ── 4. Convert accumulators to tensors ───────────────────────────────────
    elbo_tensor        = torch.tensor(elbo_per_t,         dtype=torch.float32)
    entropy_tensor     = torch.tensor(entropy_per_t,      dtype=torch.float32)
    entropy_full_mat   = torch.stack(entropy_full_per_t)          # [T, L]
    consistency_tensor = torch.tensor(consistency_per_t,  dtype=torch.float32)
    hidden_norm_mat    = torch.stack(hidden_norm_per_t)           # [T, Nl]
    cosine_sim_mat     = torch.stack(cosine_sim_per_t)            # [T, Nl]

    # Head-averaged attention: [T, Nl, L, L] or None
    attn_tensor = (
        torch.stack(attn_per_t)
        if all(a is not None for a in attn_per_t)
        else None
    )

    def _stack_optional_scalars(lst: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        """Stack a list of optional scalar tensors → [T] or None."""
        valid = [x for x in lst if x is not None]
        if not valid:
            return None
        return torch.stack([x if x.dim() == 0 else x.mean() for x in lst if x is not None])

    # AttenMIA tensors
    attn_entropy_full_tensor = (
        torch.stack([x for x in attn_entropy_full_per_t if x is not None])
        if any(x is not None for x in attn_entropy_full_per_t)
        else None
    )  # [T, Nl, Nh] — gaps possible if some t had no valid attentions
    attn_entropy_mean_tensor     = _stack_optional_scalars(attn_entropy_per_t)
    attn_crosslayer_mean_tensor  = _stack_optional_scalars(attn_crosslayer_per_t)
    attn_barycenter_tensor       = _stack_optional_scalars(attn_barycenter_per_t_)
    attn_perturbation_tensor     = _stack_optional_scalars(attn_perturbation_per_t_)

    # ── 5. Finite-difference derivatives of L(t) over t ─────────────────────
    t_np      = timestep_grid.numpy()
    elbo_np   = elbo_tensor.numpy()
    dldt_np   = np.gradient(elbo_np, t_np)
    d2ldt2_np = np.gradient(dldt_np, t_np)
    dldt   = torch.from_numpy(dldt_np.copy()).float()
    d2ldt2 = torch.from_numpy(d2ldt2_np.copy()).float()

    # ── 6. Gradient norms (requires_grad pass) ───────────────────────────────
    grad_t_indices = np.linspace(0, n_timesteps - 1, grad_timesteps, dtype=int)
    grad_t_grid    = timestep_grid[grad_t_indices]
    grad_norm_list: list[float] = []

    log.info("Computing gradient norms at %d timesteps …", grad_timesteps)
    for t_idx in grad_t_indices:
        t_val = float(timestep_grid[t_idx].item())
        z_t = _apply_masking(token_ids, mask_id, t_val, rng=rng)

        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            logits_g, _, _ = _forward_with_hooks(
                model, z_t, capture_attentions=False
            )
            masked_pos_g = (z_t[0] == mask_id)
            if masked_pos_g.sum() == 0:
                grad_norm_list.append(0.0)
                model.zero_grad()
                for p in model.parameters():
                    p.requires_grad_(False)
                continue

            target_g = token_ids[0]
            log_p_g  = F.log_softmax(logits_g[0], dim=-1)
            loss_g   = F.nll_loss(
                log_p_g, target_g, reduction="none"
            )[masked_pos_g].mean()
            loss_g.backward()

        total_grad_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        ).item()
        grad_norm_list.append(total_grad_norm)

        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(False)

        log.info("  t=%.3f | ‖∇_θ L‖ = %.4f", t_val, total_grad_norm)

    grad_norms = torch.tensor(grad_norm_list, dtype=torch.float32)

    elapsed = time.time() - t_start
    log.info("Metrics extraction complete in %.1fs", elapsed)

    return MetricsBundle(
        text                    = text,
        token_ids               = token_ids[0].tolist(),
        seq_len                 = L,
        timestep_grid           = timestep_grid,
        elbo_per_t              = elbo_tensor,
        dldt                    = dldt,
        d2ldt2                  = d2ldt2,
        elbo_variance           = float(elbo_tensor.var().item()),
        pred_entropy_per_t      = entropy_tensor,
        pred_entropy_full       = entropy_full_mat,
        mask_consistency_per_t  = consistency_tensor,
        attention_maps          = attn_tensor,
        # AttenMIA signals
        attn_entropy_full       = attn_entropy_full_tensor,
        attn_entropy_per_t      = attn_entropy_mean_tensor,
        attn_crosslayer_corr    = attn_crosslayer_mean_tensor,
        attn_crosslayer_per_t   = attn_crosslayer_mean_tensor,
        attn_barycenter_per_t   = attn_barycenter_tensor,
        attn_perturbation_per_t = attn_perturbation_tensor,
        # Hidden states
        hidden_norms            = hidden_norm_mat,
        hidden_cosine_sim       = cosine_sim_mat,
        # Gradients
        grad_t_grid             = grad_t_grid,
        grad_norms              = grad_norms,
        elapsed_seconds         = elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(bundle: MetricsBundle) -> None:
    """
    Pretty-print a human-readable summary of all extracted metrics,
    including the new AttenMIA-inspired attention signals.
    """
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  MDLM Metrics Summary  (+ AttenMIA attention signals)")
    print(f"  Text  : {bundle.text[:70]!r}")
    print(f"  Tokens: {bundle.seq_len}  |  Elapsed: {bundle.elapsed_seconds:.1f}s")
    print(sep)

    print("\n── ELBO trajectory L(t) ──────────────────────────────────────────")
    print(f"  Variance : {bundle.elbo_variance:.6f}")
    for t, l in zip(bundle.timestep_grid.tolist(), bundle.elbo_per_t.tolist()):
        print(f"  t={t:.3f}  L(t)={l:.4f}")

    print("\n── Loss curve shape  dL/dt ───────────────────────────────────────")
    for t, d1, d2 in zip(
        bundle.timestep_grid.tolist(),
        bundle.dldt.tolist(),
        bundle.d2ldt2.tolist(),
    ):
        print(f"  t={t:.3f}  dL/dt={d1:+.4f}  d²L/dt²={d2:+.4f}")

    print("\n── Prediction entropy H(p_θ(·|z_t)) ─────────────────────────────")
    for t, h in zip(bundle.timestep_grid.tolist(), bundle.pred_entropy_per_t.tolist()):
        print(f"  t={t:.3f}  H={h:.4f}")

    print("\n── Multi-mask consistency ────────────────────────────────────────")
    for t, c in zip(bundle.timestep_grid.tolist(), bundle.mask_consistency_per_t.tolist()):
        print(f"  t={t:.3f}  agreement={c:.4f}")

    print("\n── Hidden state ‖h‖₂ (mean over tokens) ─────────────────────────")
    T, Nl = bundle.hidden_norms.shape
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        t_val  = bundle.timestep_grid[t_idx].item()
        norms  = bundle.hidden_norms[t_idx].tolist()
        ns_str = "  ".join(f"L{l}:{n:.2f}" for l, n in enumerate(norms[:6]))
        print(f"  t={t_val:.3f}  {ns_str} ...")

    print("\n── Hidden cosine similarity to t=0 ──────────────────────────────")
    for t_idx in [0, T // 2, T - 1]:
        t_val = bundle.timestep_grid[t_idx].item()
        cos   = bundle.hidden_cosine_sim[t_idx].tolist()
        cs_str = "  ".join(f"L{l}:{c:.3f}" for l, c in enumerate(cos[:6]))
        print(f"  t={t_val:.3f}  {cs_str} ...")

    # ── AttenMIA section ──────────────────────────────────────────────────
    print("\n── [AttenMIA] Per-head attention entropy H(A_lh) ─────────────────")
    print("  Low entropy → concentrated attention → membership signal")
    if bundle.attn_entropy_per_t is not None:
        for t, he in zip(bundle.timestep_grid.tolist(),
                         bundle.attn_entropy_per_t.tolist()):
            print(f"  t={t:.3f}  H_attn={he:.4f}")
    else:
        print("  (not captured — run without --no_attentions)")

    print("\n── [AttenMIA] Cross-layer attention correlation ───────────────────")
    print("  High r → stable routing across depth → membership signal")
    if bundle.attn_crosslayer_per_t is not None:
        for t, r in zip(bundle.timestep_grid.tolist(),
                        bundle.attn_crosslayer_per_t.tolist()):
            print(f"  t={t:.3f}  r_crosslayer={r:.4f}")
    else:
        print("  (not captured — run without --no_attentions)")

    print("\n── [AttenMIA] Barycentric drift (CoM over token positions) ────────")
    print("  Tracks where the model attends; members show smoother drift")
    if bundle.attn_barycenter_per_t is not None:
        for t, b in zip(bundle.timestep_grid.tolist(),
                        bundle.attn_barycenter_per_t.tolist()):
            print(f"  t={t:.3f}  CoM={b:.4f}")
    else:
        print("  (not captured — run without --no_attentions)")

    print("\n── [AttenMIA] Attention perturbation variance ─────────────────────")
    print("  Var of H_attn across K mask configs; small = stable = member")
    if bundle.attn_perturbation_per_t is not None:
        for t, v in zip(bundle.timestep_grid.tolist(),
                        bundle.attn_perturbation_per_t.tolist()):
            print(f"  t={t:.3f}  Var_K[H_attn]={v:.6f}")
    else:
        print("  (not captured — run without --no_attentions)")

    print("\n── Gradient norms ‖∇_θ L‖₂ ──────────────────────────────────────")
    for t, g in zip(bundle.grad_t_grid.tolist(), bundle.grad_norms.tolist()):
        print(f"  t={t:.3f}  ‖∇_θ L‖={g:.4f}")

    if bundle.attention_maps is not None:
        T_a, Nl_a, _, _ = bundle.attention_maps.shape
        print(f"\n── Head-averaged attention maps  shape=[T={T_a}, Nl={Nl_a}, L, L]")
        print("  (stored; use --save to serialise for full inspection)")
    else:
        print("\n── Attention maps: not captured (--no_attentions)")

    # Feature vector summary
    fv = bundle.to_feature_vector()
    print(f"\n── Classifier feature vector ──────────────────────────────────────")
    print(f"  Shape: {fv.shape}  (dim = 11T+1, T={bundle.timestep_grid.shape[0]})")
    print(f"  Norm : {fv.norm():.4f}")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MDLM membership-inference signals from a text input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HuggingFace model identifier.",
    )
    p.add_argument(
        "--text",
        default=(
            "The patient was diagnosed with a rare form of lymphoma and "
            "prescribed an experimental immunotherapy protocol."
        ),
        help="Input text to analyse.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Number of t values to evaluate on the [0,1] grid (T).",
    )
    p.add_argument(
        "--mask_configs",
        type=int,
        default=8,
        help="Independent mask configurations per timestep (K).",
    )
    p.add_argument(
        "--grad_timesteps",
        type=int,
        default=4,
        help="Timesteps at which to compute gradient norms (expensive).",
    )
    p.add_argument(
        "--no_attentions",
        action="store_true",
        help="Skip attention capture and all AttenMIA signals (faster, less memory).",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="If provided, serialise the MetricsBundle to this path via torch.save.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible masking.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = select_device()

    log.info("Loading model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # SDPA attention modules never return attention weights.
    # If attentions are requested, reload with eager attention so the correct
    # class is instantiated from the start.
    if not args.no_attentions:
        log.info(
            "Reloading model with attn_implementation='eager' for "
            "AttenMIA attention capture …"
        )
        from transformers import AutoModelForMaskedLM as _AMML
        dtype = next(model.parameters()).dtype
        model = _AMML.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device).eval()

    bundle = extract_metrics(
        model               = model,
        tokenizer           = tokenizer,
        text                = args.text,
        n_timesteps         = args.timesteps,
        n_mask_configs      = args.mask_configs,
        grad_timesteps      = args.grad_timesteps,
        capture_attentions  = not args.no_attentions,
        seed                = args.seed,
    )

    print_summary(bundle)

    if args.save:
        save_path = Path(args.save)
        torch.save(bundle, save_path)
        log.info("MetricsBundle saved to %s", save_path)


if __name__ == "__main__":
    main()