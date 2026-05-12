"""
shadow_classifier.py — Shadow-transfer classifier training and feature analysis.

Called by run_shadow_mia.py Phase 4. Two public functions:

  train_all_conditions(...)  — trains 5 conditions + group-level transfer analysis
  run_distribution_analysis(...) — KDE plots, Hellinger distance, KS-test for all 46 features

Primary ablation: Condition B (46-dim) vs Condition E (30-dim, no attention dims).
If E >= B in transfer AUC, attention features are confirmed noise across shadow→target.

Feature analysis methodology follows arxiv 2601.20125v3:
  - Per-feature KDE overlay (member vs non-member)
  - Hellinger distance and KS-test statistic annotated on each plot
  - SHAP TreeExplainer values logged for the two primary XGBoost models (B and E)
  - SHAP beeswarm + per-feature scalar for wandb sorting
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on remote GPU instances
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("xgboost not found — using sklearn GradientBoostingClassifier")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not installed — SHAP analysis will be skipped")


# ---------------------------------------------------------------------------
# Metric utilities (identical to Run3 benchmark.py)
# ---------------------------------------------------------------------------

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


def bootstrap_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_thresholds: list = None,
    n_bootstraps: int = cfg.N_BOOTSTRAPS,
    seed: int = cfg.BOOTSTRAP_SEED,
) -> dict:
    """AUC-ROC + TPR@FPR with 95% bootstrap CIs. Matches Run3 methodology."""
    if fpr_thresholds is None:
        fpr_thresholds = cfg.FPR_THRESHOLDS
    rng = np.random.RandomState(seed)
    N = len(y_true)
    auc_s = []
    tpr_s = {f: [] for f in fpr_thresholds}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_s.append(roc_auc_score(yb, sb))
        for f in fpr_thresholds:
            tpr_s[f].append(tpr_at_fpr(yb, sb, f))

    def ci(s):
        s = np.array(s) if s else np.array([0.0])
        return float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    out = {}
    out["auc_mean"], out["auc_lo"], out["auc_hi"] = ci(auc_s)
    for f in fpr_thresholds:
        m, lo, hi = ci(tpr_s[f])
        out[f"tpr_at_{f}"]       = m
        out[f"tpr_at_{f}_lo"]    = lo
        out[f"tpr_at_{f}_hi"]    = hi
    return out


def make_xgb():
    if HAS_XGB:
        return XGBClassifier(**cfg.XGB_PARAMS)
    return GradientBoostingClassifier(
        n_estimators=cfg.XGB_PARAMS["n_estimators"],
        max_depth=cfg.XGB_PARAMS["max_depth"],
        learning_rate=cfg.XGB_PARAMS["learning_rate"],
        subsample=cfg.XGB_PARAMS["subsample"],
        random_state=cfg.XGB_PARAMS["random_state"],
    )


# ---------------------------------------------------------------------------
# Per-feature univariate AUC
# ---------------------------------------------------------------------------

def compute_univariate_auc(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    For each feature, compute AUC treating that feature alone as the classifier score.
    Takes max(AUC, 1-AUC) to handle features where lower = member.
    """
    result = {}
    for d, name in enumerate(feature_names):
        score = X[:, d]
        if np.std(score) < 1e-9:
            result[name] = 0.5
            continue
        try:
            auc = roc_auc_score(y, score)
            result[name] = float(max(auc, 1.0 - auc))
        except Exception:
            result[name] = 0.5
    return result


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_scaled: np.ndarray) -> np.ndarray | None:
    """
    Exact SHAP values via TreeExplainer for XGBoost / GradientBoosting.
    Returns array [N, D] of SHAP values for the positive class (member=1).
    Returns None if shap is not installed.
    """
    if not HAS_SHAP:
        return None
    try:
        # XGBoost >=2.0 serialises base_score as a string like '[5E-1]'.
        # Patch it to a float before handing to SHAP to avoid TypeError.
        if hasattr(model, 'base_score'):
            try:
                bs = model.base_score
                if isinstance(bs, str):
                    model.base_score = float(bs.strip('[]'))
            except (ValueError, AttributeError):
                pass
        explainer  = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_out   = explainer.shap_values(X_scaled)
        # XGBoost binary: returns [N, D] directly
        # GradientBoosting: returns [N, D] for regression-like output
        if isinstance(shap_out, list):
            shap_out = shap_out[1]   # positive class
        return np.array(shap_out)
    except Exception as e:
        print(f"  WARNING: SHAP computation failed: {e}")
        return None


def log_shap(shap_values: np.ndarray, feature_names: list, condition_name: str):
    """Log SHAP importance table and beeswarm plot to wandb."""
    if shap_values is None:
        return

    mean_abs = np.abs(shap_values).mean(axis=0)

    # Importance table sorted descending
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    table = wandb.Table(
        columns=["rank", "feature", "mean_abs_shap"],
        data=[[i+1, name, float(val)] for i, (name, val) in enumerate(ranked)],
    )
    wandb.log({f"shap/{condition_name}/importance_table": table})

    # Per-feature scalar for wandb sorting/filtering
    for name, val in zip(feature_names, mean_abs):
        wandb.log({f"shap/{condition_name}/{name}": float(val)})

    # Beeswarm plot (shap built-in)
    if HAS_SHAP:
        try:
            fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.25)))
            shap.summary_plot(
                shap_values, features=None,
                feature_names=feature_names,
                plot_type="dot", show=False,
                max_display=min(30, len(feature_names)),
            )
            fig = plt.gcf()
            fig.tight_layout()
            wandb.log({f"shap/{condition_name}/beeswarm": wandb.Image(fig)})
            plt.close("all")
        except Exception as e:
            print(f"  WARNING: SHAP beeswarm plot failed: {e}")
            plt.close("all")


def log_shap_b_vs_e_delta(shap_B: np.ndarray, shap_E: np.ndarray,
                           names_B: list, names_E: list):
    """
    Compare mean |SHAP| between Condition B (46-dim) and E (30-dim pruned)
    for the 30 shared features. Positive delta means B assigned more weight to
    that feature — if attention features in B have high |SHAP|, they are not noise.
    """
    if shap_B is None or shap_E is None:
        return

    mean_abs_B = dict(zip(names_B, np.abs(shap_B).mean(axis=0)))
    mean_abs_E = dict(zip(names_E, np.abs(shap_E).mean(axis=0)))

    # Only compare the 30 features shared by both conditions
    shared = [n for n in names_E if n in mean_abs_B]
    rows = []
    for name in shared:
        b_val   = mean_abs_B[name]
        e_val   = mean_abs_E[name]
        delta   = b_val - e_val
        rows.append([name, float(b_val), float(e_val), float(delta)])

    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    table = wandb.Table(
        columns=["feature", "shap_B_full46", "shap_E_pruned30", "delta_B_minus_E"],
        data=rows,
    )
    wandb.log({"shap/b_vs_e/delta_table": table})


# ---------------------------------------------------------------------------
# KDE + distribution statistics (core analytical output)
# ---------------------------------------------------------------------------

def hellinger_distance(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    """
    Hellinger distance H = sqrt(1 - BC) where BC is the Bhattacharyya coefficient.
    Approximated via shared histogram bins.
    H = 0 → identical distributions, H = 1 → completely disjoint.
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi - lo < 1e-10:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    bc = float(np.sum(np.sqrt(np.maximum(pa, 0) * np.maximum(pb, 0))) * bin_width)
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def run_distribution_analysis(
    X: np.ndarray,
    y: np.ndarray,
    signal_names: list,
    population: str,         # "shadow" or "target"
    smoke: bool = False,
) -> dict:
    """
    For all 46 features, compute member/non-member distribution statistics
    and log KDE plots to wandb.

    Methodology follows arxiv 2601.20125v3:
      - KDE curves overlaid (member=steelblue, non-member=coral)
      - Annotations: univariate AUC, Hellinger distance, KS-test p-value
      - Separate wandb.Image per feature for easy browsing

    Returns dict of per-feature stats for saving to transfer_results.pt.
    """
    members    = X[y == 1]
    nonmembers = X[y == 0]
    D = X.shape[1]

    summary_rows = []
    per_feature_stats = {}

    for d in range(D):
        name = signal_names[d] if d < len(signal_names) else f"feat_{d}"
        m_vals = members[:, d]
        n_vals = nonmembers[:, d]

        # Skip degenerate features
        if np.std(m_vals) < 1e-9 and np.std(n_vals) < 1e-9:
            per_feature_stats[name] = {"hellinger": 0.0, "ks_stat": 0.0,
                                        "ks_pval": 1.0, "uni_auc": 0.5}
            continue

        # Metrics
        uni_auc = 0.5
        try:
            auc = roc_auc_score(y, X[:, d])
            uni_auc = float(max(auc, 1.0 - auc))
        except Exception:
            pass

        h_dist = hellinger_distance(m_vals, n_vals)
        ks_stat, ks_pval = ks_2samp(m_vals, n_vals)

        per_feature_stats[name] = {
            "member_mean":    float(np.mean(m_vals)),
            "member_std":     float(np.std(m_vals)),
            "nonmember_mean": float(np.mean(n_vals)),
            "nonmember_std":  float(np.std(n_vals)),
            "uni_auc":        uni_auc,
            "hellinger":      h_dist,
            "ks_stat":        float(ks_stat),
            "ks_pval":        float(ks_pval),
        }

        summary_rows.append([
            name,
            float(np.mean(m_vals)), float(np.std(m_vals)),
            float(np.mean(n_vals)), float(np.std(n_vals)),
            uni_auc, h_dist, float(ks_stat), float(ks_pval),
        ])

        # Log scalars
        wandb.log({
            f"dist/{population}/{name}/uni_auc":   uni_auc,
            f"dist/{population}/{name}/hellinger":  h_dist,
            f"dist/{population}/{name}/ks_stat":   float(ks_stat),
            f"dist/{population}/{name}/ks_pval":   float(ks_pval),
        })

        # KDE plot
        if smoke and d >= 8:
            continue   # in smoke mode only plot first 8 features

        try:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            sns.kdeplot(m_vals, ax=ax, label="member",
                        fill=True, alpha=0.4, color="steelblue", linewidth=1.5)
            sns.kdeplot(n_vals, ax=ax, label="non-member",
                        fill=True, alpha=0.4, color="coral",     linewidth=1.5)
            ax.set_title(
                f"{name}\nUni-AUC={uni_auc:.3f}  H={h_dist:.3f}  KS-p={ks_pval:.2e}",
                fontsize=9,
            )
            ax.set_xlabel("feature value", fontsize=8)
            ax.set_ylabel("density",       fontsize=8)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            wandb.log({f"dist/{population}/{name}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"  WARNING: KDE plot failed for {name}: {e}")
            plt.close("all")

    # Summary table
    wandb.log({f"dist/{population}/summary_table": wandb.Table(
        columns=["feature", "member_mean", "member_std", "nonmember_mean", "nonmember_std",
                 "uni_auc", "hellinger", "ks_stat", "ks_pval"],
        data=summary_rows,
    )})

    return per_feature_stats


# ---------------------------------------------------------------------------
# Single condition: train on shadow, evaluate on target
# ---------------------------------------------------------------------------

def train_condition(
    condition_name: str,
    feat_indices: list,
    X_shadow: np.ndarray,    # [3000, 46]
    y_shadow: np.ndarray,    # [3000]
    X_target: np.ndarray,    # [2000, 46]
    y_target: np.ndarray,    # [2000]
    signal_names: list,      # 46 names
    oracle: bool = False,    # if True: 5-fold CV on target (oracle mode)
    smoke: bool = False,
) -> dict:
    """
    Non-oracle (B/C/D/E):
      - Slice features by feat_indices
      - Fit StandardScaler on shadow, transform both shadow and target
      - Train XGBoost on full shadow set, predict_proba on target
      - No cross-validation (shadow and target are from different models)

    Oracle (A):
      - 5-fold CV on target features (replays Run3 classifier)
    """
    feat_names = [signal_names[i] for i in feat_indices if i < len(signal_names)]
    n_feat = len(feat_indices)

    try:
        X_sh = X_shadow[:, feat_indices]
        X_ta = X_target[:, feat_indices]

        scaler  = StandardScaler()
        X_sh_s  = scaler.fit_transform(X_sh)
        X_ta_s  = scaler.transform(X_ta)

        if oracle:
            # 5-fold CV on target to get OOF probabilities (oracle upper bound)
            n_folds = 3 if smoke else 5
            skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            oof     = np.zeros(len(y_target))
            for tr, va in skf.split(X_ta_s, y_target):
                sc_fold = StandardScaler()
                clf     = make_xgb()
                clf.fit(sc_fold.fit_transform(X_ta_s[tr]), y_target[tr])
                oof[va] = clf.predict_proba(sc_fold.transform(X_ta_s[va]))[:, 1]
            probs  = oof
            # Also train on full target for SHAP
            clf_full = make_xgb()
            clf_full.fit(X_ta_s, y_target)
            shap_vals = compute_shap_values(clf_full, X_ta_s)
        else:
            clf = make_xgb()
            clf.fit(X_sh_s, y_shadow)
            probs     = clf.predict_proba(X_ta_s)[:, 1]
            shap_vals = compute_shap_values(clf, X_ta_s)

        metrics = bootstrap_metrics(y_target, probs,
                                    n_bootstraps=100 if smoke else cfg.N_BOOTSTRAPS)
        uni_auc = compute_univariate_auc(X_ta, y_target, feat_names)

        # Log SHAP for conditions B and E only (primary ablation pair)
        if condition_name in ("B_full46", "E_pruned30", "A_oracle"):
            log_shap(shap_vals, feat_names, condition_name)

        return {
            "condition":    condition_name,
            "n_features":   n_feat,
            "feature_names": feat_names,
            "probs":        probs,
            "metrics":      metrics,
            "shap_values":  shap_vals,
            "univariate_auc": uni_auc,
        }

    except Exception as e:
        print(f"  ERROR: condition {condition_name} failed: {e}")
        return {
            "condition":   condition_name,
            "n_features":  n_feat,
            "feature_names": feat_names,
            "probs":       np.full(len(y_target), 0.5),
            "metrics":     {"auc_mean": 0.5, "auc_lo": 0.5, "auc_hi": 0.5},
            "shap_values": None,
            "univariate_auc": {},
            "error":       str(e),
        }


# ---------------------------------------------------------------------------
# Group-level transfer analysis (13 groups)
# ---------------------------------------------------------------------------

def run_group_transfer(
    X_shadow: np.ndarray,
    y_shadow: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    smoke: bool,
) -> dict:
    """
    For each of the 13 signal groups, train XGBoost on shadow features for that
    group, evaluate transfer AUC on target. Compare to Run3 solo AUC to measure
    degradation. Expected: attention groups degrade > 0.10, ELBO/entropy < 0.05.
    """
    group_aucs = {}
    for group_name, slc in cfg.GROUP_SLICES.items():
        try:
            X_sh_g = X_shadow[:, slc]
            X_ta_g = X_target[:, slc]

            if X_sh_g.shape[1] == 0:
                group_aucs[group_name] = 0.5
                continue

            sc  = StandardScaler()
            clf = make_xgb()
            clf.fit(sc.fit_transform(X_sh_g), y_shadow)
            probs = clf.predict_proba(sc.transform(X_ta_g))[:, 1]
            auc   = float(roc_auc_score(y_target, probs))
            group_aucs[group_name] = max(auc, 1.0 - auc)

        except Exception as e:
            print(f"  WARNING: group {group_name} transfer failed: {e}")
            group_aucs[group_name] = 0.5

    return group_aucs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_all_conditions(
    X_shadow: np.ndarray,
    y_shadow: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    oracle_auc: float,
    run3_solo_auc: dict,
    signal_names: list,
    smoke: bool = False,
) -> tuple[dict, dict]:
    """
    Train all 5 conditions and run group-level transfer analysis.

    Conditions:
      A_oracle       — 5-fold CV on target (replays Run3, oracle upper bound)
      B_full46       — shadow → target, all 46 dims  [PRIMARY BASELINE]
      C_elbo_entropy8 — shadow → target, 8 permutation-invariant dims
      D_attn16       — shadow → target, attention-only control (expect ~0.50)
      E_pruned30     — shadow → target, 30 dims (no attention) [PRIMARY ABLATION]

    Returns:
      condition_results: dict[cname → result dict]
      group_results:     dict[group_name → transfer_auc]
    """
    condition_results = {}

    for cname, feat_idx in cfg.SHADOW_CONDITIONS.items():
        is_oracle = (cname == "A_oracle")
        print(f"  Training condition {cname} ({len(feat_idx)} features)...")
        result = train_condition(
            condition_name=cname,
            feat_indices=feat_idx,
            X_shadow=X_shadow,
            y_shadow=y_shadow,
            X_target=X_target,
            y_target=y_target,
            signal_names=signal_names,
            oracle=is_oracle,
            smoke=smoke,
        )
        condition_results[cname] = result
        m = result["metrics"]
        auc_val = m.get("auc_mean", m.get("auc", 0.5))
        print(f"    {cname}: AUC={auc_val:.4f} [{m.get('auc_lo', 0):.4f}, {m.get('auc_hi', 0):.4f}]")

    # B vs E SHAP delta — the key cross-condition comparison
    shap_B = condition_results.get("B_full46",   {}).get("shap_values")
    shap_E = condition_results.get("E_pruned30", {}).get("shap_values")
    names_B = condition_results.get("B_full46",   {}).get("feature_names", [])
    names_E = condition_results.get("E_pruned30", {}).get("feature_names", [])
    log_shap_b_vs_e_delta(shap_B, shap_E, names_B, names_E)

    # 13-group transfer analysis
    print("  Running group-level transfer analysis (13 groups)...")
    group_results = run_group_transfer(X_shadow, y_shadow, X_target, y_target, smoke)

    return condition_results, group_results
