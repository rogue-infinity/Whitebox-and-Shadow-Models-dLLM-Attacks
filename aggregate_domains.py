"""
aggregate_domains.py — Post-hoc cross-domain aggregation for Run_4_Shadow_MIA.

Run this LOCALLY after downloading all 6 domain results from JarvisLabs.

Reads:  ../shadow_results/{domain}/transfer_results.pt  (all 6 domains)
Reads:  ../Run_3_Full_MIMIR/results/{domain}/classifier_results.pt  (oracle AUC)
Reads:  ../Run_3_Full_MIMIR/results/{domain}/sama_scores.pt          (SAMA AUC)

Outputs (stdout + wandb + plots/):
  Table 1:  AUC per condition × domain  (6×5 grid)
  Table 2:  Mean/std AUC per condition  (aggregate)
  Table 3:  Oracle gap per condition    (oracle AUC - shadow transfer AUC)
  Table 3b: TPR at low FPR (0.1%, 1%, 5%, 10%) per condition × domain
  Table 4:  SHAP rank stability         (Spearman r across domain pairs)
  Table 5:  Hellinger distance per feature (member vs non-member, averaged across domains)
  Table 6:  Unified 3-attack comparison (oracle vs SAMA vs shadow B_full46 vs E_pruned30)

  fig_r4_1_condition_comparison.{pdf,png}   grouped bar chart: conditions × domains
  fig_r4_2_oracle_gap.{pdf,png}             bar chart: oracle gap per condition
  fig_r4_3_hellinger_heatmap.{pdf,png}      heatmap: feature × domain Hellinger distance
  fig_r4_4_shap_rank_stability.{pdf,png}    Spearman rank correlation matrix
  fig_r4_5_b_vs_e_delta.{pdf,png}           SHAP delta B−E across domains
  fig_r4_6_tpr_at_1pct_fpr.{pdf,png}        TPR@1%FPR grouped bar chart
  fig_r4_7_unified_comparison.{pdf,png}     4-bar: oracle / SAMA / B_full46 / E_pruned30

Usage:
    python aggregate_domains.py
    python aggregate_domains.py --results_base ../shadow_results --run3_results ../Run_3_Full_MIMIR/results
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


DOMAINS    = cfg.MIMIR_DATASETS
CONDITIONS = list(cfg.SHADOW_CONDITIONS.keys())   # A_oracle … E_pruned30

CONDITION_LABELS = {
    "A_oracle":        "A: Oracle (Run3)",
    "B_full46":        "B: Shadow 46-dim",
    "C_elbo_entropy8": "C: Shadow 8-dim (ELBO+H)",
    "D_attn16":        "D: Shadow Attn-only (ctrl)",
    "E_pruned30":      "E: Shadow 30-dim (pruned)",
}

PALETTE = {
    "A_oracle":        "#2c7bb6",
    "B_full46":        "#d7191c",
    "C_elbo_entropy8": "#fdae61",
    "D_attn16":        "#abdda4",
    "E_pruned30":      "#1a9641",
}

# FPR threshold grid — computed from raw scores for ALL attacks so comparisons are consistent.
# Run3 oracle/SAMA bootstrap CIs only covered [0.001, 0.01, 0.10]; we recompute everything
# exactly from stored probs/scores so 0.005, 0.02, 0.05 are also available for all attacks.
LOW_FPR_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
LOW_FPR_LABELS     = ["TPR@0.1%FPR", "TPR@0.5%FPR", "TPR@1%FPR",
                      "TPR@2%FPR",   "TPR@5%FPR",   "TPR@10%FPR"]


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


def _tpr_row(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute TPR at every LOW_FPR_THRESHOLDS entry from raw scores."""
    return {label: _tpr_at_fpr(y_true, y_score, fpr)
            for fpr, label in zip(LOW_FPR_THRESHOLDS, LOW_FPR_LABELS)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_domain_results(results_base: str, run3_results: str) -> dict:
    """Load transfer_results.pt for each domain. Warn on missing."""
    domain_data = {}
    for domain in DOMAINS:
        path = os.path.join(results_base, domain, "transfer_results.pt")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — domain {domain} skipped")
            continue
        data = torch.load(path, weights_only=False)
        domain_data[domain] = data
        print(f"  Loaded {domain}: {list(data['conditions'].keys())}")
    return domain_data


def load_sama_auc(domain: str, run3_results: str) -> float | None:
    """
    Load SAMA scores from Run3 and compute AUC.
    sama_scores.pt keys: 'scores' [N], 'labels' [N]  (N=128 subsets, T=4 steps)
    Returns None if file absent.
    """
    path = os.path.join(run3_results, domain, "sama_scores.pt")
    if not os.path.exists(path):
        print(f"  WARNING: SAMA scores not found at {path}")
        return None
    try:
        d      = torch.load(path, weights_only=False)
        scores = d["scores"]
        labels = d["labels"]
        scores = scores.numpy() if hasattr(scores, "numpy") else np.array(scores)
        labels = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
        if len(np.unique(labels)) < 2:
            print(f"  WARNING: SAMA labels for {domain} have only one class — AUC undefined")
            return None
        return float(roc_auc_score(labels, scores))
    except Exception as e:
        print(f"  WARNING: failed to load SAMA scores for {domain}: {e}")
        return None


def load_all_sama_aucs(run3_results: str) -> dict:
    """Returns {domain: sama_auc_or_None} for all DOMAINS."""
    sama = {}
    for domain in DOMAINS:
        auc = load_sama_auc(domain, run3_results)
        sama[domain] = auc
        status = f"{auc:.4f}" if auc is not None else "MISSING"
        print(f"  SAMA {domain}: {status}")
    return sama


def load_oracle_raw(domain: str, run3_results: str):
    """
    Load Run3 oracle XGBoost OOF probs and labels.
    Returns (probs np.ndarray, y_true np.ndarray) or (None, None).
    classifier_results.pt keys: xgb_probs [N], y_true [N]
    """
    path = os.path.join(run3_results, domain, "classifier_results.pt")
    if not os.path.exists(path):
        return None, None
    try:
        d = torch.load(path, weights_only=False)
        probs  = d["xgb_probs"]
        y_true = d["y_true"]
        probs  = probs.numpy()  if hasattr(probs,  "numpy") else np.array(probs)
        y_true = y_true.numpy() if hasattr(y_true, "numpy") else np.array(y_true)
        return probs, y_true
    except Exception as e:
        print(f"  WARNING: could not load oracle raw scores for {domain}: {e}")
        return None, None


def load_sama_raw(domain: str, run3_results: str):
    """
    Load SAMA raw scores and labels.
    Returns (scores np.ndarray, labels np.ndarray) or (None, None).
    sama_scores.pt keys: scores [N], labels [N]
    """
    path = os.path.join(run3_results, domain, "sama_scores.pt")
    if not os.path.exists(path):
        return None, None
    try:
        d      = torch.load(path, weights_only=False)
        scores = d["scores"]
        labels = d["labels"]
        scores = scores.numpy() if hasattr(scores, "numpy") else np.array(scores)
        labels = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
        # Flip if AUC < 0.5 (same logic as run_sama.py)
        if len(np.unique(labels)) > 1 and roc_auc_score(labels, scores) < 0.5:
            scores = 1.0 - scores
        return scores, labels
    except Exception as e:
        print(f"  WARNING: could not load SAMA raw scores for {domain}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Table 1 + 2: AUC per condition × domain
# ---------------------------------------------------------------------------

def build_auc_table(domain_data: dict) -> pd.DataFrame:
    """
    Returns DataFrame with columns: domain, condition, auc, auc_lo, auc_hi.
    """
    rows = []
    for domain, data in domain_data.items():
        for cname in CONDITIONS:
            cdata = data["conditions"].get(cname)
            if cdata is None:
                continue
            m    = cdata["metrics"]
            auc  = m.get("auc_mean", m.get("auc", float("nan")))
            lo   = m.get("auc_lo",   float("nan"))
            hi   = m.get("auc_hi",   float("nan"))
            rows.append({"domain": domain, "condition": cname,
                         "auc": auc, "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(rows)


def log_auc_tables(df: pd.DataFrame):
    # Wide table: domain × condition
    pivot = df.pivot(index="domain", columns="condition", values="auc").round(4)
    print("\n=== AUC per Condition × Domain ===")
    print(pivot.to_string())

    # wandb Table 1
    t1_cols = ["domain"] + [CONDITION_LABELS.get(c, c) for c in CONDITIONS]
    t1_data = []
    for domain in pivot.index:
        row = [domain] + [float(pivot.loc[domain, c]) if c in pivot.columns else float("nan")
                          for c in CONDITIONS]
        t1_data.append(row)
    wandb.log({"aggregate/auc_grid": wandb.Table(columns=t1_cols, data=t1_data)})

    # wandb Table 2: mean ± std per condition
    print("\n=== Mean ± Std AUC per Condition ===")
    t2_data = []
    for c in CONDITIONS:
        sub = df[df["condition"] == c]["auc"].dropna()
        mean_auc = float(sub.mean()) if len(sub) else float("nan")
        std_auc  = float(sub.std())  if len(sub) > 1 else 0.0
        print(f"  {CONDITION_LABELS.get(c,c):35s}  {mean_auc:.4f} ± {std_auc:.4f}")
        t2_data.append([CONDITION_LABELS.get(c, c), round(mean_auc, 4),
                         round(std_auc, 4), len(sub)])
        wandb.log({f"aggregate/{c}/mean_auc": mean_auc, f"aggregate/{c}/std_auc": std_auc})
    wandb.log({"aggregate/mean_std_table": wandb.Table(
        columns=["condition", "mean_auc", "std_auc", "n_domains"], data=t2_data)})

    return pivot


# ---------------------------------------------------------------------------
# Table 3: Oracle gap
# ---------------------------------------------------------------------------

def log_oracle_gap(df: pd.DataFrame):
    """oracle_auc (A) - condition_auc for each condition/domain."""
    oracle_df = df[df["condition"] == "A_oracle"][["domain", "auc"]].rename(
        columns={"auc": "oracle_auc"})
    merged = df.merge(oracle_df, on="domain")
    merged["gap"] = merged["oracle_auc"] - merged["auc"]

    print("\n=== Oracle Gap (oracle AUC − condition AUC) ===")
    gap_rows = []
    for c in CONDITIONS:
        sub = merged[merged["condition"] == c]["gap"].dropna()
        mean_gap = float(sub.mean()) if len(sub) else float("nan")
        std_gap  = float(sub.std())  if len(sub) > 1 else 0.0
        print(f"  {CONDITION_LABELS.get(c,c):35s}  gap={mean_gap:.4f} ± {std_gap:.4f}")
        gap_rows.append([CONDITION_LABELS.get(c, c), round(mean_gap, 4), round(std_gap, 4)])
        wandb.log({f"aggregate/{c}/mean_oracle_gap": mean_gap})

    wandb.log({"aggregate/oracle_gap_table": wandb.Table(
        columns=["condition", "mean_gap", "std_gap"], data=gap_rows)})
    return merged


# ---------------------------------------------------------------------------
# Table 3b: TPR at low FPR — built from stored probs (more thresholds than bootstrap)
# ---------------------------------------------------------------------------

def build_tpr_fpr_table(domain_data: dict, run3_results: str) -> pd.DataFrame:
    """
    Compute TPR@FPR from raw scores for ALL attacks and shadow conditions.
    All values are exact (not bootstrap) so oracle, SAMA, and shadow are comparable
    at the same threshold grid, including thresholds not stored in bootstrap CIs.

    Attack columns: oracle, sama, + each shadow condition (A_oracle..E_pruned30)
    """
    rows = []
    for domain in DOMAINS:
        data = domain_data.get(domain)

        # Load exact target labels (shared by oracle and shadow)
        y_target = None
        y_pt = os.path.join(run3_results, domain, "y.pt")
        if os.path.exists(y_pt):
            try:
                y_target = torch.load(y_pt, weights_only=True).numpy()
            except Exception as e:
                print(f"  WARNING: could not load y.pt for {domain}: {e}")

        # Oracle raw scores from Run3 classifier_results.pt
        oracle_probs, oracle_y = load_oracle_raw(domain, run3_results)

        # SAMA raw scores
        sama_scores, sama_labels = load_sama_raw(domain, run3_results)

        def safe_tpr_row(y, scores, label_prefix):
            if y is None or scores is None or len(np.unique(y)) < 2:
                return {lbl: float("nan") for lbl in LOW_FPR_LABELS}
            try:
                return _tpr_row(y, scores)
            except Exception:
                return {lbl: float("nan") for lbl in LOW_FPR_LABELS}

        oracle_tpr = safe_tpr_row(oracle_y, oracle_probs, "oracle")
        sama_tpr   = safe_tpr_row(sama_labels, sama_scores, "sama")

        # Shadow conditions — all use y_target from y.pt
        cond_tprs = {}
        if data is not None:
            for cname in CONDITIONS:
                cdata = data["conditions"].get(cname)
                probs = cdata.get("probs") if cdata else None
                if probs is not None and y_target is not None:
                    p = np.array(probs) if not isinstance(probs, np.ndarray) else probs
                    cond_tprs[cname] = safe_tpr_row(y_target, p, cname)
                else:
                    cond_tprs[cname] = {lbl: float("nan") for lbl in LOW_FPR_LABELS}

        row = {"domain": domain}
        for lbl in LOW_FPR_LABELS:
            row[f"oracle_{lbl}"] = oracle_tpr.get(lbl, float("nan"))
            row[f"sama_{lbl}"]   = sama_tpr.get(lbl, float("nan"))
            for cname in CONDITIONS:
                row[f"{cname}_{lbl}"] = cond_tprs.get(cname, {}).get(lbl, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


def log_tpr_fpr_tables(df: pd.DataFrame):
    """
    Print and log wandb tables for each low-FPR threshold.
    df is wide-format: columns are {oracle|sama|cname}_{fpr_label} per domain row.
    All values computed from raw scores — comparable across oracle, SAMA, and shadow.
    """
    ATTACK_COLS = (
        [("oracle", "Oracle (white-box)"), ("sama", "SAMA (T=4)")] +
        [(c, CONDITION_LABELS.get(c, c)) for c in CONDITIONS]
    )

    print("\n=== TPR at Low FPR — all attacks, from raw scores (exact, not bootstrap) ===")

    for fpr_label in LOW_FPR_LABELS:
        cols = [f"{key}_{fpr_label}" for key, _ in ATTACK_COLS]
        # Only include columns that exist and have any non-NaN data
        present = [c for c in cols if c in df.columns and not df[c].isna().all()]
        if not present:
            continue

        print(f"\n  {fpr_label}")
        display = df[["domain"] + present].round(4)
        display.columns = ["domain"] + [lbl for key, lbl in ATTACK_COLS
                                         if f"{key}_{fpr_label}" in present]
        print(display.to_string(index=False))

        t_cols = list(display.columns)
        t_data = display.values.tolist()
        safe   = fpr_label.replace(".", "_").replace("%", "pct").replace("@", "_at_")
        wandb.log({f"aggregate/tpr/{safe}": wandb.Table(columns=t_cols, data=t_data)})

    # Mean across domains per attack per threshold
    print("\n  Mean TPR across domains:")
    summary_rows = []
    for key, label in ATTACK_COLS:
        row = [label]
        for fpr_label in LOW_FPR_LABELS:
            col = f"{key}_{fpr_label}"
            mean_val = float(df[col].dropna().mean()) if col in df.columns else float("nan")
            row.append(round(mean_val, 4))
            safe = fpr_label.replace(".", "_").replace("%", "pct").replace("@", "_at_")
            wandb.log({f"aggregate/{key}/{safe}_mean": mean_val})
        summary_rows.append(row)

    print(pd.DataFrame(summary_rows, columns=["attack"] + LOW_FPR_LABELS).to_string(index=False))
    wandb.log({"aggregate/tpr_low_fpr_mean": wandb.Table(
        columns=["attack"] + LOW_FPR_LABELS, data=summary_rows)})


def plot_tpr_at_low_fpr(df: pd.DataFrame, out_dir: str):
    """
    Grouped bar chart: TPR@1%FPR per domain for oracle, SAMA, shadow-B, shadow-E.
    All four attacks on the same chart for direct comparison.
    """
    fpr_label = "TPR@1%FPR"
    ATTACK_BARS = [
        ("oracle",     "Oracle (white-box)", "#2c7bb6"),
        ("sama",       "SAMA (T=4, N=128)", "#9467bd"),
        ("B_full46",   "Shadow B: 46-dim",  "#d7191c"),
        ("E_pruned30", "Shadow E: 30-dim",  "#1a9641"),
    ]

    domain_order = [d for d in DOMAINS if d in df["domain"].values]
    if not domain_order:
        return

    x       = np.arange(len(domain_order))
    n_bars  = len(ATTACK_BARS)
    width   = 0.18
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * width

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (key, label, color) in enumerate(ATTACK_BARS):
        col  = f"{key}_{fpr_label}"
        vals = [float(df.loc[df["domain"] == d, col].values[0])
                if col in df.columns and d in df["domain"].values else float("nan")
                for d in domain_order]
        ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.87)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_order, rotation=20, ha="right")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_title("TPR @ 1% FPR — Oracle vs SAMA vs Shadow Transfer (exact, all 6 domains)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.01, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_6_tpr_at_1pct_fpr.{ext}"), dpi=150)
    wandb.log({"plots/tpr_at_1pct_fpr": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 4: SHAP rank stability (Spearman r across domains)
# ---------------------------------------------------------------------------

def log_shap_rank_stability(domain_data: dict, condition: str = "B_full46"):
    """
    For each domain, rank features by mean |SHAP|.
    Compute Spearman rank correlation between all domain pairs.
    High correlation → feature importance is consistent across domains.
    """
    domain_rankings = {}
    for domain, data in domain_data.items():
        cdata = data["conditions"].get(condition)
        if cdata is None or cdata.get("shap_values") is None:
            continue
        sv    = cdata["shap_values"]
        names = cdata.get("feature_names", [])
        if sv is None or len(names) == 0:
            continue
        mean_abs = np.abs(sv).mean(axis=0)
        ranked   = {name: float(mean_abs[i]) for i, name in enumerate(names)}
        domain_rankings[domain] = ranked

    if len(domain_rankings) < 2:
        print("  Not enough domains with SHAP values for rank stability analysis.")
        return

    domains_with_shap = list(domain_rankings.keys())
    all_features = sorted(
        set(f for r in domain_rankings.values() for f in r.keys())
    )

    # Build rank matrix [n_domains × n_features]
    rank_matrix = np.zeros((len(domains_with_shap), len(all_features)))
    for i, d in enumerate(domains_with_shap):
        for j, f in enumerate(all_features):
            rank_matrix[i, j] = domain_rankings[d].get(f, 0.0)

    # Spearman correlation between domain pairs
    n_d = len(domains_with_shap)
    corr_matrix = np.eye(n_d)
    for i in range(n_d):
        for j in range(i + 1, n_d):
            r, _ = spearmanr(rank_matrix[i], rank_matrix[j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    mean_r = float(np.mean(corr_matrix[np.triu_indices(n_d, k=1)]))
    print(f"\n  SHAP rank stability ({condition}): mean Spearman r = {mean_r:.4f}")
    wandb.log({f"aggregate/shap_rank_stability_{condition}": mean_r})

    # Log as wandb Table
    rows = [[domains_with_shap[i]] + [round(corr_matrix[i, j], 3) for j in range(n_d)]
            for i in range(n_d)]
    wandb.log({f"aggregate/shap_spearman_{condition}": wandb.Table(
        columns=["domain"] + domains_with_shap, data=rows)})

    return corr_matrix, domains_with_shap


# ---------------------------------------------------------------------------
# Table 5: Hellinger distance per feature (averaged across domains)
# ---------------------------------------------------------------------------

def build_hellinger_table(domain_data: dict, population: str = "target") -> pd.DataFrame:
    """Mean Hellinger distance per feature across domains, for a given population."""
    rows = []
    for domain, data in domain_data.items():
        dist_key = f"{population}_dist"
        dist = data.get(dist_key, {})
        for feat, stats in dist.items():
            if "hellinger" in stats:
                rows.append({"domain": domain, "feature": feat,
                             "hellinger": stats["hellinger"],
                             "uni_auc":   stats.get("uni_auc", 0.5)})
    return pd.DataFrame(rows)


def log_hellinger_table(df: pd.DataFrame, population: str):
    if df.empty:
        print(f"  No Hellinger data for population={population}")
        return

    pivot = df.pivot_table(index="feature", columns="domain",
                           values="hellinger", aggfunc="mean")
    pivot["mean_hellinger"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_hellinger", ascending=False)

    print(f"\n=== Top 10 features by mean Hellinger ({population}) ===")
    print(pivot[["mean_hellinger"]].head(10).to_string())

    # wandb table
    wandb.log({f"aggregate/hellinger_{population}": wandb.Table(
        columns=["feature", "mean_hellinger"] + list(pivot.columns[:-1]),
        data=[[idx, round(row["mean_hellinger"], 4)] +
              [round(row[d], 4) if d in row.index else float("nan")
               for d in list(pivot.columns[:-1])]
              for idx, row in pivot.iterrows()],
    )})
    return pivot


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_condition_comparison(df: pd.DataFrame, out_dir: str):
    """Grouped bar chart: x=domain, hue=condition, y=AUC."""
    fig, ax = plt.subplots(figsize=(14, 5))
    domain_order = DOMAINS
    x       = np.arange(len(domain_order))
    n_cond  = len(CONDITIONS)
    width   = 0.15
    offsets = np.linspace(-(n_cond-1)/2, (n_cond-1)/2, n_cond) * width

    for i, cname in enumerate(CONDITIONS):
        sub = df[df["condition"] == cname].set_index("domain")
        aucs = [float(sub.loc[d, "auc"]) if d in sub.index else float("nan")
                for d in domain_order]
        lo   = [float(sub.loc[d, "auc_lo"]) if d in sub.index else float("nan")
                for d in domain_order]
        hi   = [float(sub.loc[d, "auc_hi"]) if d in sub.index else float("nan")
                for d in domain_order]
        yerr_lo = [a - l if not np.isnan(a) else 0 for a, l in zip(aucs, lo)]
        yerr_hi = [h - a if not np.isnan(a) else 0 for a, h in zip(aucs, hi)]

        ax.bar(x + offsets[i], aucs, width,
               label=CONDITION_LABELS.get(cname, cname),
               color=PALETTE.get(cname, f"C{i}"),
               alpha=0.85)
        ax.errorbar(x + offsets[i], aucs,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", linewidth=0.8, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_order, rotation=20, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Run4 Shadow MIA — Transfer AUC by Condition and Domain")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_1_condition_comparison.{ext}"), dpi=150)
    wandb.log({"plots/condition_comparison": wandb.Image(fig)})
    plt.close(fig)


def plot_oracle_gap(df_gap: pd.DataFrame, out_dir: str):
    """Bar chart: oracle gap per condition (mean ± std across domains)."""
    gap_stats = []
    for c in CONDITIONS:
        sub = df_gap[df_gap["condition"] == c]["gap"].dropna()
        gap_stats.append({
            "condition": CONDITION_LABELS.get(c, c),
            "mean":      float(sub.mean()) if len(sub) else 0.0,
            "std":       float(sub.std())  if len(sub) > 1 else 0.0,
        })

    labels = [r["condition"] for r in gap_stats]
    means  = [r["mean"] for r in gap_stats]
    stds   = [r["std"]  for r in gap_stats]
    colors = [PALETTE.get(c, f"C{i}") for i, c in enumerate(CONDITIONS)]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85)
    ax.errorbar(range(len(labels)), means, yerr=stds,
                fmt="none", color="black", linewidth=1.0, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Oracle AUC − Condition AUC  (lower = better transfer)")
    ax.set_title("Run4: Oracle Gap per Condition (mean ± std, 6 domains)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_2_oracle_gap.{ext}"), dpi=150)
    wandb.log({"plots/oracle_gap": wandb.Image(fig)})
    plt.close(fig)


def plot_hellinger_heatmap(hellinger_pivot: pd.DataFrame, out_dir: str, population: str):
    """Heatmap: feature (rows, top 20 by mean) × domain (cols)."""
    if hellinger_pivot is None or hellinger_pivot.empty:
        return
    top20 = hellinger_pivot.head(20).drop(columns=["mean_hellinger"], errors="ignore")
    fig, ax = plt.subplots(figsize=(max(6, len(top20.columns) * 1.2), 8))
    sns.heatmap(
        top20.astype(float),
        ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3,
    )
    ax.set_title(f"Hellinger Distance (member vs non-member) — {population} features\n"
                 f"Top 20 features by mean across domains")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_3_hellinger_{population}.{ext}"), dpi=150)
    wandb.log({f"plots/hellinger_{population}": wandb.Image(fig)})
    plt.close(fig)


def plot_shap_rank_stability(corr_matrix: np.ndarray, domain_list: list, out_dir: str):
    """Spearman correlation heatmap across domain pairs."""
    if corr_matrix is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f",
                xticklabels=domain_list, yticklabels=domain_list,
                cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("SHAP Rank Stability (Spearman r)\nCondition B: full 46-dim shadow transfer")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_4_shap_rank_stability.{ext}"), dpi=150)
    wandb.log({"plots/shap_rank_stability": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 6 + Fig 7: Unified 3-attack comparison (oracle / SAMA / shadow)
# ---------------------------------------------------------------------------

UNIFIED_ATTACKS = [
    ("oracle",    "Oracle (white-box)",     "#2c7bb6"),
    ("sama",      "SAMA (T=4, N=128)",      "#9467bd"),
    ("shadow_B",  "Shadow B: 46-dim",       "#d7191c"),
    ("shadow_E",  "Shadow E: 30-dim pruned","#1a9641"),
]


def build_unified_table(domain_data: dict, sama_aucs: dict,
                        run3_results: str) -> pd.DataFrame:
    """
    One row per domain: AUC + TPR@1%FPR for oracle, SAMA, shadow-B, shadow-E.
    AUC from stored bootstrap means (oracle/SAMA/shadow).
    TPR@1%FPR computed exactly from raw scores so all four attacks are comparable.
    """
    rows = []
    for domain in DOMAINS:
        data = domain_data.get(domain)
        if data is None:
            continue
        conds = data["conditions"]

        # AUC from bootstrap means
        oracle_m   = conds.get("A_oracle",   {}).get("metrics", {})
        shadow_B_m = conds.get("B_full46",   {}).get("metrics", {})
        shadow_E_m = conds.get("E_pruned30", {}).get("metrics", {})

        oracle_auc   = oracle_m.get("auc_mean",   oracle_m.get("auc",   float("nan")))
        shadow_B_auc = shadow_B_m.get("auc_mean", shadow_B_m.get("auc", float("nan")))
        shadow_E_auc = shadow_E_m.get("auc_mean", shadow_E_m.get("auc", float("nan")))
        sama_auc     = float(sama_aucs[domain]) if sama_aucs.get(domain) is not None else float("nan")

        # TPR@1%FPR — exact from raw scores (fair across all attacks)
        y_target = None
        y_pt = os.path.join(run3_results, domain, "y.pt")
        if os.path.exists(y_pt):
            try:
                y_target = torch.load(y_pt, weights_only=True).numpy()
            except Exception:
                pass

        oracle_probs, oracle_y  = load_oracle_raw(domain, run3_results)
        sama_scores, sama_labels = load_sama_raw(domain, run3_results)

        def exact_tpr1(y, scores):
            if y is None or scores is None or len(np.unique(y)) < 2:
                return float("nan")
            try:
                return _tpr_at_fpr(y, np.array(scores), 0.01)
            except Exception:
                return float("nan")

        shadow_B_probs = conds.get("B_full46",   {}).get("probs")
        shadow_E_probs = conds.get("E_pruned30", {}).get("probs")

        rows.append({
            "domain":           domain,
            "oracle_auc":       oracle_auc,
            "sama_auc":         sama_auc,
            "shadow_B_auc":     shadow_B_auc,
            "shadow_E_auc":     shadow_E_auc,
            "oracle_tpr1pct":   exact_tpr1(oracle_y,  oracle_probs),
            "sama_tpr1pct":     exact_tpr1(sama_labels, sama_scores),
            "shadow_B_tpr1pct": exact_tpr1(y_target,  shadow_B_probs),
            "shadow_E_tpr1pct": exact_tpr1(y_target,  shadow_E_probs),
            "gap_sama":         oracle_auc - sama_auc,
            "gap_shadow_B":     oracle_auc - shadow_B_auc,
            "gap_shadow_E":     oracle_auc - shadow_E_auc,
        })
    return pd.DataFrame(rows)


def log_unified_table(df: pd.DataFrame):
    print("\n=== Unified Attack Comparison (AUC + TPR@1%FPR) ===")
    print("  AUC columns (bootstrap mean):")
    auc_cols = ["domain", "oracle_auc", "sama_auc", "shadow_B_auc", "shadow_E_auc",
                "gap_sama", "gap_shadow_B", "gap_shadow_E"]
    print(df[auc_cols].round(4).to_string(index=False))

    print("\n  TPR@1%FPR columns (exact from raw scores):")
    tpr_cols = ["domain", "oracle_tpr1pct", "sama_tpr1pct", "shadow_B_tpr1pct", "shadow_E_tpr1pct"]
    print(df[tpr_cols].round(4).to_string(index=False))

    # Means
    print("\n  Means:")
    for col in ["oracle_auc", "sama_auc", "shadow_B_auc", "shadow_E_auc",
                "oracle_tpr1pct", "sama_tpr1pct", "shadow_B_tpr1pct", "shadow_E_tpr1pct"]:
        mean_val = float(df[col].dropna().mean())
        print(f"    {col:22s}: {mean_val:.4f}")
        wandb.log({f"unified/{col}_mean": mean_val})

    # wandb table — AUC
    t_auc_cols = ["domain", "Oracle AUC", "SAMA AUC", "Shadow-B AUC", "Shadow-E AUC",
                  "Gap Oracle−SAMA", "Gap Oracle−B", "Gap Oracle−E"]
    t_auc_data = []
    for _, row in df.iterrows():
        def v(x): return round(float(x), 4) if not np.isnan(x) else None
        t_auc_data.append([row["domain"], v(row["oracle_auc"]), v(row["sama_auc"]),
                           v(row["shadow_B_auc"]), v(row["shadow_E_auc"]),
                           v(row["gap_sama"]), v(row["gap_shadow_B"]), v(row["gap_shadow_E"])])
    wandb.log({"comparison/unified_auc_table": wandb.Table(columns=t_auc_cols, data=t_auc_data)})

    # wandb table — TPR@1%FPR
    t_tpr_cols = ["domain", "Oracle TPR@1%", "SAMA TPR@1%", "Shadow-B TPR@1%", "Shadow-E TPR@1%"]
    t_tpr_data = []
    for _, row in df.iterrows():
        def v(x): return round(float(x), 4) if not np.isnan(x) else None
        t_tpr_data.append([row["domain"], v(row["oracle_tpr1pct"]), v(row["sama_tpr1pct"]),
                           v(row["shadow_B_tpr1pct"]), v(row["shadow_E_tpr1pct"])])
    wandb.log({"comparison/unified_tpr1pct_table": wandb.Table(columns=t_tpr_cols, data=t_tpr_data)})


def plot_unified_comparison(df: pd.DataFrame, out_dir: str):
    """
    Grouped bar chart per domain — 4 bars: oracle, SAMA, shadow_B, shadow_E.
    This is the headline figure for the threat-model progression.
    """
    domain_order = [d for d in DOMAINS if d in df["domain"].values]
    x       = np.arange(len(domain_order))
    n_bars  = len(UNIFIED_ATTACKS)
    width   = 0.18
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * width

    auc_keys = ["oracle_auc", "sama_auc", "shadow_B_auc", "shadow_E_auc"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ((key, label, color), auc_col) in enumerate(zip(UNIFIED_ATTACKS, auc_keys)):
        sub  = df.set_index("domain")
        vals = [float(sub.loc[d, auc_col]) if d in sub.index else float("nan")
                for d in domain_order]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.87)

        # Annotate bar tops
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_order, rotation=20, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Threat-Model Progression: Oracle → SAMA → Shadow Transfer\n"
                 "(MDLM-OWT, MIMIR benchmark, 6 domains)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.45, 1.08)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_r4_7_unified_comparison.{ext}"), dpi=150)
    wandb.log({"plots/unified_comparison": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_base", default="../shadow_results",
                        help="Path to shadow_results/ directory")
    parser.add_argument("--run3_results", default="../Run_3_Full_MIMIR/results",
                        help="Path to Run3 results directory")
    parser.add_argument("--plots_dir",    default="../plots",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=f"{cfg.WANDB_RUN_PREFIX}-aggregate-all-domains",
        tags=["run4", "aggregate", "shadow-mia"],
    )

    print("Loading domain results...")
    domain_data = load_domain_results(args.results_base, args.run3_results)
    if not domain_data:
        print("ERROR: No domain results found. Check --results_base path.")
        wandb.finish()
        return

    print("\nLoading Run3 SAMA scores...")
    sama_aucs = load_all_sama_aucs(args.run3_results)

    # ----------------------------------------------------------------
    # AUC tables
    # ----------------------------------------------------------------
    df_auc  = build_auc_table(domain_data)
    pivot   = log_auc_tables(df_auc)
    df_gap  = log_oracle_gap(df_auc)

    # ----------------------------------------------------------------
    # TPR at low FPR
    # ----------------------------------------------------------------
    df_tpr = build_tpr_fpr_table(domain_data, args.run3_results)
    log_tpr_fpr_tables(df_tpr)
    plot_tpr_at_low_fpr(df_tpr, args.plots_dir)

    # ----------------------------------------------------------------
    # SHAP rank stability
    # ----------------------------------------------------------------
    shap_result = log_shap_rank_stability(domain_data, condition="B_full46")
    corr_matrix, domain_list = (shap_result if shap_result else (None, []))

    # ----------------------------------------------------------------
    # Hellinger tables
    # ----------------------------------------------------------------
    for pop in ("target", "shadow"):
        h_df = build_hellinger_table(domain_data, population=pop)
        h_pivot = log_hellinger_table(h_df, population=pop)
        plot_hellinger_heatmap(h_pivot, args.plots_dir, population=pop)

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------
    # Unified 3-attack comparison (oracle / SAMA / shadow B / shadow E)
    # ----------------------------------------------------------------
    df_unified = build_unified_table(domain_data, sama_aucs, args.run3_results)
    log_unified_table(df_unified)

    # ----------------------------------------------------------------
    print("\nGenerating figures...")
    plot_condition_comparison(df_auc, args.plots_dir)
    plot_oracle_gap(df_gap, args.plots_dir)
    if corr_matrix is not None:
        plot_shap_rank_stability(corr_matrix, domain_list, args.plots_dir)
    plot_unified_comparison(df_unified, args.plots_dir)

    # ----------------------------------------------------------------
    # Save summary
    # ----------------------------------------------------------------
    torch.save({
        "auc_df":        df_auc.to_dict(),
        "gap_df":        df_gap.to_dict(),
        "tpr_fpr_df":    df_tpr.to_dict(),
        "unified_df":         df_unified.to_dict(),
        "sama_aucs":          sama_aucs,
        "low_fpr_note":       "all TPR@FPR computed from raw scores (exact, not bootstrap) for cross-attack comparability",
        "domains":       list(domain_data.keys()),
        "conditions":    CONDITIONS,
        "low_fpr_thresholds": LOW_FPR_THRESHOLDS,
    }, os.path.join(args.results_base, "summary_table.pt"))
    print(f"\nSaved summary_table.pt to {args.results_base}/")

    wandb.finish()
    print("Aggregation complete.")


if __name__ == "__main__":
    main()
