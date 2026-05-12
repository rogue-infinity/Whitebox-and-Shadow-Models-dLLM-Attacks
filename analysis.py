"""
analysis.py — Publication-quality analysis for Run_3_Full_MIMIR.

Run AFTER all 9 dataset pipelines have completed and results/ are downloaded.
Reads results/{DATASET}/benchmark.pt and results/{DATASET}/{X,y,classifier_results}.pt
for all 9 datasets and produces all tables + figures for the paper.

Usage:
    python analysis.py                          # all 9 datasets
    python analysis.py --datasets arxiv github  # subset
    python analysis.py --out_dir plots/analysis

Outputs (in --out_dir):
  Tables (CSV + LaTeX):
    Table_1_main_results.{csv,tex}
    Table_2_feature_ablation.{csv,tex}
    Table_3_bootstrap_ci.{csv,tex}
    Table_4_aggregate.{csv,tex}

  Figures (PDF + PNG):
    fig_roc_per_dataset_{D}.{pdf,png}       — per-dataset ROC
    fig_roc_aggregate.{pdf,png}             — mean ROC ± std
    fig_pr_per_dataset_{D}.{pdf,png}        — per-dataset PR curve
    fig_tpr_bar.{pdf,png}                   — TPR@1%FPR bar chart
    fig_feature_hellinger.{pdf,png}         — Hellinger dist per signal group
    fig_feature_importance.{pdf,png}        — XGB feature importance
    fig_kde_top_features.{pdf,png}          — KDE of top-5 features
    fig_attn_entropy_trajectory.{pdf,png}   — attn entropy over timesteps
    fig_cross_layer_corr.{pdf,png}          — cross-layer correlation heatmap diff
    fig_elbo_gap.{pdf,png}                  — ELBO gap per dataset
    fig_training_loss.{pdf,png}             — fine-tuning loss curves (from wandb)
    fig_calibration.{pdf,png}              — reliability diagrams
    fig_umap.{pdf,png}                      — UMAP of feature space
    fig_tsne.{pdf,png}                      — t-SNE of feature space

  wandb: all tables + figures logged to project="da5001-mia", run="run3-analysis"
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Signal group names (11 groups, T scalars each) ────────────────────────────
SIGNAL_GROUPS = [
    "ELBO_traj",       # T: L(t) across masking ratios
    "ELBO_var",        # 1: variance of L(t)
    "dL_dt",           # T: dL/dt
    "d2L_dt2",         # T: d²L/dt²
    "pred_entropy",    # T: prediction entropy
    "mask_consistency",# T: multi-mask consistency
    "hidden_norm",     # T: hidden state norm (mean over layers)
    "hidden_cosine",   # T: hidden cosine sim (mean over layers)
    "attn_entropy",    # T: attention entropy (AttenMIA)
    "attn_crosslayer", # T: cross-layer correlation (AttenMIA)
    "attn_barycenter", # T: barycentric drift (AttenMIA)
    "attn_perturbation",# T: attention perturbation var (AttenMIA)
]
# Feature vector layout: [T, 1, T, T, T, T, T, T, T, T, T, T] then +1 cross-model cosine
# Total = 11*T + 1 + 1 (cross-model) but cross-model is appended separately

FPR_THRESHOLDS = [0.001, 0.01, 0.10]
METHOD_COLORS = {
    "Loss (baseline)":    "#aaaaaa",
    "Zlib (baseline)":    "#888888",
    "Ratio (baseline)":   "#666666",
    "SAMA (baseline)":    "#444444",
    "XGBoost (ours)":     "#2196F3",
    "LightGBM (ours)":    "#4CAF50",
    "MLP (ours)":         "#FF5722",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true, y_score, fpr_threshold):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


def bootstrap_ci(y_true, y_score, fpr_thresholds=FPR_THRESHOLDS, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y_true)
    auc_s = []
    tpr_s = {f: [] for f in fpr_thresholds}
    for _ in range(n):
        idx = rng.randint(0, N, N)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_s.append(roc_auc_score(yb, sb))
        for f in fpr_thresholds:
            tpr_s[f].append(tpr_at_fpr(yb, sb, f))
    def ci(s):
        s = np.array(s) if s else np.array([0.])
        return float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
    out = {}
    out["auc"], out["auc_lo"], out["auc_hi"] = ci(auc_s)
    for f in fpr_thresholds:
        m, lo, hi = ci(tpr_s[f])
        out[f"tpr_{f}"], out[f"tpr_{f}_lo"], out[f"tpr_{f}_hi"] = m, lo, hi
    return out


def hellinger(p, q):
    """Hellinger distance between two 1-D empirical distributions (via histograms)."""
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max()) + 1e-9
    bins = np.linspace(lo, hi, 50)
    ph, _ = np.histogram(p, bins=bins, density=True)
    qh, _ = np.histogram(q, bins=bins, density=True)
    ph = ph / (ph.sum() + 1e-9)
    qh = qh / (qh.sum() + 1e-9)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(ph) - np.sqrt(qh)) ** 2)))


def save_fig(fig, path_no_ext, out_dir):
    pdf = os.path.join(out_dir, path_no_ext + ".pdf")
    png = os.path.join(out_dir, path_no_ext + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return png


def load_dataset_results(dataset, results_base):
    """Load benchmark.pt + classifier_results.pt for one dataset."""
    rdir = os.path.join(results_base, dataset)
    bench_path = os.path.join(rdir, "benchmark.pt")
    clf_path   = os.path.join(rdir, "classifier_results.pt")
    x_path     = os.path.join(rdir, "X.pt")
    y_path     = os.path.join(rdir, "y.pt")

    if not os.path.exists(bench_path):
        print(f"  [SKIP] {dataset}: benchmark.pt not found")
        return None

    bench = torch.load(bench_path, weights_only=False)
    out = {"dataset": dataset, "methods": {}}

    for mname, mdata in bench["methods"].items():
        scores = np.array(mdata["scores"], dtype=np.float64)
        labels = np.array(mdata["labels"], dtype=np.int32)
        out["methods"][mname] = {"scores": scores, "labels": labels}

    # Feature matrix (for ablation, UMAP, etc.)
    if os.path.exists(x_path) and os.path.exists(y_path):
        out["X"] = torch.load(x_path, weights_only=True).numpy()
        out["y"] = torch.load(y_path, weights_only=True).numpy()
    else:
        out["X"] = None
        out["y"] = None

    # XGB feature importances
    if os.path.exists(clf_path):
        clf = torch.load(clf_path, weights_only=False)
        out["clf"] = clf
    else:
        out["clf"] = None

    return out


# ── Table 1: Main results ──────────────────────────────────────────────────────

def build_table1(all_results):
    """AUC + TPR@FPR for all datasets × methods."""
    rows = []
    methods_seen = set()
    for ds, res in all_results.items():
        for mname, mdata in res["methods"].items():
            methods_seen.add(mname)
            m = bootstrap_ci(mdata["labels"], mdata["scores"])
            rows.append({
                "Dataset": ds,
                "Method":  mname,
                "AUC":     f"{m['auc']:.4f}",
                "AUC_CI":  f"[{m['auc_lo']:.4f}, {m['auc_hi']:.4f}]",
                "TPR@0.1%FPR": f"{m[f'tpr_0.001']*100:.1f}",
                "TPR@1%FPR":   f"{m[f'tpr_0.01']*100:.1f}",
                "TPR@10%FPR":  f"{m[f'tpr_0.1']*100:.1f}",
            })
    return rows


def rows_to_latex(rows, caption, label):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def save_table(rows, name, out_dir, caption="", label=""):
    import csv
    csv_path = os.path.join(out_dir, f"{name}.csv")
    tex_path = os.path.join(out_dir, f"{name}.tex")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(tex_path, "w") as f:
            f.write(rows_to_latex(rows, caption, label))
    return csv_path, tex_path


# ── Figure: ROC curves per dataset ────────────────────────────────────────────

def plot_roc_per_dataset(ds, res, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    for mname, mdata in res["methods"].items():
        fpr, tpr, _ = roc_curve(mdata["labels"], mdata["scores"])
        auc_val = roc_auc_score(mdata["labels"], mdata["scores"])
        color = METHOD_COLORS.get(mname, None)
        lw = 2 if "ours" in mname else 1
        ls = "-" if "ours" in mname else "--"
        ax.plot(fpr, tpr, label=f"{mname} (AUC={auc_val:.3f})",
                color=color, lw=lw, ls=ls)
    ax.plot([0,1],[0,1],"k:",lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {ds}")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout()
    return save_fig(fig, f"fig_roc_per_dataset_{ds}", out_dir)


def plot_roc_aggregate(all_results, out_dir):
    """Mean ROC ± std across datasets for each method."""
    from collections import defaultdict
    method_fprs = defaultdict(list)
    method_tprs = defaultdict(list)
    common_fpr = np.linspace(0, 1, 200)

    for ds, res in all_results.items():
        for mname, mdata in res["methods"].items():
            fpr, tpr, _ = roc_curve(mdata["labels"], mdata["scores"])
            method_tprs[mname].append(np.interp(common_fpr, fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    for mname, tpr_list in method_tprs.items():
        tprs = np.array(tpr_list)
        mean_tpr = tprs.mean(axis=0)
        std_tpr  = tprs.std(axis=0)
        mean_auc = auc(common_fpr, mean_tpr)
        color = METHOD_COLORS.get(mname, None)
        lw = 2 if "ours" in mname else 1
        ls = "-" if "ours" in mname else "--"
        ax.plot(common_fpr, mean_tpr,
                label=f"{mname} (AUC={mean_auc:.3f})",
                color=color, lw=lw, ls=ls)
        if "ours" in mname:
            ax.fill_between(common_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            alpha=0.15, color=color)
    ax.plot([0,1],[0,1],"k:",lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Aggregate ROC (mean ± std across datasets)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout()
    return save_fig(fig, "fig_roc_aggregate", out_dir)


def plot_pr_per_dataset(ds, res, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    for mname, mdata in res["methods"].items():
        prec, rec, _ = precision_recall_curve(mdata["labels"], mdata["scores"])
        pr_auc = auc(rec, prec)
        color = METHOD_COLORS.get(mname, None)
        lw = 2 if "ours" in mname else 1
        ls = "-" if "ours" in mname else "--"
        ax.plot(rec, prec, label=f"{mname} (AP={pr_auc:.3f})",
                color=color, lw=lw, ls=ls)
    base_rate = np.mean(list(res["methods"].values())[0]["labels"])
    ax.axhline(base_rate, color="k", ls=":", lw=0.8, label=f"Baseline={base_rate:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {ds}")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    return save_fig(fig, f"fig_pr_per_dataset_{ds}", out_dir)


def plot_tpr_bar(all_results, fpr_thr, out_dir):
    """Bar chart: TPR@fpr_thr for all methods × datasets."""
    datasets = list(all_results.keys())
    methods  = list(list(all_results.values())[0]["methods"].keys())
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(8, len(datasets)*1.2), 5))
    for i, mname in enumerate(methods):
        tprs = []
        for ds in datasets:
            md = all_results[ds]["methods"].get(mname)
            if md:
                tprs.append(tpr_at_fpr(md["labels"], md["scores"], fpr_thr) * 100)
            else:
                tprs.append(0.)
        color = METHOD_COLORS.get(mname, None)
        ax.bar(x + i * width, tprs, width, label=mname, color=color)

    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"TPR @ {fpr_thr*100:.1f}% FPR (%)")
    ax.set_title(f"TPR @ {fpr_thr*100:.1f}% FPR — All Datasets")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return save_fig(fig, f"fig_tpr_bar_fpr{int(fpr_thr*1000):04d}", out_dir)


# ── Feature analysis ──────────────────────────────────────────────────────────

def get_group_indices(T=cfg.T_STEPS):
    """Return start:end index slices for each signal group in the feature vector."""
    # Layout: ELBO_traj[T], ELBO_var[1], dL_dt[T], d2L_dt2[T], pred_entropy[T],
    #         mask_consistency[T], hidden_norm[T], hidden_cosine[T],
    #         attn_entropy[T], attn_crosslayer[T], attn_barycenter[T], attn_perturbation[T]
    #         + cross_model_cosine[1]
    sizes = [T, 1, T, T, T, T, T, T, T, T, T, T, 1]
    group_names = SIGNAL_GROUPS + ["cross_model_cosine"]
    slices = {}
    idx = 0
    for name, size in zip(group_names, sizes):
        slices[name] = slice(idx, idx + size)
        idx += size
    return slices


def plot_feature_hellinger(all_results, out_dir):
    """Hellinger distance per signal group, averaged across datasets."""
    slices = get_group_indices()
    group_names = list(slices.keys())
    all_h = {g: [] for g in group_names}

    for ds, res in all_results.items():
        if res["X"] is None:
            continue
        X, y = res["X"], res["y"]
        for g, sl in slices.items():
            feats = X[:, sl]
            mem    = feats[y == 1].flatten()
            nonmem = feats[y == 0].flatten()
            if len(mem) > 0 and len(nonmem) > 0:
                all_h[g].append(hellinger(mem, nonmem))

    means = [np.mean(all_h[g]) if all_h[g] else 0. for g in group_names]
    stds  = [np.std(all_h[g])  if all_h[g] else 0. for g in group_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2196F3" if "attn" in g else "#FF5722" if "ELBO" in g
              else "#4CAF50" for g in group_names]
    ax.bar(group_names, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_xlabel("Signal Group"); ax.set_ylabel("Hellinger Distance")
    ax.set_title("Member vs Non-member Separation by Signal Group\n(mean ± std across datasets)")
    ax.set_xticklabels(group_names, rotation=40, ha="right", fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "fig_feature_hellinger", out_dir)


def plot_kde_top_features(all_results, out_dir, top_k=5):
    """KDE of top-K most discriminative features (by mean Hellinger across datasets)."""
    slices = get_group_indices()
    all_X, all_y = [], []
    for ds, res in all_results.items():
        if res["X"] is not None:
            all_X.append(res["X"])
            all_y.append(res["y"])
    if not all_X:
        return None

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    D = X.shape[1]

    # Per-feature Hellinger
    h_scores = []
    for i in range(D):
        mem    = X[y == 1, i]
        nonmem = X[y == 0, i]
        h_scores.append(hellinger(mem, nonmem))
    top_idx = np.argsort(h_scores)[::-1][:top_k]

    fig, axes = plt.subplots(1, top_k, figsize=(3 * top_k, 3.5))
    for ax, fi in zip(axes, top_idx):
        mem    = X[y == 1, fi]
        nonmem = X[y == 0, fi]
        sns.kdeplot(mem,    ax=ax, label="Member",     fill=True, alpha=0.4, color="#2196F3")
        sns.kdeplot(nonmem, ax=ax, label="Non-member", fill=True, alpha=0.4, color="#FF5722")
        ax.set_title(f"Feature {fi}\nH={h_scores[fi]:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("")
    fig.suptitle("KDE — Top-5 Discriminative Features", y=1.01)
    fig.tight_layout()
    return save_fig(fig, "fig_kde_top_features", out_dir)


def plot_feature_ablation(all_results, out_dir):
    """AUC drop when each signal group is removed (leave-one-group-out ablation)."""
    from sklearn.linear_model import LogisticRegression
    slices = get_group_indices()
    group_names = list(slices.keys())

    all_X, all_y = [], []
    for ds, res in all_results.items():
        if res["X"] is not None:
            all_X.append(res["X"])
            all_y.append(res["y"])
    if not all_X:
        return None

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Baseline AUC (all features)
    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    base_clf.fit(Xs, y)
    base_auc = roc_auc_score(y, base_clf.predict_proba(Xs)[:, 1])

    drops = []
    for g, sl in slices.items():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[sl] = False
        Xm = scaler.fit_transform(X[:, mask])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xm, y)
        drop = base_auc - roc_auc_score(y, clf.predict_proba(Xm)[:, 1])
        drops.append(drop)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e53935" if d > 0.02 else "#1976D2" for d in drops]
    ax.bar(group_names, drops, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Removed Signal Group")
    ax.set_ylabel("AUC Drop (baseline − ablated)")
    ax.set_title(f"Feature Group Ablation (baseline AUC={base_auc:.4f})")
    ax.set_xticklabels(group_names, rotation=40, ha="right", fontsize=8)
    fig.tight_layout()

    rows = [{"Group": g, "AUC_drop": f"{d:.4f}"} for g, d in zip(group_names, drops)]
    return save_fig(fig, "fig_feature_ablation", out_dir), rows


def plot_umap(all_results, out_dir):
    try:
        from umap import UMAP
    except ImportError:
        print("  [SKIP] umap-learn not installed — skipping UMAP plot")
        return None

    all_X, all_y = [], []
    for ds, res in all_results.items():
        if res["X"] is not None:
            all_X.append(res["X"])
            all_y.append(res["y"])
    if not all_X:
        return None

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y)

    n_max = 5000
    if len(X) > n_max:
        idx = np.random.choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]

    Xs = StandardScaler().fit_transform(X)
    emb = UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[y==0, 0], emb[y==0, 1], s=5, alpha=0.3, c="#FF5722", label="Non-member")
    ax.scatter(emb[y==1, 0], emb[y==1, 1], s=5, alpha=0.3, c="#2196F3", label="Member")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title("UMAP — 177-D Feature Space")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return save_fig(fig, "fig_umap", out_dir)


def plot_tsne(all_results, out_dir):
    all_X, all_y = [], []
    for ds, res in all_results.items():
        if res["X"] is not None:
            all_X.append(res["X"])
            all_y.append(res["y"])
    if not all_X:
        return None

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y)

    n_max = 3000
    if len(X) > n_max:
        idx = np.random.choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]

    Xs = StandardScaler().fit_transform(X)
    emb = TSNE(n_components=2, random_state=42, perplexity=40, n_jobs=-1).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[y==0, 0], emb[y==0, 1], s=5, alpha=0.3, c="#FF5722", label="Non-member")
    ax.scatter(emb[y==1, 0], emb[y==1, 1], s=5, alpha=0.3, c="#2196F3", label="Member")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title("t-SNE — 177-D Feature Space")
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    fig.tight_layout()
    return save_fig(fig, "fig_tsne", out_dir)


def plot_calibration(all_results, out_dir):
    """Reliability diagrams for XGBoost, LightGBM, MLP."""
    clf_methods = ["XGBoost (ours)", "LightGBM (ours)", "MLP (ours)"]
    fig, axes = plt.subplots(1, len(clf_methods), figsize=(4 * len(clf_methods), 4))

    for ax, mname in zip(axes, clf_methods):
        all_prob, all_y_true = [], []
        for ds, res in all_results.items():
            md = res["methods"].get(mname)
            if md:
                all_prob.append(md["scores"])
                all_y_true.append(md["labels"])
        if not all_prob:
            continue
        prob = np.concatenate(all_prob)
        ytrue = np.concatenate(all_y_true)
        frac_pos, mean_pred = calibration_curve(ytrue, prob, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label=mname, color=METHOD_COLORS.get(mname))
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration — {mname.split(' ')[0]}")
        ax.set_xlim(0,1); ax.set_ylim(0,1)

    fig.suptitle("Reliability Diagrams (all datasets pooled)")
    fig.tight_layout()
    return save_fig(fig, "fig_calibration", out_dir)


def plot_elbo_gap(all_results, out_dir):
    """Bar chart of ELBO gap per dataset (if verify logs are present in results)."""
    # ELBO gap info comes from the feature vectors:
    # members have lower ELBO (fine-tuned), so gap = mean(X_member[:,0:T]) - mean(X_nonmember[:,0:T])
    slices = get_group_indices()
    elbo_sl = slices["ELBO_traj"]

    datasets, gaps, stds = [], [], []
    for ds, res in all_results.items():
        if res["X"] is None:
            continue
        X, y = res["X"], res["y"]
        mem_elbo    = X[y == 1, elbo_sl].mean(axis=1)
        nonmem_elbo = X[y == 0, elbo_sl].mean(axis=1)
        gap = float(nonmem_elbo.mean() - mem_elbo.mean())
        datasets.append(ds)
        gaps.append(gap)
        stds.append(float(nonmem_elbo.std() + mem_elbo.std()) / 2)

    if not datasets:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(datasets)*1.1), 4))
    colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gaps]
    ax.bar(datasets, gaps, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ELBO Gap (non-member − member)")
    ax.set_title("Memorization ELBO Gap per Dataset\n(positive = fine-tuning induced memorization)")
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    fig.tight_layout()
    return save_fig(fig, "fig_elbo_gap", out_dir)


def plot_attn_entropy_trajectory(all_results, out_dir):
    """Attention entropy signal over timesteps (T=16, αmin=5%→αmax=50%)."""
    slices = get_group_indices()
    ae_sl = slices["attn_entropy"]
    T = cfg.T_STEPS
    alphas = np.linspace(cfg.ALPHA_MIN, cfg.ALPHA_MAX, T) * 100  # in %

    fig, ax = plt.subplots(figsize=(7, 4))
    for ds, res in all_results.items():
        if res["X"] is None:
            continue
        X, y = res["X"], res["y"]
        mem_ae    = X[y == 1, ae_sl].mean(axis=0)
        nonmem_ae = X[y == 0, ae_sl].mean(axis=0)
        ax.plot(alphas, mem_ae,    ls="-",  alpha=0.6, lw=1.5, label=f"{ds}:member")
        ax.plot(alphas, nonmem_ae, ls="--", alpha=0.6, lw=1.5, label=f"{ds}:non-member")

    ax.set_xlabel("Masking ratio α (%)")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title("Attention Entropy Trajectory (T=16)")
    # Compact legend — just show member/non-member indicator
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="gray", ls="-",  lw=2, label="Members"),
        Line2D([0],[0], color="gray", ls="--", lw=2, label="Non-members"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "fig_attn_entropy_trajectory", out_dir)


# ── Table 4: Aggregate ─────────────────────────────────────────────────────────

def build_table4(all_results):
    """Macro-average AUC and TPR across all datasets per method."""
    from collections import defaultdict
    method_aucs  = defaultdict(list)
    method_tprs1 = defaultdict(list)

    for ds, res in all_results.items():
        for mname, mdata in res["methods"].items():
            try:
                auc_val = roc_auc_score(mdata["labels"], mdata["scores"])
                tpr1    = tpr_at_fpr(mdata["labels"], mdata["scores"], 0.01)
                method_aucs[mname].append(auc_val)
                method_tprs1[mname].append(tpr1)
            except Exception:
                pass

    rows = []
    for mname in method_aucs:
        aucs = method_aucs[mname]
        t1s  = method_tprs1[mname]
        rows.append({
            "Method":       mname,
            "N_datasets":   len(aucs),
            "AUC_mean":     f"{np.mean(aucs):.4f}",
            "AUC_std":      f"{np.std(aucs):.4f}",
            "AUC_min":      f"{np.min(aucs):.4f}",
            "AUC_max":      f"{np.max(aucs):.4f}",
            "TPR@1%_mean":  f"{np.mean(t1s)*100:.1f}%",
            "TPR@1%_std":   f"{np.std(t1s)*100:.1f}%",
        })
    return sorted(rows, key=lambda r: r["AUC_mean"], reverse=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",     nargs="+", default=cfg.ALL_DATASETS)
    parser.add_argument("--results_base", default="results")
    parser.add_argument("--out_dir",      default="plots/analysis")
    parser.add_argument("--no_wandb",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(project="da5001-mia", name="run3-analysis",
                   config={"datasets": args.datasets})

    # ── Load all results ────────────────────────────────────────────────────
    print("Loading results...")
    all_results = {}
    for ds in args.datasets:
        res = load_dataset_results(ds, args.results_base)
        if res is not None:
            all_results[ds] = res
            print(f"  Loaded: {ds} ({list(res['methods'].keys())})")

    if not all_results:
        print("No results found. Run the pipeline first.")
        return

    logged_images = {}

    # ── Table 1: Main results ───────────────────────────────────────────────
    print("\nBuilding Table 1 (main results)...")
    t1_rows = build_table1(all_results)
    csv1, tex1 = save_table(t1_rows, "Table_1_main_results", args.out_dir,
                            "Main MIA Results — AUC and TPR across all datasets and methods",
                            "tab:main_results")
    print(f"  Saved {csv1}, {tex1}")

    # ── Table 4: Aggregate ──────────────────────────────────────────────────
    print("Building Table 4 (aggregate)...")
    t4_rows = build_table4(all_results)
    csv4, tex4 = save_table(t4_rows, "Table_4_aggregate", args.out_dir,
                            "Aggregate Results — Macro-average across 9 datasets",
                            "tab:aggregate")
    print(f"  Saved {csv4}, {tex4}")

    # ── ROC per dataset ─────────────────────────────────────────────────────
    print("\nPlotting per-dataset ROC + PR curves...")
    for ds, res in all_results.items():
        png = plot_roc_per_dataset(ds, res, args.out_dir)
        logged_images[f"roc/{ds}"] = wandb.Image(png)
        png = plot_pr_per_dataset(ds, res, args.out_dir)
        logged_images[f"pr/{ds}"] = wandb.Image(png)

    # ── Aggregate ROC ───────────────────────────────────────────────────────
    print("Plotting aggregate ROC...")
    png = plot_roc_aggregate(all_results, args.out_dir)
    logged_images["roc/aggregate"] = wandb.Image(png)

    # ── TPR bar chart ───────────────────────────────────────────────────────
    print("Plotting TPR bar charts...")
    for fpr_thr in FPR_THRESHOLDS:
        png = plot_tpr_bar(all_results, fpr_thr, args.out_dir)
        logged_images[f"tpr_bar/fpr{fpr_thr}"] = wandb.Image(png)

    # ── Feature Hellinger ───────────────────────────────────────────────────
    print("Plotting feature Hellinger distances...")
    png = plot_feature_hellinger(all_results, args.out_dir)
    if png:
        logged_images["features/hellinger"] = wandb.Image(png)

    # ── KDE top features ────────────────────────────────────────────────────
    print("Plotting KDE top features...")
    png = plot_kde_top_features(all_results, args.out_dir)
    if png:
        logged_images["features/kde_top"] = wandb.Image(png)

    # ── Feature ablation (Table 2) ──────────────────────────────────────────
    print("Running feature ablation (Table 2)...")
    ablation_result = plot_feature_ablation(all_results, args.out_dir)
    if ablation_result:
        png, t2_rows = ablation_result
        logged_images["features/ablation"] = wandb.Image(png)
        csv2, tex2 = save_table(t2_rows, "Table_2_feature_ablation", args.out_dir,
                                "Feature Group Ablation — AUC drop when group removed",
                                "tab:ablation")
        print(f"  Saved {csv2}, {tex2}")

    # ── ELBO gap ────────────────────────────────────────────────────────────
    print("Plotting ELBO gap...")
    png = plot_elbo_gap(all_results, args.out_dir)
    if png:
        logged_images["features/elbo_gap"] = wandb.Image(png)

    # ── Attention entropy trajectory ────────────────────────────────────────
    print("Plotting attention entropy trajectory...")
    png = plot_attn_entropy_trajectory(all_results, args.out_dir)
    if png:
        logged_images["features/attn_entropy_traj"] = wandb.Image(png)

    # ── Calibration ─────────────────────────────────────────────────────────
    print("Plotting calibration curves...")
    png = plot_calibration(all_results, args.out_dir)
    if png:
        logged_images["calibration"] = wandb.Image(png)

    # ── UMAP ────────────────────────────────────────────────────────────────
    print("Plotting UMAP...")
    png = plot_umap(all_results, args.out_dir)
    if png:
        logged_images["features/umap"] = wandb.Image(png)

    # ── t-SNE ───────────────────────────────────────────────────────────────
    print("Plotting t-SNE...")
    png = plot_tsne(all_results, args.out_dir)
    if png:
        logged_images["features/tsne"] = wandb.Image(png)

    # ── Table 3: Bootstrap CI for best method ───────────────────────────────
    print("Building Table 3 (bootstrap CI)...")
    t3_rows = []
    for ds, res in all_results.items():
        best_method = "XGBoost (ours)"
        md = res["methods"].get(best_method)
        if md is None:
            continue
        m = bootstrap_ci(md["labels"], md["scores"])
        t3_rows.append({
            "Dataset": ds,
            "Method":  best_method,
            "AUC":     f"{m['auc']:.4f}",
            "AUC_95CI":f"[{m['auc_lo']:.4f}, {m['auc_hi']:.4f}]",
            "TPR@0.1%FPR": f"{m['tpr_0.001']*100:.1f}% [{m['tpr_0.001_lo']*100:.1f}%, {m['tpr_0.001_hi']*100:.1f}%]",
            "TPR@1%FPR":   f"{m['tpr_0.01']*100:.1f}% [{m['tpr_0.01_lo']*100:.1f}%, {m['tpr_0.01_hi']*100:.1f}%]",
            "TPR@10%FPR":  f"{m['tpr_0.1']*100:.1f}% [{m['tpr_0.1_lo']*100:.1f}%, {m['tpr_0.1_hi']*100:.1f}%]",
        })
    csv3, tex3 = save_table(t3_rows, "Table_3_bootstrap_ci", args.out_dir,
                            "Bootstrap 95\\% CI for Best Method (XGBoost) per Dataset",
                            "tab:ci")
    print(f"  Saved {csv3}, {tex3}")

    # ── Log everything to wandb ─────────────────────────────────────────────
    if not args.no_wandb:
        print("\nLogging to wandb...")

        # Log wandb Tables
        if t1_rows:
            wt1 = wandb.Table(columns=list(t1_rows[0].keys()))
            for r in t1_rows:
                wt1.add_data(*r.values())
            wandb.log({"tables/main_results": wt1})

        if t4_rows:
            wt4 = wandb.Table(columns=list(t4_rows[0].keys()))
            for r in t4_rows:
                wt4.add_data(*r.values())
            wandb.log({"tables/aggregate": wt4})

        # Log all images
        wandb.log(logged_images)
        wandb.finish()

    print(f"\nDone. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
