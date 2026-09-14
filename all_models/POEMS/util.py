from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:  # pragma: no cover
    umap = None


def _ensure_dir(path: str | None) -> None:
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _to_numpy(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def _safe_pearson(a, b, eps: float = 1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, np.nan
    a = a[ok]
    b = b[ok]
    if np.std(a) < eps or np.std(b) < eps:
        return np.nan, np.nan
    return pearsonr(a, b)


# -----------------------------
# CSV writers
# -----------------------------

def save_latent_csv(metrics: dict, out_dir: str, split_name: str, prefix: str = "mu") -> pd.DataFrame:
    """Save deterministic latent means (or any latent matrix stored in metrics)."""
    _ensure_dir(out_dir)
    Z = _to_numpy(metrics[prefix])
    df = pd.DataFrame(Z, columns=[f"{prefix}{i+1}" for i in range(Z.shape[1])])
    df.insert(0, "sample_name", metrics["sample_names"])
    if "labels" in metrics:
        df.insert(1, "label", _to_numpy(metrics["labels"]).astype(int))
    df.to_csv(os.path.join(out_dir, f"{split_name}_latent_{prefix}.csv"), index=False)
    return df


def save_reconstruction_csv(metrics: dict, snp_names: list[str], out_dir: str, split_name: str) -> None:
    """Save observed SNP input/reconstruction in long form for diagnostics.
    This can become large for many SNPs; call only when needed.
    """
    _ensure_dir(out_dir)
    x_raw = _to_numpy(metrics["x_raw"])
    x_hat = _to_numpy(metrics["x_hat"])
    mask = _to_numpy(metrics["obs_mask"])
    rows = []
    for i, sample in enumerate(metrics["sample_names"]):
        observed = np.where(mask[i] > 0)[0]
        for j in observed:
            rows.append({"sample_name": sample, "snp_name": snp_names[j], "x_true": x_raw[i, j], "x_hat": x_hat[i, j]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{split_name}_reconstruction_long.csv"), index=False)


# -----------------------------
# Embedding and training plots
# -----------------------------

def plot_latent_heatmap(latent, labels=None, out_dir: str = ".", filename: str = "final_em_mu.pdf", prefix: str = "mu") -> None:
    _ensure_dir(out_dir)
    Z = _safe_zscore(_to_numpy(latent))
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] == 0:
        return
    if labels is not None:
        order = np.argsort(_to_numpy(labels).astype(int))
        Z = Z[order]
        labels_sorted = _to_numpy(labels).astype(int)[order]
    else:
        labels_sorted = None

    fig, ax = plt.subplots(figsize=(max(6, Z.shape[1] * 0.35), 4))
    im = ax.imshow(Z, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_xlabel("Latent dimensions")
    ax.set_ylabel("Samples")
    ax.set_xticks(np.arange(Z.shape[1]))
    ax.set_xticklabels([f"{prefix}{i+1}" for i in range(Z.shape[1])], fontsize=7)
    ax.set_title(f"{prefix} heatmap (column-z-scored)")
    if labels_sorted is not None and len(np.unique(labels_sorted)) > 1:
        _, counts = np.unique(labels_sorted, return_counts=True)
        for b in np.cumsum(counts)[:-1]:
            ax.axhline(b - 0.5, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def _plot_2d_embedding(emb, labels=None, sample_names=None, out_path: str = "embedding.pdf", title: str = "Embedding") -> None:
    labels_arr = None if labels is None else _to_numpy(labels).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels_arr is None or len(np.unique(labels_arr)) <= 1:
        ax.scatter(emb[:, 0], emb[:, 1], s=35, alpha=0.85)
    else:
        for lab in sorted(np.unique(labels_arr)):
            idx = labels_arr == lab
            ax.scatter(emb[idx, 0], emb[idx, 1], s=35, alpha=0.85, label=str(lab))
        ax.legend(frameon=False, fontsize=8, title="label")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_latent(
    latent,
    labels=None,
    sample_names=None,
    out_dir=".",
    filename="tsne_mu.pdf",
    random_state=21,
):
    _ensure_dir(out_dir)

    Z = _to_numpy(latent).astype(np.float64)

    valid_cols = np.nanvar(Z, axis=0) > 1e-8

    if valid_cols.sum() < 2:
        with open(
            os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
            "w"
        ) as f:
            f.write(
                "t-SNE skipped because fewer than 2 latent dimensions have usable variance.\n"
            )
        return

    Z = Z[:, valid_cols]
    Z = _safe_zscore(Z)

    if not np.isfinite(Z).all():
        with open(
            os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
            "w"
        ) as f:
            f.write(
                "t-SNE skipped because latent matrix contains non-finite values.\n"
            )
        return

    if np.nanstd(Z) < 1e-8:
        with open(
            os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
            "w"
        ) as f:
            f.write(
                "t-SNE skipped because latent matrix is nearly constant.\n"
            )
        return

    if Z.shape[0] < 5 or Z.shape[1] < 2:
        return

    perplexity = min(30, max(2, Z.shape[0] // 4))
    if perplexity >= Z.shape[0]:
        perplexity = max(2, Z.shape[0] - 1)

    try:
        emb = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(Z)
    except Exception as e:
        with open(
            os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
            "w"
        ) as f:
            f.write(f"t-SNE failed: {e}\n")
        return

    _plot_2d_embedding(
        emb,
        labels=labels,
        sample_names=sample_names,
        out_path=os.path.join(out_dir, filename),
        title="t-SNE of latent means",
    )


def plot_umap_latent(
    latent,
    labels=None,
    sample_names=None,
    out_dir: str = ".",
    filename: str = "umap_mu.pdf",
    random_state: int = 21,
) -> None:
    _ensure_dir(out_dir)

    if umap is None:
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because umap-learn is not installed.\n")
        return

    Z = _to_numpy(latent).astype(np.float64)

    if Z.ndim != 2 or Z.shape[0] < 5 or Z.shape[1] < 2:
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because latent matrix is too small.\n")
        return

    valid_cols = np.nanvar(Z, axis=0) > 1e-8

    if valid_cols.sum() < 2:
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because fewer than 2 latent dimensions have usable variance.\n")
        return

    Z = Z[:, valid_cols]
    Z = _safe_zscore(Z)

    if not np.isfinite(Z).all():
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because latent matrix contains non-finite values.\n")
        return

    if np.nanstd(Z) < 1e-8:
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because latent matrix is nearly constant.\n")
        return

    n_neighbors = min(15, max(2, Z.shape[0] - 1))

    try:
        emb = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2,
            random_state=random_state,
        ).fit_transform(Z)

    except Exception as e:
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write(f"UMAP failed: {e}\n")
        return

    if not np.isfinite(emb).all():
        with open(os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")), "w") as f:
            f.write("UMAP skipped because embedding contains non-finite values.\n")
        return

    _plot_2d_embedding(
        emb,
        labels=labels,
        sample_names=sample_names,
        out_path=os.path.join(out_dir, filename),
        title="UMAP of latent means",
    )


def plot_training_history(history_df: pd.DataFrame, out_dir: str = ".") -> None:
    _ensure_dir(out_dir)
    pairs = [
        ("train_rec_loss_all", "val_rec_loss_all", "Reconstruction loss", "Reconstruction_Loss.pdf"),
        ("train_kl_loss_all", "val_kl_loss_all", "KL loss", "Loss_KL.pdf"),
        ("train_trait_loss_all", "val_trait_loss_all", "Trait loss", "Trait_Loss.pdf"),
        ("train_total_loss_all", "val_total_loss_all", "Total loss", "Loss_Total.pdf"),
    ]
    for train_col, val_col, title, filename in pairs:
        if train_col not in history_df or val_col not in history_df:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history_df["epoch"], history_df[train_col], label="Train")
        ax.plot(history_df["epoch"], history_df[val_col], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# Trait prediction plots and metrics
# -----------------------------

def plot_trait_scatter(pred_df: pd.DataFrame, out_dir: str = ".", filename: str = "Trait_Scatter.pdf") -> None:
    _ensure_dir(out_dir)
    traits = list(pred_df["trait"].drop_duplicates())
    n = len(traits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, trait in zip(axes.ravel(), traits):
        df = pred_df[pred_df["trait"] == trait].copy()
        ok = np.isfinite(df["y_true"]) & np.isfinite(df["y_mu"])
        df = df[ok]
        ax.scatter(df["y_true"], df["y_mu"], s=40, alpha=0.85)
        if len(df) >= 2:
            lo = min(df["y_true"].min(), df["y_mu"].min())
            hi = max(df["y_true"].max(), df["y_mu"].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            r, _ = _safe_pearson(df["y_true"], df["y_mu"])
            r2 = r2_score(df["y_true"], df["y_mu"]) if len(df) >= 2 else np.nan
            mse = mean_squared_error(df["y_true"], df["y_mu"]) if len(df) >= 2 else np.nan
            ax.set_title(f"{trait}\nR²={r2:.3f}, r={r:.3f}, MSE={mse:.2f}")
        else:
            ax.set_title(trait)
        ax.set_xlabel("True trait")
        ax.set_ylabel("Predicted mean")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_trait_residuals(pred_df: pd.DataFrame, out_dir: str = ".", filename: str = "Trait_Residuals.pdf") -> None:
    _ensure_dir(out_dir)
    traits = list(pred_df["trait"].drop_duplicates())
    n = len(traits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, trait in zip(axes.ravel(), traits):
        df = pred_df[pred_df["trait"] == trait].copy()
        res = df["y_true"] - df["y_mu"]
        res = res[np.isfinite(res)]
        ax.hist(res, bins=min(30, max(5, len(res) // 3)))
        ax.set_title(f"{trait} residuals")
        ax.set_xlabel("True - predicted")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_trait_prediction_intervals(pred_df: pd.DataFrame, out_dir: str = ".", filename: str = "Trait_Uncertainty_Intervals.pdf") -> None:
    _ensure_dir(out_dir)
    traits = list(pred_df["trait"].drop_duplicates())
    n = len(traits)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(4, 3.5 * n)), squeeze=False)
    for ax, trait in zip(axes.ravel(), traits):
        df = pred_df[pred_df["trait"] == trait].copy()
        df = df[np.isfinite(df["y_mu"])].sort_values("y_mu").reset_index(drop=True)
        x = np.arange(len(df))
        ax.errorbar(x, df["y_mu"], yerr=1.96 * df["y_std"], fmt="o", markersize=3, linewidth=0.8, alpha=0.8)
        ok = np.isfinite(df["y_true"])
        ax.scatter(x[ok], df.loc[ok, "y_true"], s=18, marker="x", label="Observed")
        ax.set_title(f"{trait}: predicted mean with 95% interval")
        ax.set_xlabel("Samples ordered by predicted mean")
        ax.set_ylabel(trait)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_calibration(pred_df: pd.DataFrame, out_dir: str = ".", filename: str = "Trait_Uncertainty_Calibration.pdf") -> None:
    _ensure_dir(out_dir)
    traits = list(pred_df["trait"].drop_duplicates())
    fig, axes = plt.subplots(1, len(traits), figsize=(5 * len(traits), 4), squeeze=False)
    for ax, trait in zip(axes.ravel(), traits):
        df = pred_df[pred_df["trait"] == trait].copy()
        err = np.abs(df["y_true"] - df["y_mu"])
        std = df["y_std"]
        ok = np.isfinite(err) & np.isfinite(std)
        ax.scatter(std[ok], err[ok], s=35, alpha=0.8)
        r, _ = _safe_pearson(std[ok], err[ok])
        ax.set_title(f"{trait}: |error| vs predicted SD\nr={r:.3f}")
        ax.set_xlabel("Predicted SD")
        ax.set_ylabel("Absolute error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Latent-trait correlations
# -----------------------------

def save_and_plot_latent_trait_correlations(latent, y_true, trait_names: list[str], out_dir: str = ".", prefix: str = "mu"):
    _ensure_dir(out_dir)
    Z = _to_numpy(latent)
    Y = _to_numpy(y_true)
    if Y.ndim == 1:
        Y = Y[:, None]
    corr = np.full((Z.shape[1], Y.shape[1]), np.nan)
    pval = np.full_like(corr, np.nan)
    rows = []
    for i in range(Z.shape[1]):
        for j, trait in enumerate(trait_names):
            r, p = _safe_pearson(Z[:, i], Y[:, j])
            corr[i, j] = r
            pval[i, j] = p
            rows.append({"latent_dim": f"{prefix}{i+1}", "trait": trait, "pearson": r, "pvalue": p})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(out_dir, f"Latent_Trait_Correlations_{prefix}_long.csv"), index=False)
    pd.DataFrame(corr, index=[f"{prefix}{i+1}" for i in range(Z.shape[1])], columns=trait_names).to_csv(os.path.join(out_dir, f"Latent_Trait_Correlations_{prefix}.csv"))
    pd.DataFrame(pval, index=[f"{prefix}{i+1}" for i in range(Z.shape[1])], columns=trait_names).to_csv(os.path.join(out_dir, f"Latent_Trait_Correlations_{prefix}_pvalues.csv"))

    fig, ax = plt.subplots(figsize=(max(4, len(trait_names) * 1.2), max(4, Z.shape[1] * 0.35)))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(np.arange(len(trait_names)))
    ax.set_xticklabels(trait_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(Z.shape[1]))
    ax.set_yticklabels([f"{prefix}{i+1}" for i in range(Z.shape[1])])
    ax.set_title("Latent-trait Pearson correlations")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Latent_Trait_Correlations_{prefix}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return long_df


# -----------------------------
# SNP PCA structure plots
# -----------------------------

def plot_snp_structure(structure_df: pd.DataFrame, out_dir: str = ".", filename: str = "SNP_structure_PCA.pdf") -> None:
    _ensure_dir(out_dir)
    if "PC1" not in structure_df or "PC2" not in structure_df:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    if "pca_cluster" in structure_df:
        for cl in sorted(structure_df["pca_cluster"].dropna().unique()):
            df = structure_df[structure_df["pca_cluster"] == cl]
            ax.scatter(df["PC1"], df["PC2"], s=35, alpha=0.85, label=f"cluster {cl}")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(structure_df["PC1"], structure_df["PC2"], s=35, alpha=0.85)
    ax.set_xlabel("SNP PC1")
    ax.set_ylabel("SNP PC2")
    ax.set_title("Inferred SNP structure")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(
    pve,
    out_dir: str = ".",
    filename: str = "SNP_structure_PCA_variance.pdf",
) -> None:

    _ensure_dir(out_dir)

    try:
        pve = np.asarray(pve, dtype=np.float64)

        if pve.size == 0:
            with open(
                os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
                "w"
            ) as f:
                f.write("PCA variance plot skipped because pve is empty.\n")
            return

        if not np.isfinite(pve).all():
            with open(
                os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
                "w"
            ) as f:
                f.write(
                    "PCA variance plot skipped because pve contains NaN or Inf values.\n"
                )
            return

        if np.nansum(pve) <= 1e-12:
            with open(
                os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
                "w"
            ) as f:
                f.write(
                    "PCA variance plot skipped because total explained variance is ~0.\n"
                )
            return

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            np.arange(1, len(pve) + 1),
            pve
        )

        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title("SNP PCA variance")

        fig.tight_layout()

        fig.savefig(
            os.path.join(out_dir, filename),
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:

        with open(
            os.path.join(out_dir, filename.replace(".pdf", "_SKIPPED.txt")),
            "w"
        ) as f:
            f.write(f"PCA variance plotting failed: {e}\n")

        return


# -----------------------------
# Ensemble uncertainty summaries
# -----------------------------

def summarize_ensemble_predictions(run_rows: pd.DataFrame, out_dir: str = ".") -> pd.DataFrame:
    """Aggregate test prediction CSVs from CV/ensemble runs.

    Input must contain out_dir, fold, ensemble_member columns. The per-run directory must
    contain test_predictions_with_uncertainty.csv.
    """
    _ensure_dir(out_dir)
    pred_tables = []
    for _, row in run_rows.iterrows():
        pred_path = os.path.join(str(row["out_dir"]), "test_predictions_with_uncertainty.csv")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        df["fold"] = int(row.get("fold", -1))
        df["ensemble_member"] = int(row.get("ensemble_member", 1))
        df["run_seed"] = int(row.get("seed", -1))
        pred_tables.append(df)
    if not pred_tables:
        return pd.DataFrame()
    all_pred = pd.concat(pred_tables, ignore_index=True)
    all_pred.to_csv(os.path.join(out_dir, "ensemble_all_test_predictions_long.csv"), index=False)

    group_cols = ["fold", "sample_name", "trait"]
    summary = all_pred.groupby(group_cols, dropna=False).agg(
        y_true=("y_true", "first"),
        mean_y_mu=("y_mu", "mean"),
        var_between_models=("y_mu", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
        mean_aleatoric_var=("y_std", lambda x: float(np.nanmean(np.asarray(x) ** 2))),
        n_models=("y_mu", "size"),
    ).reset_index()
    summary["total_var"] = summary["var_between_models"] + summary["mean_aleatoric_var"]
    summary["total_std"] = np.sqrt(summary["total_var"])
    summary["lower_95_total"] = summary["mean_y_mu"] - 1.96 * summary["total_std"]
    summary["upper_95_total"] = summary["mean_y_mu"] + 1.96 * summary["total_std"]
    summary.to_csv(os.path.join(out_dir, "ensemble_test_uncertainty_by_sample.csv"), index=False)
    return summary


def plot_ensemble_uncertainty(summary_df: pd.DataFrame, out_dir: str = ".", filename: str = "Ensemble_Uncertainty.pdf") -> None:
    _ensure_dir(out_dir)
    if summary_df.empty:
        return
    traits = list(summary_df["trait"].drop_duplicates())
    fig, axes = plt.subplots(len(traits), 1, figsize=(8, max(4, 3.5 * len(traits))), squeeze=False)
    for ax, trait in zip(axes.ravel(), traits):
        df = summary_df[summary_df["trait"] == trait].copy().sort_values("mean_y_mu").reset_index(drop=True)
        x = np.arange(len(df))
        ax.errorbar(x, df["mean_y_mu"], yerr=1.96 * df["total_std"], fmt="o", markersize=3, linewidth=0.8, alpha=0.8)
        ok = np.isfinite(df["y_true"])
        ax.scatter(x[ok], df.loc[ok, "y_true"], s=18, marker="x", label="Observed")
        ax.set_title(f"{trait}: ensemble mean with total 95% interval")
        ax.set_xlabel("Held-out samples ordered by ensemble mean")
        ax.set_ylabel(trait)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)
