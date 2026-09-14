# helper.py
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from evaluation import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Training helpers
# -------------------------

class EarlyStopper:
    def __init__(self, patience: int = 30, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")

    def early_stop(self, validation_metric: float) -> bool:
        if validation_metric < self.best_value - self.min_delta:
            self.best_value = validation_metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, obs_mask, eps: float = 1e-8):
        se = ((pred - target) ** 2) * obs_mask
        per_sample = se.sum(dim=1) / obs_mask.sum(dim=1).clamp_min(eps)
        return per_sample.mean()


def masked_mse_numpy(pred, target, mask, eps: float = 1e-8) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    mask = np.asarray(mask)
    se = ((pred - target) ** 2) * mask
    per_sample = se.sum(axis=1) / np.clip(mask.sum(axis=1), eps, None)
    return float(np.mean(per_sample))


def y_to_1d(y, trait_index: int = 0):
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    return y[:, trait_index]


def _safe_pearson(y_true, y_pred, eps: float = 1e-12):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return np.nan
    yt = y_true[valid]
    yp = y_pred[valid]
    if np.std(yt) < eps or np.std(yp) < eps:
        return np.nan
    return pearsonr(yt, yp)[0]


def trait_metrics(y_true, y_pred) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    out = {
        "mse_per_trait": [],
        "r2_per_trait": [],
        "pearson_per_trait": [],
    }

    for j in range(y_true.shape[1]):
        valid = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        yt = y_true[valid, j]
        yp = y_pred[valid, j]

        if len(yt) < 2:
            mse = np.nan
            r2 = np.nan
            corr = np.nan
        else:
            mse = mean_squared_error(yt, yp)
            r2 = r2_score(yt, yp)
            corr = _safe_pearson(yt, yp)

        out["mse_per_trait"].append(float(mse) if np.isfinite(mse) else np.nan)
        out["r2_per_trait"].append(float(r2) if np.isfinite(r2) else np.nan)
        out["pearson_per_trait"].append(float(corr) if np.isfinite(corr) else np.nan)

    out["mse_mean"] = float(np.nanmean(out["mse_per_trait"]))
    out["r2_mean"] = float(np.nanmean(out["r2_per_trait"]))
    out["pearson_mean"] = float(np.nanmean(out["pearson_per_trait"]))
    return out


# -------------------------
# Clustering helpers
# -------------------------

def perform_kmeans(final_embedding, labels, n_clusters: int):
    seeds = [0, 12, 21, 42, 1234]
    acc, nmi, ari, silhouette = [], [], [], []

    em = final_embedding.detach().cpu().numpy() if torch.is_tensor(final_embedding) else np.asarray(final_embedding)
    labels = np.asarray(labels)

    for seed in seeds:
        km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=seed, n_init=20)
        y_pred = km.fit_predict(em)
        nmi_, ari_, _, acc_ = evaluate(labels, y_pred)

        acc.append(acc_)
        nmi.append(nmi_)
        ari.append(ari_)

        if len(np.unique(y_pred)) > 1 and len(np.unique(y_pred)) < len(y_pred):
            silhouette.append(silhouette_score(em, y_pred, metric="euclidean"))
        else:
            silhouette.append(np.nan)

    return {
        "kmeans_acc_mean": float(np.nanmean(acc)),
        "kmeans_acc_std": float(np.nanstd(acc)),
        "kmeans_nmi_mean": float(np.nanmean(nmi)),
        "kmeans_nmi_std": float(np.nanstd(nmi)),
        "kmeans_ari_mean": float(np.nanmean(ari)),
        "kmeans_ari_std": float(np.nanstd(ari)),
        "silhouette_mean": float(np.nanmean(silhouette)),
        "silhouette_std": float(np.nanstd(silhouette)),
    }


def perform_knn(final_embedding, labels, n_neighbors: int = 5):
    seeds = [0, 12, 21, 42, 1234]
    acc = []

    em = final_embedding.detach().cpu().numpy() if torch.is_tensor(final_embedding) else np.asarray(final_embedding)
    labels = np.asarray(labels)

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            em,
            labels,
            test_size=0.25,
            random_state=seed,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
        k = min(n_neighbors, max(1, len(X_train) - 1))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    return {
        "knn_acc_mean": float(np.nanmean(acc)),
        "knn_acc_std": float(np.nanstd(acc)),
    }


# -------------------------
# SHAP for the trained VAE trait head only
# -------------------------

def _subsample_background(X, max_background: int = 80, random_state: int = 42):
    X = np.asarray(X)
    if len(X) <= max_background:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_background, replace=False)
    return X[idx]


def _subsample_eval(X, max_eval: int = 100, random_state: int = 43):
    X = np.asarray(X)
    if len(X) <= max_eval:
        return X, np.arange(len(X))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_eval, replace=False)
    idx = np.sort(idx)
    return X[idx], idx


def shap_trained_trait_head(
    model,
    Z_mu,
    device=device,
    max_background: int = 80,
    max_eval: int = 100,
    random_state: int = 42,
):
    """
    SHAP for the trained VAE trait head only.

    Input:
        Z_mu: deterministic latent means, shape [n_samples, latent_dim]

    Explained function:
        mu -> trained trait head -> predicted trait mean y_mu

    This replaces the old posthoc Linear/Ridge/Lasso/ElasticNet SHAP.
    """
    import shap

    model.eval()
    Z_mu = np.asarray(Z_mu, dtype=np.float32)

    background = _subsample_background(Z_mu, max_background=max_background, random_state=random_state)
    Z_eval, eval_idx = _subsample_eval(Z_mu, max_eval=max_eval, random_state=random_state + 1)

    def f(z_numpy):
        z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model.predict_traits(z_tensor)

            # Gaussian head: returns (y_mu, y_logvar)
            if isinstance(pred, tuple):
                y_mu = pred[0]
            else:
                y_mu = pred

            return y_mu.detach().cpu().numpy().reshape(z_numpy.shape[0], -1)[:, 0]

    explainer = shap.KernelExplainer(f, background)
    values = explainer.shap_values(Z_eval, nsamples="auto")

    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]

    abs_mean = np.mean(np.abs(values), axis=0)
    signed_mean = np.mean(values, axis=0)

    df = pd.DataFrame({
        "latent_dim": np.arange(1, values.shape[1] + 1),
        "shap_abs_mean": abs_mean,
        "shap_signed_mean": signed_mean,
    }).sort_values("shap_abs_mean", ascending=False)

    return {
        "values": values,
        "eval_idx": eval_idx,
        "df": df,
    }


# Backwards-compatible alias, so old train.py calls do not immediately break.
def shap_neural_head(model, Z, device=device, max_background: int = 80, max_eval: int = 100):
    return shap_trained_trait_head(
        model=model,
        Z_mu=Z,
        device=device,
        max_background=max_background,
        max_eval=max_eval,
    )


# -------------------------
# SNP perturbation importance
# -------------------------

def perturbation_snp_trait_effects(
    model,
    X_filled,
    snp_names,
    device=device,
    delta: float = 1.0,
    batch_size: int = 64,
    max_snps: Optional[int] = None,
):
    """
    Signed SNP effect using input perturbation.

    For each SNP j:
        y_base = f(X)
        y_plus = f(X with SNP j increased by delta)
        effect_j = mean(y_plus - y_base)

    This respects the nonlinear encoder and does NOT treat decoder W as a matrix projection.
    """
    model.eval()
    X = np.asarray(X_filled, dtype=np.float32)
    snp_names = list(snp_names)

    n, p = X.shape
    if max_snps is not None:
        p_eval = min(p, max_snps)
    else:
        p_eval = p

    def predict_mu_y(X_np):
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_np), batch_size):
                xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=device)
                out = model(xb, deterministic=True) if "deterministic" in model.forward.__code__.co_varnames else model(xb)
                if "y_mu" in out:
                    y = out["y_mu"]
                else:
                    y = out["y_hat"]
                preds.append(y.detach().cpu().numpy())
        return np.vstack(preds)

    y_base = predict_mu_y(X)

    rows = []
    for j in range(p_eval):
        Xp = X.copy()
        Xp[:, j] = Xp[:, j] + delta
        y_plus = predict_mu_y(Xp)
        diff = y_plus - y_base

        rows.append({
            "snp_name": snp_names[j],
            "mean_delta_y": float(np.nanmean(diff[:, 0])),
            "abs_mean_delta_y": float(np.nanmean(np.abs(diff[:, 0]))),
            "std_delta_y": float(np.nanstd(diff[:, 0])),
        })

    df = pd.DataFrame(rows).sort_values("abs_mean_delta_y", ascending=False)
    return df


# -------------------------
# Ensemble uncertainty
# -------------------------

def combine_ensemble_predictions(prediction_tables, out_csv: Optional[str] = None):
    """
    Combine per-model prediction CSVs.

    Expected columns in each table:
        sample_name, y_true, y_mu, y_std

    Output:
        mean_y_mu
        epistemic_var = variance of y_mu across models
        aleatoric_var = mean(y_std^2)
        total_var = epistemic_var + aleatoric_var
        total_std
    """
    dfs = []
    for i, tab in enumerate(prediction_tables):
        df = pd.read_csv(tab) if isinstance(tab, str) else tab.copy()
        df["model_id"] = i
        dfs.append(df)

    long_df = pd.concat(dfs, ignore_index=True)

    grouped = long_df.groupby("sample_name", as_index=False).agg(
        y_true=("y_true", "first"),
        mean_y_mu=("y_mu", "mean"),
        epistemic_var=("y_mu", "var"),
        aleatoric_var=("y_std", lambda x: float(np.mean(np.asarray(x) ** 2))),
        n_models=("model_id", "nunique"),
    )

    grouped["epistemic_var"] = grouped["epistemic_var"].fillna(0.0)
    grouped["total_var"] = grouped["epistemic_var"] + grouped["aleatoric_var"]
    grouped["total_std"] = np.sqrt(grouped["total_var"])
    grouped["lower_95"] = grouped["mean_y_mu"] - 1.96 * grouped["total_std"]
    grouped["upper_95"] = grouped["mean_y_mu"] + 1.96 * grouped["total_std"]

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        grouped.to_csv(out_csv, index=False)

    return grouped, long_df
