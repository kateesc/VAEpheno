import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import SNPPOEMSVAE
from evaluation import evaluate_genotype_reconstruction
import setup_seed
import util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.path.abspath(os.path.dirname(Path(__file__).resolve()))
root_dir = os.getcwd()
sys.path.insert(1, root_dir)


# -----------------------------
# data loading and preprocessing
# -----------------------------

def load_plant_data(data_dir, trait_name=None, require_labels=False):
    snp_path = os.path.join(data_dir, "1_all.csv")
    snp_name_path = os.path.join(data_dir, "1_featname.csv")
    trait_path = os.path.join(data_dir, "2_all.csv")
    trait_name_path = os.path.join(data_dir, "2_featname.csv")
    label_path = os.path.join(data_dir, "labels_all.csv")
    label_map_path = os.path.join(data_dir, "labels_map.csv")
    samples_path = os.path.join(data_dir, "samples.txt")

    snp_df = pd.read_csv(snp_path, header=None, na_values=["nan", "NA", "", "NaN"])
    trait_df = pd.read_csv(trait_path, header=None, na_values=["nan", "NA", "", "NaN"])
    snp_names = pd.read_csv(snp_name_path, header=None).iloc[:, 0].astype(str).tolist()
    trait_names = pd.read_csv(trait_name_path, header=None).iloc[:, 0].astype(str).tolist()

    if os.path.exists(label_path):
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].astype(int).to_numpy()
        has_labels = True
    else:
        if require_labels:
            raise FileNotFoundError(f"labels_all.csv not found: {label_path}")
        labels = np.zeros(snp_df.shape[0], dtype=int)
        has_labels = False

    if os.path.exists(samples_path):
        sample_names = pd.read_csv(samples_path, header=None).iloc[:, 0].astype(str).tolist()
    else:
        sample_names = [f"sample_{i}" for i in range(snp_df.shape[0])]

    label_map = None
    if os.path.exists(label_map_path):
        tmp = pd.read_csv(label_map_path)
        label_map = {int(r["label"]): str(r["population"]) for _, r in tmp.iterrows()}

    if len(snp_names) != snp_df.shape[1]:
        raise ValueError("Mismatch between 1_all.csv columns and 1_featname.csv rows.")
    if len(trait_names) != trait_df.shape[1]:
        raise ValueError("Mismatch between 2_all.csv columns and 2_featname.csv rows.")
    if snp_df.shape[0] != trait_df.shape[0]:
        raise ValueError("Mismatch between sample counts in SNP and trait matrices.")
    if len(labels) != snp_df.shape[0]:
        raise ValueError("Mismatch between labels_all.csv rows and sample count.")
    if len(sample_names) != snp_df.shape[0]:
        raise ValueError("Mismatch between samples.txt rows and sample count.")

    x_raw = snp_df.to_numpy(dtype=np.float32)
    traits = trait_df.to_numpy(dtype=np.float32)

    if trait_name is None:
        y_raw = traits
        selected_trait_names = trait_names
    else:
        if trait_name not in trait_names:
            raise ValueError(f"Trait '{trait_name}' not found. Available traits: {trait_names}")
        idx = trait_names.index(trait_name)
        y_raw = traits[:, [idx]]
        selected_trait_names = [trait_name]

    return {
        "x_raw": x_raw,
        "obs_mask": (~np.isnan(x_raw)).astype(np.float32),
        "y_raw": y_raw,
        "trait_mask": (~np.isnan(y_raw)).astype(np.float32),
        "labels": labels,
        "has_labels": has_labels,
        "sample_names": np.asarray(sample_names),
        "label_map": label_map,
        "snp_names": snp_names,
        "trait_names": selected_trait_names,
        "all_trait_names": trait_names,
    }


def fit_train_only_preprocessing(x_raw, y_raw, train_idx):
    """Fit SNP imputation and trait scaling on training samples only."""
    x_train = x_raw[train_idx]
    y_train = y_raw[train_idx]

    x_means = np.nanmean(x_train, axis=0)
    x_means = np.where(np.isnan(x_means), 0.0, x_means).astype(np.float32)

    y_fill_means = np.nanmean(y_train, axis=0)
    y_fill_means = np.where(np.isnan(y_fill_means), 0.0, y_fill_means).astype(np.float32)

    y_train_filled = np.where(np.isnan(y_train), y_fill_means[None, :], y_train).astype(np.float32)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train_filled)

    def transform_x(x):
        return np.where(np.isnan(x), x_means[None, :], x).astype(np.float32)

    def transform_y(y):
        y_filled = np.where(np.isnan(y), y_fill_means[None, :], y).astype(np.float32)
        return y_scaler.transform(y_filled).astype(np.float32)

    return transform_x, transform_y, y_scaler, x_means, y_fill_means


# -----------------------------
# splitting strategies
# -----------------------------

def trait_outlier_indices(y_raw, quantile=0.05):
    """Return indices in the lower/upper tail of each trait."""
    y = np.asarray(y_raw, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    out = set()
    for j in range(y.shape[1]):
        col = y[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 5:
            continue
        lo = np.nanquantile(col, quantile)
        hi = np.nanquantile(col, 1.0 - quantile)
        out.update(np.where(valid & ((col <= lo) | (col >= hi)))[0].tolist())
    return np.array(sorted(out), dtype=int)


def split_train_val_test(
    n,
    y_raw,
    labels=None,
    has_labels=False,
    seed=21,
    test_size=0.15,
    val_size=0.15,
    split_strategy="auto",  # auto, random, stratified
    force_trait_outliers_train=True,
    outlier_quantile=0.05,
):
    idx = np.arange(n)
    protected = trait_outlier_indices(y_raw, outlier_quantile) if force_trait_outliers_train else np.array([], dtype=int)
    candidate = np.setdiff1d(idx, protected)

    use_stratify = (
        split_strategy in {"auto", "stratified"}
        and has_labels
        and labels is not None
        and len(np.unique(labels[candidate])) > 1
        and np.min(np.bincount(labels[candidate])) >= 2
    )
    strat = labels[candidate] if use_stratify else None

    trainval_cand, test_idx = train_test_split(
        candidate,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )

    val_fraction_of_trainval = val_size / (1.0 - test_size)
    use_stratify_2 = use_stratify and np.min(np.bincount(labels[trainval_cand])) >= 2
    strat2 = labels[trainval_cand] if use_stratify_2 else None
    train_cand, val_idx = train_test_split(
        trainval_cand,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        shuffle=True,
        stratify=strat2,
    )

    train_idx = np.sort(np.concatenate([train_cand, protected]))
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)
    return train_idx, val_idx, test_idx, protected


def infer_snp_structure(x_raw, train_idx=None, n_pcs=10, n_clusters=3, random_state=21):
    """
    Infer unsupervised SNP structure for visualization/confounding/blocking.

    If train_idx is provided:
      * SNP imputation means are fitted on training samples only.
      * Standardization is fitted on training samples only.
      * PCA is fitted on training samples only.
      * All samples are then projected into that training-derived PCA basis.

    If train_idx is None, the full dataset is used. This is appropriate for
    deliberately global descriptive/blocking analyses, but not for a strict
    prospective predictive benchmark.
    """
    x_raw = np.asarray(x_raw, dtype=np.float32)

    if train_idx is None:
        fit_idx = np.arange(x_raw.shape[0], dtype=int)
    else:
        fit_idx = np.asarray(train_idx, dtype=int)

    fit_x = x_raw[fit_idx]
    means = np.nanmean(fit_x, axis=0)
    means = np.where(np.isnan(means), 0.0, means).astype(np.float32)

    x_filled = np.where(np.isnan(x_raw), means[None, :], x_raw).astype(np.float32)
    fit_filled = x_filled[fit_idx]

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(fit_filled)

    fit_scaled = scaler.transform(fit_filled)
    all_scaled = scaler.transform(x_filled)

    n_pcs_eff = min(
        int(n_pcs),
        fit_scaled.shape[0] - 1,
        fit_scaled.shape[1],
    )
    if n_pcs_eff < 1:
        raise ValueError("Not enough training samples/features for SNP PCA.")

    pca = PCA(n_components=n_pcs_eff, random_state=random_state)
    pca.fit(fit_scaled)
    pcs = pca.transform(all_scaled)

    # Clustering is fitted on the PCA coordinates used for the requested
    # structure analysis. For train_idx=None this is intentionally global.
    # For train_idx!=None, fit KMeans on training PCs and predict all samples.
    n_clusters_eff = min(int(n_clusters), len(fit_idx))
    if n_clusters_eff >= 2:
        km = KMeans(
            n_clusters=n_clusters_eff,
            n_init=50,
            random_state=random_state,
        )
        km.fit(pcs[fit_idx])
        clusters = km.predict(pcs)
    else:
        clusters = np.zeros(x_raw.shape[0], dtype=int)

    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    structure_df = pd.DataFrame(pcs, columns=pc_cols)
    structure_df["pca_cluster"] = clusters
    return structure_df, pca.explained_variance_ratio_, clusters


# -----------------------------
# dataset, losses, metrics
# -----------------------------

class SNPDataset(torch.utils.data.Dataset):
    def __init__(self, x_raw, x_in, obs_mask, y_scaled, y_raw, trait_mask, labels, sample_names):
        self.x_raw = torch.tensor(x_raw, dtype=torch.float32)
        self.x_in = torch.tensor(x_in, dtype=torch.float32)
        self.obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
        self.y = torch.tensor(y_scaled, dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.trait_mask = torch.tensor(trait_mask, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_names = np.asarray(sample_names)

    def __len__(self):
        return self.x_raw.shape[0]

    def __getitem__(self, idx):
        return {
            "x_raw": self.x_raw[idx],
            "x_in": self.x_in[idx],
            "obs_mask": self.obs_mask[idx],
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "trait_mask": self.trait_mask[idx],
            "label": self.labels[idx],
            "sample_name": self.sample_names[idx],
        }


def gaussian_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def masked_mse_loss(pred, target, obs_mask, eps=1e-8):
    diff = torch.where(obs_mask > 0, pred - target, torch.zeros_like(pred))
    per_sample = (diff ** 2).sum(dim=1) / obs_mask.sum(dim=1).clamp_min(eps)
    return per_sample.mean()


def validate_genotype_encoding(x_raw, genotype_likelihood, n_genotype_classes=3):
    """Fail early when the requested genotype likelihood does not match the data."""
    likelihood = str(genotype_likelihood).lower()
    observed = np.asarray(x_raw, dtype=float)
    observed = observed[np.isfinite(observed)]

    if observed.size == 0:
        raise ValueError("No observed genotype values were found.")

    if likelihood == "bernoulli":
        vals = np.unique(observed)
        if not np.all(np.isin(vals, [0.0, 1.0])):
            raise ValueError(
                "Bernoulli SNP reconstruction requires observed values in {0,1}. "
                f"Observed values include: {vals[:20]}"
            )

    elif likelihood == "categorical":
        rounded = np.round(observed)
        if not np.allclose(observed, rounded):
            raise ValueError(
                "Categorical SNP reconstruction requires integer-coded genotype classes."
            )
        vals = rounded.astype(int)
        if vals.min() < 0 or vals.max() >= int(n_genotype_classes):
            raise ValueError(
                f"Categorical SNP values must be in [0,{int(n_genotype_classes)-1}], "
                f"but observed range is [{vals.min()},{vals.max()}]."
            )

    elif likelihood != "mse":
        raise ValueError(
            "genotype_likelihood must be one of {'mse','bernoulli','categorical'}."
        )


def masked_bernoulli_bce_with_logits(logits, target, obs_mask, eps=1e-8):
    """
    Bernoulli negative log-likelihood averaged per observed SNP per sample.

    Missing target entries are replaced before BCE is evaluated so NaNs do not
    propagate through the element-wise loss.
    """
    target_safe = torch.where(
        obs_mask > 0,
        target,
        torch.zeros_like(target),
    )
    element = F.binary_cross_entropy_with_logits(
        logits,
        target_safe,
        reduction="none",
    )
    element = torch.where(
        obs_mask > 0,
        element,
        torch.zeros_like(element),
    )
    per_sample = (
        element.sum(dim=1)
        / obs_mask.sum(dim=1).clamp_min(eps)
    )
    return per_sample.mean()


def masked_categorical_cross_entropy(
    logits,
    target,
    obs_mask,
    n_genotype_classes,
    eps=1e-8,
):
    """
    Categorical genotype NLL averaged per observed SNP per sample.

    logits: [B, P, C]
    target: [B, P] integer-coded genotype classes
    """
    if logits.ndim != 3:
        raise ValueError("Categorical logits must have shape [B,P,C].")
    if logits.shape[-1] != int(n_genotype_classes):
        raise ValueError(
            f"Decoder returned {logits.shape[-1]} classes but "
            f"n_genotype_classes={n_genotype_classes}."
        )

    target_safe = torch.where(
        obs_mask > 0,
        target,
        torch.zeros_like(target),
    ).long()

    B, P, C = logits.shape
    element = F.cross_entropy(
        logits.reshape(B * P, C),
        target_safe.reshape(B * P),
        reduction="none",
    ).view(B, P)

    element = torch.where(
        obs_mask > 0,
        element,
        torch.zeros_like(element),
    )
    per_sample = (
        element.sum(dim=1)
        / obs_mask.sum(dim=1).clamp_min(eps)
    )
    return per_sample.mean()


def masked_genotype_reconstruction_loss(
    model,
    model_out,
    target,
    obs_mask,
    eps=1e-8,
):
    likelihood = model.genotype_likelihood

    if likelihood == "mse":
        return masked_mse_loss(
            model_out["x_hat"],
            target,
            obs_mask,
            eps=eps,
        )

    if likelihood == "bernoulli":
        return masked_bernoulli_bce_with_logits(
            model_out["x_logits"],
            target,
            obs_mask,
            eps=eps,
        )

    if likelihood == "categorical":
        return masked_categorical_cross_entropy(
            model_out["x_logits"],
            target,
            obs_mask,
            n_genotype_classes=model.n_genotype_classes,
            eps=eps,
        )

    raise ValueError(f"Unknown genotype_likelihood: {likelihood}")


def masked_genotype_hard_accuracy(model, model_out, target, obs_mask):
    """Descriptive hard-call accuracy; not part of the optimization objective."""
    mask = obs_mask > 0
    if not torch.any(mask):
        return torch.tensor(float("nan"), device=target.device)

    target_safe = torch.where(
        mask,
        target,
        torch.zeros_like(target),
    )

    if model.genotype_likelihood == "bernoulli":
        pred = (model_out["x_probs"] >= 0.5).to(target_safe.dtype)
    elif model.genotype_likelihood == "categorical":
        pred = torch.argmax(model_out["x_logits"], dim=-1).to(target_safe.dtype)
    else:
        # MSE is not intrinsically a classification model, but rounding provides
        # a useful descriptive diagnostic when the targets are discrete.
        pred = torch.round(model_out["x_hat"]).to(target_safe.dtype)

    return (pred[mask] == target_safe[mask]).float().mean()


def masked_trait_gaussian_nll(y_mu, y_logvar, target, trait_mask, eps=1e-8):
    var = torch.exp(y_logvar).clamp_min(eps)
    nll = 0.5 * (y_logvar + (target - y_mu).pow(2) / var)
    nll = torch.where(trait_mask > 0, nll, torch.zeros_like(nll))
    per_sample = nll.sum(dim=1) / trait_mask.sum(dim=1).clamp_min(eps)
    return per_sample.mean()


def masked_trait_mse_loss(y_mu, target, trait_mask, eps=1e-8):
    diff = torch.where(trait_mask > 0, y_mu - target, torch.zeros_like(y_mu))
    per_sample = (diff ** 2).sum(dim=1) / trait_mask.sum(dim=1).clamp_min(eps)
    return per_sample.mean()


def _safe_pearson(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    ok = np.isfinite(y_true) & np.isfinite(y_pred)

    if ok.sum() < 3:
        return np.nan

    yt = y_true[ok]
    yp = y_pred[ok]

    # Stronger constant checks than std alone
    if np.ptp(yt) < eps or np.ptp(yp) < eps:
        return np.nan

    if np.std(yt) < eps or np.std(yp) < eps:
        return np.nan

    try:
        r, _ = pearsonr(yt, yp)
        return float(r) if np.isfinite(r) else np.nan
    except Exception:
        return np.nan

def trait_metrics_unscaled(y_true, y_pred, trait_names):
    rows = []
    for j, tr in enumerate(trait_names):
        ok = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if ok.sum() < 2:
            rows.append({"trait": tr, "mse": np.nan, "r2": np.nan, "pearson": np.nan, "n_valid": int(ok.sum())})
        else:
            rows.append({
                "trait": tr,
                "mse": float(mean_squared_error(y_true[ok, j], y_pred[ok, j])),
                "r2": float(r2_score(y_true[ok, j], y_pred[ok, j])),
                "pearson": _safe_pearson(y_true[ok, j], y_pred[ok, j]),
                "n_valid": int(ok.sum()),
            })
    return pd.DataFrame(rows)


def run_epoch(
    model,
    loader,
    optimizer,
    beta_kl,
    alpha_trait,
    decoder_l1_lambda,
    train=True,
):
    model.train() if train else model.eval()
    deterministic = not train

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "trait": 0.0,
        "decoder_reg": 0.0,
        "geno_acc": 0.0,
    }
    n_batches = 0

    # Keep the same compact saved outputs as the old pipeline.
    # We intentionally do not concatenate full x_logits/x_probs because this
    # can become very large for 75k SNPs, especially for categorical output.
    store = {
        k: []
        for k in [
            "mu",
            "z",
            "y",
            "y_raw",
            "y_mu",
            "y_logvar",
            "y_std",
            "labels",
            "x_raw",
            "x_in",
            "x_hat",
            "obs_mask",
            "trait_mask",
        ]
    }
    sample_names_all = []

    for batch in loader:
        x_raw = batch["x_raw"].to(device)
        x_in = batch["x_in"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        y = batch["y"].to(device)
        y_raw = batch["y_raw"].to(device)
        trait_mask = batch["trait_mask"].to(device)
        labels = batch["label"].to(device)
        sample_names = batch["sample_name"]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(
                x_in,
                deterministic=deterministic,
            )

            for key in ["mu", "logvar", "z", "x_hat", "y_mu"]:
                if torch.isnan(out[key]).any():
                    raise ValueError(
                        f"NaN detected in model output: {key}"
                    )
            if out.get("x_logits") is not None and torch.isnan(out["x_logits"]).any():
                raise ValueError("NaN detected in model output: x_logits")

            recon = masked_genotype_reconstruction_loss(
                model,
                out,
                x_raw,
                obs_mask,
            )
            kl = gaussian_kl(
                out["mu"],
                out["logvar"],
            ).mean()

            if out["y_logvar"] is None:
                trait_loss = masked_trait_mse_loss(
                    out["y_mu"],
                    y,
                    trait_mask,
                )
            else:
                trait_loss = masked_trait_gaussian_nll(
                    out["y_mu"],
                    out["y_logvar"],
                    y,
                    trait_mask,
                )

            decoder_reg = (
                float(decoder_l1_lambda)
                * model.decoder_regularization()
            )

            loss = (
                recon
                + beta_kl * kl
                + alpha_trait * trait_loss
                + decoder_reg
            )

            geno_acc = masked_genotype_hard_accuracy(
                model,
                out,
                x_raw,
                obs_mask,
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()

        totals["loss"] += float(loss.item())
        totals["recon"] += float(recon.item())
        totals["kl"] += float(kl.item())
        totals["trait"] += float(trait_loss.item())
        totals["decoder_reg"] += float(decoder_reg.item())
        totals["geno_acc"] += float(geno_acc.item())
        n_batches += 1

        store["mu"].append(out["mu"].detach().cpu())
        store["z"].append(out["z"].detach().cpu())
        store["y"].append(y.detach().cpu())
        store["y_raw"].append(y_raw.detach().cpu())
        store["y_mu"].append(out["y_mu"].detach().cpu())

        if out["y_logvar"] is None:
            store["y_logvar"].append(
                torch.full_like(
                    out["y_mu"],
                    float("nan"),
                ).detach().cpu()
            )
            store["y_std"].append(
                torch.full_like(
                    out["y_mu"],
                    float("nan"),
                ).detach().cpu()
            )
        else:
            store["y_logvar"].append(
                out["y_logvar"].detach().cpu()
            )
            store["y_std"].append(
                out["y_std"].detach().cpu()
            )

        store["labels"].append(labels.detach().cpu())
        store["x_raw"].append(x_raw.detach().cpu())
        store["x_in"].append(x_in.detach().cpu())
        store["x_hat"].append(out["x_hat"].detach().cpu())
        store["obs_mask"].append(obs_mask.detach().cpu())
        store["trait_mask"].append(trait_mask.detach().cpu())
        sample_names_all.extend(list(sample_names))

    metrics = {
        k: v / max(n_batches, 1)
        for k, v in totals.items()
    }
    # Backward-compatible alias used in some older output-processing code.
    metrics["w_l1"] = metrics["decoder_reg"]

    for key, vals in store.items():
        metrics[key] = torch.cat(vals, dim=0)

    metrics["sample_names"] = sample_names_all
    metrics["y_hat"] = metrics["y_mu"]
    return metrics


def choose_early_stop_value(metrics, early_stop_metric="trait"):
    if early_stop_metric == "total":
        return metrics["loss"]
    if early_stop_metric == "recon":
        return metrics["recon"]
    if early_stop_metric == "trait":
        return metrics["trait"]
    raise ValueError(f"Unknown early_stop_metric: {early_stop_metric}")


def make_history_row(epoch, train_metrics, val_metrics):
    return {
        "epoch": epoch,
        "train_rec_loss_all": train_metrics["recon"],
        "train_kl_loss_all": train_metrics["kl"],
        "train_trait_loss_all": train_metrics["trait"],
        "train_total_loss_all": train_metrics["loss"],
        "val_rec_loss_all": val_metrics["recon"],
        "val_kl_loss_all": val_metrics["kl"],
        "val_trait_loss_all": val_metrics["trait"],
        "val_total_loss_all": val_metrics["loss"],
    }


def make_prediction_table(metrics, y_scaler, trait_names):
    y_true = metrics["y_raw"].numpy()
    y_mu_scaled = metrics["y_mu"].numpy()
    y_mu = y_scaler.inverse_transform(y_mu_scaled)

    y_logvar_scaled = metrics["y_logvar"].numpy()
    y_std_scaled = np.exp(0.5 * y_logvar_scaled)
    # Approximate unscaled SD: scaled SD times training trait scaler scale.
    y_std = y_std_scaled * y_scaler.scale_[None, :]

    rows = []
    for i, sample in enumerate(metrics["sample_names"]):
        for j, trait in enumerate(trait_names):
            rows.append({
                "sample_name": sample,
                "trait": trait,
                "y_true": float(y_true[i, j]) if np.isfinite(y_true[i, j]) else np.nan,
                "y_mu": float(y_mu[i, j]),
                "y_std": float(y_std[i, j]) if np.isfinite(y_std[i, j]) else np.nan,
                "y_lower_95": float(y_mu[i, j] - 1.96 * y_std[i, j]) if np.isfinite(y_std[i, j]) else np.nan,
                "y_upper_95": float(y_mu[i, j] + 1.96 * y_std[i, j]) if np.isfinite(y_std[i, j]) else np.nan,
                "label": int(metrics["labels"].numpy()[i]),
            })
    return pd.DataFrame(rows)


# -----------------------------
# main train function
# -----------------------------

def train_POEMS(
    lr_in,
    wd_in,
    batch_size_in,
    nepoch_in,
    experiment_note,
    dataset,
    trait_name,
    latent_dim,
    enc_hidden_dim,
    dec_hidden_dim,
    dropout,
    beta_kl,
    alpha_trait,
    decoder_l1_lambda,
    seed,
    early_stop_metric="trait",
    patience=30,
    is_test=False,
    regressor_type="mlp",
    trait_likelihood="gaussian",
    decoder_type="poems",
    genotype_likelihood="mse",
    dec_hidden_dim2=512,
    n_genotype_classes=3,
    split_strategy="auto",
    force_trait_outliers_train=True,
    outlier_quantile=0.05,
    train_idx_override=None,
    val_idx_override=None,
    test_idx_override=None,
    infer_structure=True,
    n_structure_pcs=10,
    n_structure_clusters=3,
    skip_interpretation=True,
):
    setup_seed.setup_seed(seed)

    if decoder_type == "dense_mlp" and float(decoder_l1_lambda) != 0.0:
        print(
            "WARNING: dense_mlp does not use the POEMS gate W. "
            "For the controlled decoder ablation, decoder_l1_lambda should normally be 0."
        )

    data_dir = os.path.join(root_dir, "data", dataset)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    data = load_plant_data(data_dir, trait_name=trait_name, require_labels=False)
    x_raw = data["x_raw"]
    obs_mask = data["obs_mask"]
    y_raw = data["y_raw"]
    trait_mask = data["trait_mask"]
    labels = data["labels"]
    sample_names = data["sample_names"]
    n = x_raw.shape[0]

    validate_genotype_encoding(
        x_raw,
        genotype_likelihood=genotype_likelihood,
        n_genotype_classes=n_genotype_classes,
    )

    if train_idx_override is not None and val_idx_override is not None and test_idx_override is not None:
        train_idx = np.asarray(train_idx_override, dtype=int)
        val_idx = np.asarray(val_idx_override, dtype=int)
        test_idx = np.asarray(test_idx_override, dtype=int)
        protected_outliers = np.array([], dtype=int)
    else:
        train_idx, val_idx, test_idx, protected_outliers = split_train_val_test(
            n=n,
            y_raw=y_raw,
            labels=labels,
            has_labels=data["has_labels"],
            seed=seed,
            split_strategy=split_strategy,
            force_trait_outliers_train=force_trait_outliers_train,
            outlier_quantile=outlier_quantile,
        )

    transform_x, transform_y, y_scaler, x_means, y_fill_means = fit_train_only_preprocessing(
        x_raw=x_raw,
        y_raw=y_raw,
        train_idx=train_idx,
    )
    x_filled = transform_x(x_raw)
    y_scaled = transform_y(y_raw)

    train_ds = SNPDataset(x_raw[train_idx], x_filled[train_idx], obs_mask[train_idx], y_scaled[train_idx], y_raw[train_idx], trait_mask[train_idx], labels[train_idx], sample_names[train_idx])
    val_ds = SNPDataset(x_raw[val_idx], x_filled[val_idx], obs_mask[val_idx], y_scaled[val_idx], y_raw[val_idx], trait_mask[val_idx], labels[val_idx], sample_names[val_idx])
    test_ds = SNPDataset(x_raw[test_idx], x_filled[test_idx], obs_mask[test_idx], y_scaled[test_idx], y_raw[test_idx], trait_mask[test_idx], labels[test_idx], sample_names[test_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_in, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size_in, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_in, shuffle=False, drop_last=False)

    trait_tag = trait_name if trait_name is not None else "all_traits"
    run_name = (
        "test_run"
        if is_test
        else
        f"{experiment_note}__{dataset}__{trait_tag}"
        f"__decoder-{decoder_type}__geno-{genotype_likelihood}"
        f"__{regressor_type}_{trait_likelihood}"
        f"__lr{lr_in}__wd{wd_in}__bs{batch_size_in}__z{latent_dim}"
        f"__a{alpha_trait}__b{beta_kl}__es{early_stop_metric}__seed{seed}"
    )
    out_dir = os.path.join(cwd, "results", run_name)
    model_dir = os.path.join(cwd, "trained", run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save split indices and outlier protection information.
    pd.DataFrame({"train_idx": train_idx}).to_csv(os.path.join(out_dir, "split_train_idx.csv"), index=False)
    pd.DataFrame({"val_idx": val_idx}).to_csv(os.path.join(out_dir, "split_val_idx.csv"), index=False)
    pd.DataFrame({"test_idx": test_idx}).to_csv(os.path.join(out_dir, "split_test_idx.csv"), index=False)
    pd.DataFrame({"forced_train_outlier_idx": protected_outliers}).to_csv(os.path.join(out_dir, "forced_train_outlier_idx.csv"), index=False)

    run_config = {
        "dataset": dataset,
        "trait_name": trait_name,
        "seed": int(seed),
        "lr": float(lr_in),
        "wd": float(wd_in),
        "batch_size": int(batch_size_in),
        "epochs": int(nepoch_in),
        "latent_dim": int(latent_dim),
        "enc_hidden_dim": int(enc_hidden_dim),
        "dec_hidden_dim": int(dec_hidden_dim),
        "dec_hidden_dim2": int(dec_hidden_dim2),
        "decoder_type": decoder_type,
        "genotype_likelihood": genotype_likelihood,
        "n_genotype_classes": int(n_genotype_classes),
        "dropout": float(dropout),
        "beta_kl": float(beta_kl),
        "alpha_trait": float(alpha_trait),
        "decoder_l1_lambda": float(decoder_l1_lambda),
        "regressor_type": regressor_type,
        "trait_likelihood": trait_likelihood,
        "early_stop_metric": early_stop_metric,
        "patience": int(patience),
        "split_strategy": split_strategy,
    }
    with open(os.path.join(out_dir, "run_configuration.json"), "w") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    if infer_structure:
        structure_df, pve, pca_clusters = infer_snp_structure(
            x_raw=x_raw,
            train_idx=train_idx,
            n_pcs=n_structure_pcs,
            n_clusters=n_structure_clusters,
            random_state=seed,
        )
        structure_df.insert(0, "sample_name", sample_names)
        structure_df["label"] = labels
        structure_df.to_csv(os.path.join(out_dir, "SNP_structure_PCA_clusters.csv"), index=False)
        pd.DataFrame({"PC": np.arange(1, len(pve) + 1), "explained_variance_ratio": pve}).to_csv(
            os.path.join(out_dir, "SNP_structure_PCA_variance.csv"), index=False
        )
        util.plot_snp_structure(structure_df, out_dir=out_dir, filename="SNP_structure_PCA.pdf")
        util.plot_pca_variance(pve, out_dir=out_dir, filename="SNP_structure_PCA_variance.pdf")

    model = SNPPOEMSVAE(
        input_dim=x_raw.shape[1],
        latent_dim=latent_dim,
        enc_hidden_dim=enc_hidden_dim,
        dec_hidden_dim=dec_hidden_dim,
        dec_hidden_dim2=dec_hidden_dim2,
        n_genotype_classes=n_genotype_classes,
        n_traits=y_scaled.shape[1],
        dropout=dropout,
        regressor_type=regressor_type,
        trait_likelihood=trait_likelihood,
        decoder_type=decoder_type,
        genotype_likelihood=genotype_likelihood,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_in, weight_decay=wd_in)
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    wait = 0

    print("Training configuration")
    print(f"dataset: {dataset}")
    print(f"x shape: {x_raw.shape}")
    print(f"trait names: {data['trait_names']}")
    print(f"labels available: {data['has_labels']}")
    print(f"train / val / test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"regressor_type: {regressor_type}")
    print(f"trait_likelihood: {trait_likelihood}")
    print(f"decoder_type: {decoder_type}")
    print(f"genotype_likelihood: {genotype_likelihood}")
    print(f"dec_hidden_dim / dec_hidden_dim2: {dec_hidden_dim} / {dec_hidden_dim2}")
    print(f"n_genotype_classes: {n_genotype_classes}")
    print(f"decoder_l1_lambda: {decoder_l1_lambda}")
    print(f"forced trait outliers into train: {len(protected_outliers)}")

    for epoch in range(1, nepoch_in + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, beta_kl, alpha_trait, decoder_l1_lambda, train=True)
        val_metrics = run_epoch(model, val_loader, None, beta_kl, alpha_trait, decoder_l1_lambda, train=False)
        history.append(make_history_row(epoch, train_metrics, val_metrics))

        print(
            f"Epoch {epoch:03d} | "
            f"train total={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} trait={train_metrics['trait']:.4f} | "
            f"val total={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f} trait={val_metrics['trait']:.4f}"
        )

        current_stop = choose_early_stop_value(val_metrics, early_stop_metric)
        if current_stop < best_val:
            best_val = current_stop
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, beta_kl, alpha_trait, decoder_l1_lambda, train=False)
    val_eval = run_epoch(model, val_loader, None, beta_kl, alpha_trait, decoder_l1_lambda, train=False)
    test_eval = run_epoch(model, test_loader, None, beta_kl, alpha_trait, decoder_l1_lambda, train=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
    util.plot_training_history(history_df, out_dir=out_dir)

    prediction_tables = {}
    for split_name, metrics in [("train", train_eval), ("val", val_eval), ("test", test_eval)]:
        pred_df = make_prediction_table(metrics, y_scaler, data["trait_names"])
        prediction_tables[split_name] = pred_df
        pred_df.to_csv(os.path.join(out_dir, f"{split_name}_predictions_with_uncertainty.csv"), index=False)
        util.save_latent_csv(metrics, out_dir=out_dir, split_name=split_name, prefix="mu")
        y_true = metrics["y_raw"].numpy()
        y_pred = y_scaler.inverse_transform(metrics["y_mu"].numpy())
        trait_metrics_unscaled(y_true, y_pred, data["trait_names"]).to_csv(
            os.path.join(out_dir, f"{split_name}_Trait_Metrics.csv"), index=False
        )

        # Common reconstruction summary on x_hat. For Bernoulli this uses the
        # probability-scale reconstruction. For categorical the training NLL is
        # retained from run_epoch; x_hat is the expected dosage.
        recon_summary = {
            "split": split_name,
            "decoder_type": decoder_type,
            "genotype_likelihood": genotype_likelihood,
            "recon_loss_training_scale": float(metrics["recon"]),
            "hard_call_accuracy": float(metrics["geno_acc"]),
        }
        try:
            if genotype_likelihood in {"mse", "bernoulli"}:
                extra = evaluate_genotype_reconstruction(
                    metrics["x_raw"].numpy(),
                    genotype_likelihood=genotype_likelihood,
                    x_hat=metrics["x_hat"].numpy(),
                    x_probs=(
                        metrics["x_hat"].numpy()
                        if genotype_likelihood == "bernoulli"
                        else None
                    ),
                    obs_mask=metrics["obs_mask"].numpy(),
                )
                recon_summary.update(extra)
            else:
                # Full categorical class probabilities are intentionally not
                # concatenated across all SNPs to control memory. The exact
                # categorical NLL and hard-call accuracy are already reported.
                recon_summary["recon_nll"] = float(metrics["recon"])
        except Exception as exc:
            recon_summary["diagnostic_warning"] = str(exc)

        pd.DataFrame([recon_summary]).to_csv(
            os.path.join(
                out_dir,
                f"{split_name}_Genotype_Reconstruction_Metrics.csv",
            ),
            index=False,
        )

    # Main plots requested for the deterministic test-set output.
    util.plot_latent_heatmap(test_eval["mu"], labels=test_eval["labels"], out_dir=out_dir, filename="final_em_mu.pdf", prefix="mu")
    util.plot_tsne_latent(test_eval["mu"], labels=test_eval["labels"], sample_names=test_eval["sample_names"], out_dir=out_dir, filename="tsne_mu.pdf", random_state=seed)
    util.plot_umap_latent(test_eval["mu"], labels=test_eval["labels"], sample_names=test_eval["sample_names"], out_dir=out_dir, filename="umap_mu.pdf", random_state=seed)
    util.save_and_plot_latent_trait_correlations(test_eval["mu"], test_eval["y_raw"], data["trait_names"], out_dir=out_dir, prefix="mu")
    util.plot_trait_scatter(prediction_tables["test"], out_dir=out_dir, filename="Trait_Scatter.pdf")
    util.plot_trait_residuals(prediction_tables["test"], out_dir=out_dir, filename="Trait_Residuals.pdf")
    util.plot_trait_prediction_intervals(prediction_tables["test"], out_dir=out_dir, filename="Trait_Uncertainty_Intervals.pdf")
    util.plot_uncertainty_calibration(prediction_tables["test"], out_dir=out_dir, filename="Trait_Uncertainty_Calibration.pdf")

    # Genotype reconstruction diagnostics on the deterministic held-out test set.
    util.plot_genotype_reconstruction_diagnostics(
        test_eval["x_raw"],
        test_eval["x_hat"],
        obs_mask=test_eval["obs_mask"],
        genotype_likelihood=genotype_likelihood,
        out_dir=out_dir,
        filename="Genotype_Reconstruction_Diagnostics.pdf",
        random_state=seed,
    )
    if genotype_likelihood == "bernoulli":
        util.plot_bernoulli_genotype_calibration(
            test_eval["x_raw"],
            test_eval["x_hat"],
            obs_mask=test_eval["obs_mask"],
            out_dir=out_dir,
            filename="Bernoulli_Genotype_Calibration.pdf",
        )

    y_test_true = test_eval["y_raw"].numpy()
    y_test_pred = y_scaler.inverse_transform(test_eval["y_mu"].numpy())
    test_trait_df = trait_metrics_unscaled(y_test_true, y_test_pred, data["trait_names"])

    # Store deterministic mu-based outputs.
    torch.save(
        {
            "train_mu": train_eval["mu"],
            "train_y_mu": train_eval["y_mu"],
            "train_y_logvar": train_eval["y_logvar"],
            "val_mu": val_eval["mu"],
            "val_y_mu": val_eval["y_mu"],
            "val_y_logvar": val_eval["y_logvar"],
            "test_mu": test_eval["mu"],
            "test_y_mu": test_eval["y_mu"],
            "test_y_logvar": test_eval["y_logvar"],
            "test_x_raw": test_eval["x_raw"],
            "test_x_in": test_eval["x_in"],
            "test_x_hat": test_eval["x_hat"],
            "test_obs_mask": test_eval["obs_mask"],
            "test_sample_names": test_eval["sample_names"],
        },
        os.path.join(out_dir, "all_outputs_deterministic.pt"),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_metric": best_val,
            "config": {
                "lr": lr_in,
                "wd": wd_in,
                "batch_size": batch_size_in,
                "epochs": nepoch_in,
                "latent_dim": latent_dim,
                "enc_hidden_dim": enc_hidden_dim,
                "dec_hidden_dim": dec_hidden_dim,
                "dec_hidden_dim2": dec_hidden_dim2,
                "decoder_type": decoder_type,
                "genotype_likelihood": genotype_likelihood,
                "n_genotype_classes": n_genotype_classes,
                "dropout": dropout,
                "beta_kl": beta_kl,
                "alpha_trait": alpha_trait,
                "decoder_l1_lambda": decoder_l1_lambda,
                "dataset": dataset,
                "trait_name": trait_name,
                "seed": seed,
                "early_stop_metric": early_stop_metric,
                "regressor_type": regressor_type,
                "trait_likelihood": trait_likelihood,
            },
            "snp_names": data["snp_names"],
            "trait_names": data["trait_names"],
            "all_trait_names": data["all_trait_names"],
            "label_map": data["label_map"],
            "x_train_means": x_means,
            "y_train_fill_means": y_fill_means,
            "y_scaler_mean": y_scaler.mean_,
            "y_scaler_scale": y_scaler.scale_,
        },
        os.path.join(model_dir, "model.pt"),
    )

    def safe_nanmean(values):
        values = np.asarray(values, dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

    base_result = {
        "out_dir": out_dir,
        "model_dir": model_dir,
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_val),
        "train_total_loss": float(train_eval["loss"]),
        "train_recon_loss": float(train_eval["recon"]),
        "train_trait_loss": float(train_eval["trait"]),
        "val_total_loss": float(val_eval["loss"]),
        "val_recon_loss": float(val_eval["recon"]),
        "val_trait_loss": float(val_eval["trait"]),
        "test_total_loss": float(test_eval["loss"]),
        "test_recon_loss": float(test_eval["recon"]),
        "test_trait_loss": float(test_eval["trait"]),
        "test_mse_mean": safe_nanmean(test_trait_df["mse"]),
        "test_r2_mean": safe_nanmean(test_trait_df["r2"]),
        "test_pearson_mean": safe_nanmean(test_trait_df["pearson"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_forced_train_outliers": int(len(protected_outliers)),
        "labels_available": bool(data["has_labels"]),
        "regressor_type": regressor_type,
        "trait_likelihood": trait_likelihood,
        "decoder_type": decoder_type,
        "genotype_likelihood": genotype_likelihood,
        "dec_hidden_dim": int(dec_hidden_dim),
        "dec_hidden_dim2": int(dec_hidden_dim2),
        "n_genotype_classes": int(n_genotype_classes),
        "decoder_l1_lambda": float(decoder_l1_lambda),
        "train_genotype_hard_accuracy": float(train_eval["geno_acc"]),
        "val_genotype_hard_accuracy": float(val_eval["geno_acc"]),
        "test_genotype_hard_accuracy": float(test_eval["geno_acc"]),
    }
    pd.DataFrame([base_result]).to_csv(os.path.join(out_dir, "Run_Metrics.csv"), index=False)
    with open(os.path.join(out_dir, "Run_Metrics.json"), "w") as f:
        json.dump(base_result, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"Model saved to:   {model_dir}")
    return base_result
