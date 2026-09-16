#!/usr/bin/env python3
"""
Calibration-aware Bayesian optimization for the supervised SNP-VAE with selectable decoder/likelihood.

Main changes relative to bayes_opt_corrpenalty_calibration_aware.py
------------------------------------------------------------------
1. The train/validation/test split is generated once, saved, and reused by every trial.
2. The model-initialization seed is fixed during hyperparameter optimization.
3. A phenotype-stratified split is available and recommended for continuous traits such as FT10,
   so early, intermediate, and late accessions are represented in validation/test.
4. The script records stronger diagnostics: Lin's concordance correlation coefficient (CCC),
   prediction-versus-observation slope, tail errors/biases, and Gaussian uncertainty coverage.
5. A balanced objective can penalize global error, scale compression, and disproportionate
   tail failure while retaining a small reconstruction term for decoder-based interpretation.
6. Known good trials from an earlier CSV can be enqueued and re-evaluated fairly on the fixed split.
7. The Optuna study stores a configuration fingerprint to prevent accidental resume with a
   different split, objective, or search space.

The test set is passed to train_POEMS because the existing training function expects it, but this
script NEVER uses test metrics to choose hyperparameters. Final performance should still be
reported from repeated cross-validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from train import root_dir, split_train_val_test, train_POEMS

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


SEARCH_PARAM_NAMES = [
    "lr",
    "wd",
    "latent_dim",
    "alpha_trait",
    "beta_kl",
    "decoder_l1_lambda",
    "dropout",
    "early_stop_metric",
]


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3 or np.std(y_true) < eps or np.std(y_pred) < eps:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if len(a) < 3 or np.std(a) < eps or np.std(b) < eps:
        return np.nan
    return float(spearmanr(a, b).statistic)


def lin_ccc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Lin's concordance correlation coefficient. Perfect agreement is 1."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return np.nan

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=1))
    var_pred = float(np.var(y_pred, ddof=1))
    cov = float(np.cov(y_true, y_pred, ddof=1)[0, 1])
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < eps:
        return np.nan
    return float(2.0 * cov / denom)


def _regression_slope_intercept(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """OLS y = intercept + slope*x."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.var(x, ddof=1) < eps:
        return np.nan, np.nan
    slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def compute_metrics_and_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    objective_mode: str = "balanced",
    lambda_sd: float = 0.25,
    lambda_bias: float = 0.10,
    lambda_cor: float = 0.10,
    lambda_ccc: float = 0.20,
    lambda_slope: float = 0.10,
    lambda_tail: float = 0.10,
    tail_quantile: float = 0.10,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute validation diagnostics and a lower-is-better BO score.

    objective_mode="legacy"
        scaled RMSE + spread penalty + bias penalty - Pearson reward

    objective_mode="balanced" (recommended)
        scaled RMSE
        + CCC disagreement penalty
        + prediction-slope penalty
        + excess tail-error penalty
        + mean-bias penalty

    In the balanced objective, the prediction slope is obtained from
        y_pred = intercept + slope * y_true.
    A slope below one directly quantifies prediction compression in the usual
    observed-x / predicted-y scatter plot.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float).reshape(-1)
        if len(y_std) != len(y_true):
            raise ValueError("y_std must have the same length as y_true and y_pred.")

    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if y_std is not None:
        y_std = y_std[ok]

    if len(y_true) < 3:
        raise ValueError("Too few finite validation samples to compute the BO score.")

    residual = y_true - y_pred
    abs_error = np.abs(residual)

    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    median_ae = float(np.median(abs_error))
    p90_ae = float(np.quantile(abs_error, 0.90))
    max_ae = float(np.max(abs_error))

    mean_obs = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    mean_bias = float(mean_pred - mean_obs)

    sd_obs = float(np.std(y_true, ddof=1))
    sd_pred = float(np.std(y_pred, ddof=1))
    sd_obs_safe = max(sd_obs, eps)
    sd_pred_safe = max(sd_pred, eps)

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - mean_obs) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, eps))

    pearson = safe_pearson(y_true, y_pred, eps=eps)
    spearman = safe_spearman(y_true, y_pred, eps=eps)
    ccc = lin_ccc(y_true, y_pred, eps=eps)

    pearson_for_score = pearson if np.isfinite(pearson) else 0.0
    ccc_for_score = float(np.clip(ccc, -1.0, 1.0)) if np.isfinite(ccc) else 0.0

    sd_ratio = float(sd_pred_safe / sd_obs_safe)
    rmse_scaled = float(rmse / sd_obs_safe)
    mae_scaled = float(mae / sd_obs_safe)
    bias_scaled = float(mean_bias / sd_obs_safe)

    spread_penalty = float(abs(np.log(sd_ratio)))
    bias_penalty = float(abs(bias_scaled))

    pred_on_obs_slope, pred_on_obs_intercept = _regression_slope_intercept(
        y_true, y_pred, eps=eps
    )
    obs_on_pred_slope, obs_on_pred_intercept = _regression_slope_intercept(
        y_pred, y_true, eps=eps
    )
    slope_for_score = pred_on_obs_slope if np.isfinite(pred_on_obs_slope) else 0.0
    slope_penalty = float(abs(1.0 - slope_for_score))

    q_low = float(np.quantile(y_true, tail_quantile))
    q_high = float(np.quantile(y_true, 1.0 - tail_quantile))
    lower_mask = y_true <= q_low
    upper_mask = y_true >= q_high
    tail_mask = lower_mask | upper_mask

    def subset_metrics(mask: np.ndarray, prefix: str) -> Dict[str, float]:
        if int(np.sum(mask)) == 0:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_rmse": np.nan,
                f"{prefix}_mae": np.nan,
                f"{prefix}_bias": np.nan,
            }
        res = residual[mask]
        return {
            f"{prefix}_n": int(np.sum(mask)),
            f"{prefix}_rmse": float(np.sqrt(np.mean(res**2))),
            f"{prefix}_mae": float(np.mean(np.abs(res))),
            # prediction minus observation; negative means underprediction.
            f"{prefix}_bias": float(np.mean(y_pred[mask] - y_true[mask])),
        }

    lower_metrics = subset_metrics(lower_mask, "lower_tail")
    upper_metrics = subset_metrics(upper_mask, "upper_tail")
    tail_metrics = subset_metrics(tail_mask, "both_tails")

    tail_rmse = float(tail_metrics["both_tails_rmse"])
    tail_rmse_scaled = float(tail_rmse / sd_obs_safe) if np.isfinite(tail_rmse) else rmse_scaled
    tail_excess_penalty = float(max(0.0, tail_rmse_scaled - rmse_scaled))

    metrics: Dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "p90_ae": p90_ae,
        "max_ae": max_ae,
        "mse": mse,
        "r2": r2,
        "pearson": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman": float(spearman) if np.isfinite(spearman) else np.nan,
        "ccc": float(ccc) if np.isfinite(ccc) else np.nan,
        "mean_obs": mean_obs,
        "mean_pred": mean_pred,
        "mean_bias": mean_bias,
        "sd_obs": sd_obs,
        "sd_pred": sd_pred,
        "sd_ratio": sd_ratio,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "bias_scaled": bias_scaled,
        "spread_penalty": spread_penalty,
        "bias_penalty": bias_penalty,
        "pred_on_obs_slope": float(pred_on_obs_slope),
        "pred_on_obs_intercept": float(pred_on_obs_intercept),
        "obs_on_pred_slope": float(obs_on_pred_slope),
        "obs_on_pred_intercept": float(obs_on_pred_intercept),
        "slope_penalty": slope_penalty,
        "tail_quantile": float(tail_quantile),
        "tail_q_low": q_low,
        "tail_q_high": q_high,
        "tail_rmse_scaled": tail_rmse_scaled,
        "tail_excess_penalty": tail_excess_penalty,
        "n_val_finite": int(len(y_true)),
    }
    metrics.update(lower_metrics)
    metrics.update(upper_metrics)
    metrics.update(tail_metrics)

    # Gaussian uncertainty diagnostics. These are reported but not part of the
    # default objective because interval calibration can be noisy on one validation split.
    if y_std is not None:
        valid_std = np.isfinite(y_std) & (y_std > eps)
        if int(np.sum(valid_std)) >= 3:
            yt = y_true[valid_std]
            yp = y_pred[valid_std]
            ys = np.maximum(y_std[valid_std], eps)
            err = yt - yp
            abs_err = np.abs(err)
            z = err / ys

            gaussian_nll = float(
                np.mean(0.5 * (np.log(2.0 * np.pi * ys**2) + (err**2) / (ys**2)))
            )
            metrics.update(
                {
                    "uncertainty_n": int(len(yt)),
                    "mean_predicted_sd": float(np.mean(ys)),
                    "median_predicted_sd": float(np.median(ys)),
                    "gaussian_nll": gaussian_nll,
                    "standardized_residual_mean": float(np.mean(z)),
                    "standardized_residual_sd": float(np.std(z, ddof=1)),
                    "uncertainty_error_spearman": safe_spearman(ys, abs_err, eps=eps),
                }
            )

            for label, nominal, zcrit in [
                ("50", 0.50, 0.67448975),
                ("80", 0.80, 1.28155157),
                ("95", 0.95, 1.95996398),
            ]:
                covered = np.abs(err) <= zcrit * ys
                metrics[f"coverage_{label}"] = float(np.mean(covered))
                metrics[f"coverage_error_{label}"] = float(abs(np.mean(covered) - nominal))
                metrics[f"mean_interval_width_{label}"] = float(np.mean(2.0 * zcrit * ys))

    if objective_mode == "legacy":
        base_score = (
            rmse_scaled
            + lambda_sd * spread_penalty
            + lambda_bias * bias_penalty
            - lambda_cor * pearson_for_score
        )
        metrics["objective_rmse_component"] = rmse_scaled
        metrics["objective_spread_component"] = lambda_sd * spread_penalty
        metrics["objective_bias_component"] = lambda_bias * bias_penalty
        metrics["objective_correlation_component"] = -lambda_cor * pearson_for_score
        metrics["objective_ccc_component"] = 0.0
        metrics["objective_slope_component"] = 0.0
        metrics["objective_tail_component"] = 0.0
    elif objective_mode == "balanced":
        base_score = (
            rmse_scaled
            + lambda_ccc * (1.0 - ccc_for_score)
            + lambda_slope * slope_penalty
            + lambda_tail * tail_excess_penalty
            + lambda_bias * bias_penalty
        )
        metrics["objective_rmse_component"] = rmse_scaled
        metrics["objective_spread_component"] = 0.0
        metrics["objective_bias_component"] = lambda_bias * bias_penalty
        metrics["objective_correlation_component"] = 0.0
        metrics["objective_ccc_component"] = lambda_ccc * (1.0 - ccc_for_score)
        metrics["objective_slope_component"] = lambda_slope * slope_penalty
        metrics["objective_tail_component"] = lambda_tail * tail_excess_penalty
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    metrics["base_bo_score"] = float(base_score)
    return metrics


def read_validation_predictions(
    out_dir: str,
    trait_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, pd.DataFrame]:
    """Read deterministic validation prediction means and optional Gaussian SDs."""
    pred_path = os.path.join(out_dir, "val_predictions_with_uncertainty.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Could not find validation prediction file:\n{pred_path}\n"
            "This script expects train_POEMS to save val_predictions_with_uncertainty.csv."
        )

    df = pd.read_csv(pred_path)
    required = ["sample_name", "trait", "y_true", "y_mu"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {pred_path}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if trait_name is not None:
        sub = df[df["trait"].astype(str) == str(trait_name)].copy()
        if sub.empty:
            available = sorted(df["trait"].astype(str).unique())
            raise ValueError(
                f"No rows for trait_name={trait_name} in {pred_path}. "
                f"Available traits: {available}"
            )
    else:
        sub = df.copy()

    agg: Dict[str, str] = {"y_true": "mean", "y_mu": "mean"}
    if "y_std" in sub.columns:
        agg["y_std"] = "mean"

    sub = sub.groupby(["sample_name", "trait"], as_index=False).agg(agg)
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_pred = sub["y_mu"].to_numpy(dtype=float)
    y_std = sub["y_std"].to_numpy(dtype=float) if "y_std" in sub.columns else None
    return y_true, y_pred, y_std, pred_path, sub


# -----------------------------------------------------------------------------
# Fixed split generation and validation
# -----------------------------------------------------------------------------


def load_split_inputs(
    data_dir: Path,
    trait_name: Optional[str],
) -> Dict[str, Any]:
    """Load only phenotype/labels/sample names; avoids loading the large SNP matrix."""
    trait_path = data_dir / "2_all.csv"
    trait_name_path = data_dir / "2_featname.csv"
    label_path = data_dir / "labels_all.csv"
    sample_path = data_dir / "samples.txt"

    if not trait_path.exists() or not trait_name_path.exists():
        raise FileNotFoundError(
            f"Expected {trait_path} and {trait_name_path} for fixed split generation."
        )

    trait_df = pd.read_csv(trait_path, header=None, na_values=["nan", "NA", "", "NaN"])
    trait_names = pd.read_csv(trait_name_path, header=None).iloc[:, 0].astype(str).tolist()

    if trait_df.shape[1] != len(trait_names):
        raise ValueError("Mismatch between 2_all.csv columns and 2_featname.csv rows.")

    if trait_name is None:
        if trait_df.shape[1] != 1:
            raise ValueError(
                "Fixed split generation requires --trait_name when 2_all.csv has multiple traits."
            )
        selected_name = trait_names[0]
        y = trait_df.iloc[:, 0].to_numpy(dtype=float)
    else:
        if trait_name not in trait_names:
            raise ValueError(
                f"Trait '{trait_name}' not found. Available traits: {trait_names}"
            )
        selected_name = trait_name
        y = trait_df.iloc[:, trait_names.index(trait_name)].to_numpy(dtype=float)

    n = len(y)
    if label_path.exists():
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].astype(int).to_numpy()
        if len(labels) != n:
            raise ValueError("Mismatch between labels_all.csv and phenotype rows.")
        has_labels = True
    else:
        labels = np.zeros(n, dtype=int)
        has_labels = False

    if sample_path.exists():
        sample_names = pd.read_csv(sample_path, header=None).iloc[:, 0].astype(str).to_numpy()
        if len(sample_names) != n:
            raise ValueError("Mismatch between samples.txt and phenotype rows.")
    else:
        sample_names = np.asarray([f"sample_{i}" for i in range(n)], dtype=str)

    return {
        "y": y,
        "labels": labels,
        "has_labels": has_labels,
        "sample_names": sample_names,
        "trait_name": selected_name,
        "n": n,
    }


def _phenotype_strata(y: np.ndarray, max_bins: int) -> Tuple[np.ndarray, int]:
    """Quantile bins for continuous-trait stratification."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 10:
        raise ValueError("Too few finite trait values for phenotype-stratified splitting.")

    max_bins = min(int(max_bins), len(np.unique(y)), max(2, len(y) // 4))
    for bins in range(max_bins, 1, -1):
        try:
            strata = pd.qcut(y, q=bins, labels=False, duplicates="drop")
            strata = np.asarray(strata, dtype=int)
            counts = np.bincount(strata)
            if len(counts) >= 2 and counts.min() >= 4:
                return strata, len(counts)
        except (ValueError, TypeError):
            continue
    raise ValueError(
        "Could not construct stable phenotype quantile bins. "
        "Use --split_mode random or reduce --phenotype_bins."
    )


def _two_stage_split(
    indices: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    strata: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if strata is not None and len(strata) != len(indices):
        raise ValueError("strata and indices must have identical length.")

    trainval_idx, test_idx, trainval_strata, _ = train_test_split(
        indices,
        strata if strata is not None else np.zeros(len(indices), dtype=int),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )

    val_fraction_of_trainval = val_size / (1.0 - test_size)
    strat2 = trainval_strata if strata is not None else None
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        shuffle=True,
        stratify=strat2,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def generate_fixed_split(
    split_inputs: Dict[str, Any],
    split_mode: str,
    split_seed: int,
    test_size: float,
    val_size: float,
    phenotype_bins: int,
    force_trait_outliers_train: bool,
    outlier_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    y = np.asarray(split_inputs["y"], dtype=float)
    labels = np.asarray(split_inputs["labels"], dtype=int)
    n = int(split_inputs["n"])
    idx = np.arange(n)
    finite_idx = idx[np.isfinite(y)]
    missing_idx = idx[~np.isfinite(y)]

    metadata: Dict[str, Any] = {
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "phenotype_bins_requested": int(phenotype_bins),
        "n_missing_trait_forced_train": int(len(missing_idx)),
    }

    if split_mode == "phenotype_stratified":
        strata, used_bins = _phenotype_strata(y[finite_idx], phenotype_bins)
        train_idx, val_idx, test_idx = _two_stage_split(
            finite_idx,
            test_size=test_size,
            val_size=val_size,
            seed=split_seed,
            strata=strata,
        )
        train_idx = np.sort(np.concatenate([train_idx, missing_idx]))
        metadata["phenotype_bins_used"] = int(used_bins)
        metadata["forced_trait_outliers_train"] = False

    elif split_mode == "label_stratified":
        if not split_inputs["has_labels"]:
            raise ValueError("label_stratified requested, but labels_all.csv is missing.")
        finite_labels = labels[finite_idx]
        counts = np.bincount(finite_labels)
        if len(counts) < 2 or counts.min() < 4:
            raise ValueError("Labels are not suitable for two-stage stratified splitting.")
        train_idx, val_idx, test_idx = _two_stage_split(
            finite_idx,
            test_size=test_size,
            val_size=val_size,
            seed=split_seed,
            strata=finite_labels,
        )
        train_idx = np.sort(np.concatenate([train_idx, missing_idx]))
        metadata["forced_trait_outliers_train"] = False

    elif split_mode == "random":
        train_idx, val_idx, test_idx = _two_stage_split(
            finite_idx,
            test_size=test_size,
            val_size=val_size,
            seed=split_seed,
            strata=None,
        )
        train_idx = np.sort(np.concatenate([train_idx, missing_idx]))
        metadata["forced_trait_outliers_train"] = False

    elif split_mode == "existing":
        train_idx, val_idx, test_idx, protected = split_train_val_test(
            n=n,
            y_raw=y[:, None],
            labels=labels,
            has_labels=bool(split_inputs["has_labels"]),
            seed=split_seed,
            test_size=test_size,
            val_size=val_size,
            split_strategy="auto",
            force_trait_outliers_train=force_trait_outliers_train,
            outlier_quantile=outlier_quantile,
        )
        metadata["forced_trait_outliers_train"] = bool(force_trait_outliers_train)
        metadata["n_forced_trait_outliers_train"] = int(len(protected))
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    validate_split(train_idx, val_idx, test_idx, n)
    return train_idx, val_idx, test_idx, metadata


def validate_split(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n: int,
) -> None:
    train_set = set(map(int, train_idx))
    val_set = set(map(int, val_idx))
    test_set = set(map(int, test_idx))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Fixed split indices overlap.")
    union = train_set | val_set | test_set
    if union != set(range(n)):
        missing = sorted(set(range(n)) - union)[:10]
        extra = sorted(union - set(range(n)))[:10]
        raise ValueError(f"Fixed split does not cover exactly 0..{n-1}. Missing={missing}, extra={extra}")
    if min(len(train_idx), len(val_idx), len(test_idx)) < 3:
        raise ValueError("Train, validation, and test splits must each contain at least 3 samples.")


def _hash_array_text(values: Sequence[Any]) -> str:
    joined = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _hash_float_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_summary_dataframe(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, ind in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        values = np.asarray(y[ind], dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "split": name,
                "n_total": int(len(ind)),
                "n_finite_trait": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "min": float(np.min(values)) if len(values) else np.nan,
                "q10": float(np.quantile(values, 0.10)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q90": float(np.quantile(values, 0.90)) if len(values) else np.nan,
                "max": float(np.max(values)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def get_or_create_fixed_split(
    args: argparse.Namespace,
    split_inputs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path, Dict[str, Any]]:
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file is None:
        safe_trait = str(split_inputs["trait_name"]).replace(os.sep, "_").replace(" ", "_")
        split_file = split_dir / (
            f"fixed_split__{args.dataset}__{safe_trait}__{args.split_mode}__seed{args.split_seed}.npz"
        )
    else:
        split_file = Path(args.split_file)
        split_file.parent.mkdir(parents=True, exist_ok=True)

    meta_file = split_file.with_suffix(".json")
    summary_file = split_file.with_name(split_file.stem + "__summary.csv")
    assignment_file = split_file.with_name(split_file.stem + "__assignments.csv")

    current_hashes = {
        "sample_names_sha256": _hash_array_text(split_inputs["sample_names"]),
        "trait_sha256": _hash_float_array(split_inputs["y"]),
        "n_samples": int(split_inputs["n"]),
        "dataset": args.dataset,
        "trait_name": split_inputs["trait_name"],
    }

    if split_file.exists():
        saved = np.load(split_file)
        train_idx = np.asarray(saved["train_idx"], dtype=int)
        val_idx = np.asarray(saved["val_idx"], dtype=int)
        test_idx = np.asarray(saved["test_idx"], dtype=int)
        validate_split(train_idx, val_idx, test_idx, int(split_inputs["n"]))

        if not meta_file.exists():
            raise FileNotFoundError(
                f"Split file exists but metadata is missing: {meta_file}. "
                "Delete the split file and regenerate it."
            )
        metadata = json.loads(meta_file.read_text())
        for key, value in current_hashes.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Existing split metadata mismatch for '{key}'. "
                    "The sample order or phenotype data may have changed."
                )
        print(f"Loaded fixed split: {split_file}")
    else:
        train_idx, val_idx, test_idx, metadata = generate_fixed_split(
            split_inputs=split_inputs,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            test_size=args.test_size,
            val_size=args.val_size,
            phenotype_bins=args.phenotype_bins,
            force_trait_outliers_train=args.force_trait_outliers_train,
            outlier_quantile=args.outlier_quantile,
        )
        metadata.update(current_hashes)
        np.savez_compressed(
            split_file,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        meta_file.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"Created fixed split: {split_file}")

    summary = split_summary_dataframe(
        split_inputs["y"], train_idx, val_idx, test_idx
    )
    summary.to_csv(summary_file, index=False)

    assignment = pd.DataFrame(
        {
            "sample_index": np.arange(split_inputs["n"]),
            "sample_name": split_inputs["sample_names"],
            "trait_value": split_inputs["y"],
        }
    )
    split_labels = np.empty(split_inputs["n"], dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "validation"
    split_labels[test_idx] = "test"
    assignment["split"] = split_labels
    assignment.to_csv(assignment_file, index=False)

    print("\nFixed split trait summary")
    print(summary.to_string(index=False))
    print(f"Split summary saved to: {summary_file}")
    print(f"Split assignments saved to: {assignment_file}")

    return train_idx, val_idx, test_idx, split_file, metadata


# -----------------------------------------------------------------------------
# Optuna objective, persistence, and warm start
# -----------------------------------------------------------------------------


def _trial_params(
    trial: optuna.Trial,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    params = {
        "lr": trial.suggest_float(
            "lr",
            args.lr_min,
            args.lr_max,
            log=True,
        ),
        "wd": trial.suggest_float(
            "wd",
            args.wd_min,
            args.wd_max,
            log=True,
        ),
        "latent_dim": trial.suggest_categorical(
            "latent_dim",
            args.latent_dims,
        ),
        "alpha_trait": trial.suggest_float(
            "alpha_trait",
            args.alpha_min,
            args.alpha_max,
            log=True,
        ),
        "beta_kl": trial.suggest_float(
            "beta_kl",
            args.beta_min,
            args.beta_max,
            log=True,
        ),
        "dropout": trial.suggest_float(
            "dropout",
            args.dropout_min,
            args.dropout_max,
        ),
        "early_stop_metric": trial.suggest_categorical(
            "early_stop_metric",
            args.early_stop_metrics,
        ),
    }

    # decoder_l1_lambda is a POEMS gate-W hyperparameter. It is not
    # semantically transferable to the dense decoder.
    if args.decoder_type == "poems":
        params["decoder_l1_lambda"] = trial.suggest_float(
            "decoder_l1_lambda",
            args.l1_min,
            args.l1_max,
            log=True,
        )
    else:
        params["decoder_l1_lambda"] = 0.0

    return params


def _cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_objective(
    args: argparse.Namespace,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_file: Path,
):
    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial, args)
        try:
            result = train_POEMS(
                lr_in=params["lr"],
                wd_in=params["wd"],
                batch_size_in=args.batch_size,
                nepoch_in=args.epochs,
                experiment_note=f"{args.experiment_note}_trial_{trial.number}",
                dataset=args.dataset,
                trait_name=args.trait_name,
                latent_dim=params["latent_dim"],
                enc_hidden_dim=args.enc_hidden_dim,
                dec_hidden_dim=args.dec_hidden_dim,
                dec_hidden_dim2=args.dec_hidden_dim2,
                decoder_type=args.decoder_type,
                genotype_likelihood=args.genotype_likelihood,
                n_genotype_classes=args.n_genotype_classes,
                dropout=params["dropout"],
                beta_kl=params["beta_kl"],
                alpha_trait=params["alpha_trait"],
                decoder_l1_lambda=params["decoder_l1_lambda"],
                # FIXED across trials: hyperparameters, not seed luck, are being compared.
                seed=args.model_seed,
                early_stop_metric=params["early_stop_metric"],
                patience=args.patience,
                regressor_type=args.regressor_type,
                trait_likelihood=args.trait_likelihood,
                train_idx_override=train_idx,
                val_idx_override=val_idx,
                test_idx_override=test_idx,
                # Overrides already define the split, so outlier movement is disabled here.
                force_trait_outliers_train=False,
                outlier_quantile=args.outlier_quantile,
                infer_structure=False,
                skip_interpretation=True,
            )

            out_dir = result["out_dir"]
            y_true, y_pred, y_std, pred_path, _ = read_validation_predictions(
                out_dir=out_dir,
                trait_name=args.trait_name,
            )

            metrics = compute_metrics_and_score(
                y_true=y_true,
                y_pred=y_pred,
                y_std=y_std,
                objective_mode=args.objective_mode,
                lambda_sd=args.lambda_sd,
                lambda_bias=args.lambda_bias,
                lambda_cor=args.lambda_cor,
                lambda_ccc=args.lambda_ccc,
                lambda_slope=args.lambda_slope,
                lambda_tail=args.lambda_tail,
                tail_quantile=args.tail_quantile,
            )

            recon_loss = float(result["val_recon_loss"])
            recon_component = args.recon_weight * recon_loss
            score = float(metrics["base_bo_score"] + recon_component)
            metrics["recon_loss"] = recon_loss
            metrics["objective_reconstruction_component"] = float(recon_component)
            metrics["bo_score"] = score

            trial.set_user_attr("objective_mode", args.objective_mode)
            trial.set_user_attr("objective_score", score)
            trial.set_user_attr("fixed_split_file", str(split_file))
            trial.set_user_attr("fixed_model_seed", int(args.model_seed))
            trial.set_user_attr("decoder_type", args.decoder_type)
            trial.set_user_attr("genotype_likelihood", args.genotype_likelihood)
            trial.set_user_attr("dec_hidden_dim", int(args.dec_hidden_dim))
            trial.set_user_attr("dec_hidden_dim2", int(args.dec_hidden_dim2))
            trial.set_user_attr("n_genotype_classes", int(args.n_genotype_classes))
            trial.set_user_attr("val_prediction_file", pred_path)
            trial.set_user_attr("out_dir", out_dir)
            trial.set_user_attr("model_dir", result.get("model_dir", ""))

            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    trial.set_user_attr(f"val_{key}", value)

            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    trial.set_user_attr(key, value)

            metric_out = os.path.join(out_dir, "calibration_aware_val_metrics.json")
            with open(metric_out, "w") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)

            return score
        finally:
            _cleanup_memory()

    return objective


def study_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            "state": str(trial.state),
        }
        row.update(trial.params)
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def save_trials(study: optuna.Study, out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    study_to_dataframe(study).to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"Saved Optuna trials to: {out_path}")


def save_callback(out_csv: str):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        save_trials(study, out_csv)
    return callback


def _within(value: float, lower: float, upper: float) -> bool:
    return np.isfinite(value) and lower <= value <= upper


def validate_warm_start_params(
    params: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    try:
        checks = [
            _within(float(params["lr"]), args.lr_min, args.lr_max),
            _within(float(params["wd"]), args.wd_min, args.wd_max),
            int(params["latent_dim"]) in set(args.latent_dims),
            _within(
                float(params["alpha_trait"]),
                args.alpha_min,
                args.alpha_max,
            ),
            _within(
                float(params["beta_kl"]),
                args.beta_min,
                args.beta_max,
            ),
            _within(
                float(params["dropout"]),
                args.dropout_min,
                args.dropout_max,
            ),
            str(params["early_stop_metric"])
            in set(args.early_stop_metrics),
        ]

        if args.decoder_type == "poems":
            checks.append(
                _within(
                    float(params["decoder_l1_lambda"]),
                    args.l1_min,
                    args.l1_max,
                )
            )
        else:
            # Dense decoder has no POEMS gate-W penalty.
            checks.append(
                float(params.get("decoder_l1_lambda", 0.0)) == 0.0
            )

    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid or missing parameter: {exc}"

    if not all(checks):
        return False, "one or more values fall outside the current search space"
    return True, ""


def enqueue_warm_start_trials(
    study: optuna.Study,
    csv_path: Optional[str],
    top_n: int,
    args: argparse.Namespace,
) -> None:
    if csv_path is None or top_n <= 0:
        return

    df = pd.read_csv(csv_path)
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    if "value" not in df.columns:
        raise ValueError(f"Warm-start CSV lacks a 'value' column: {csv_path}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[np.isfinite(df["value"])].sort_values("value").head(top_n)

    enqueued = 0
    for _, row in df.iterrows():
        params = {
            "lr": float(row["lr"]),
            "wd": float(row["wd"]),
            "latent_dim": int(row["latent_dim"]),
            "alpha_trait": float(row["alpha_trait"]),
            "beta_kl": float(row["beta_kl"]),
            "dropout": float(row["dropout"]),
            "early_stop_metric": str(row["early_stop_metric"]),
        }
        if args.decoder_type == "poems":
            params["decoder_l1_lambda"] = float(
                row["decoder_l1_lambda"]
            )
        ok, reason = validate_warm_start_params(params, args)
        if not ok:
            print(f"Skipping warm-start row: {reason}. Params={params}")
            continue
        study.enqueue_trial(params, skip_if_exists=True)
        enqueued += 1

    print(f"Enqueued {enqueued} previous configuration(s) for fair fixed-split re-evaluation.")


def build_study_config(
    args: argparse.Namespace,
    split_file: Path,
    split_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "trait_name": args.trait_name,
        "split_file": str(split_file.resolve()),
        "sample_names_sha256": split_metadata.get("sample_names_sha256"),
        "trait_sha256": split_metadata.get("trait_sha256"),
        "model_seed": args.model_seed,
        "sampler_seed": args.sampler_seed,
        "objective_mode": args.objective_mode,
        "weights": {
            "lambda_sd": args.lambda_sd,
            "lambda_bias": args.lambda_bias,
            "lambda_cor": args.lambda_cor,
            "lambda_ccc": args.lambda_ccc,
            "lambda_slope": args.lambda_slope,
            "lambda_tail": args.lambda_tail,
            "recon_weight": args.recon_weight,
        },
        "tail_quantile": args.tail_quantile,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "regressor_type": args.regressor_type,
        "trait_likelihood": args.trait_likelihood,
        "decoder_type": args.decoder_type,
        "genotype_likelihood": args.genotype_likelihood,
        "enc_hidden_dim": args.enc_hidden_dim,
        "dec_hidden_dim": args.dec_hidden_dim,
        "dec_hidden_dim2": args.dec_hidden_dim2,
        "n_genotype_classes": args.n_genotype_classes,
        "search_space": {
            "lr": [args.lr_min, args.lr_max],
            "wd": [args.wd_min, args.wd_max],
            "latent_dims": list(args.latent_dims),
            "alpha_trait": [args.alpha_min, args.alpha_max],
            "beta_kl": [args.beta_min, args.beta_max],
            "decoder_l1_lambda": (
                [args.l1_min, args.l1_max]
                if args.decoder_type == "poems"
                else [0.0, 0.0]
            ),
            "dropout": [args.dropout_min, args.dropout_max],
            "early_stop_metrics": list(args.early_stop_metrics),
        },
    }


def config_fingerprint(config: Dict[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize_or_validate_study(
    study: optuna.Study,
    config: Dict[str, Any],
    allow_mismatch: bool,
) -> None:
    fingerprint = config_fingerprint(config)
    old_fingerprint = study.user_attrs.get("config_fingerprint")

    if old_fingerprint is None:
        if len(study.trials) > 0 and not allow_mismatch:
            raise RuntimeError(
                "The existing Optuna study contains trials but has no configuration fingerprint. "
                "Use a new --study_name/--storage, or pass --allow_study_mismatch only if you "
                "deliberately want to mix incompatible trials."
            )
        study.set_user_attr("config_fingerprint", fingerprint)
        study.set_user_attr("study_config", config)
    elif old_fingerprint != fingerprint and not allow_mismatch:
        raise RuntimeError(
            "Existing Optuna study configuration does not match this run. "
            "Use a new study/database or explicitly pass --allow_study_mismatch."
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-split, fixed-seed, calibration-aware Bayesian optimization "
            "for the supervised SNP-VAE."
        )
    )

    parser.add_argument("--dataset", default="plant_new")
    parser.add_argument("--trait_name", default="DTF")
    parser.add_argument(
        "--experiment_note",
        default="plant_supervised_vae_calibration_aware_fixed_bayesopt",
    )

    # Separate seeds have separate scientific roles.
    parser.add_argument("--split_seed", type=int, default=21)
    parser.add_argument("--model_seed", type=int, default=21)
    parser.add_argument("--sampler_seed", type=int, default=21)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--enc_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--dec_hidden_dim",
        type=int,
        default=None,
        help="If omitted: 64 for POEMS, 256 for dense_mlp.",
    )
    parser.add_argument("--dec_hidden_dim2", type=int, default=512)
    parser.add_argument(
        "--decoder_type",
        choices=["poems", "dense_mlp"],
        default="poems",
    )
    parser.add_argument(
        "--genotype_likelihood",
        choices=["mse", "bernoulli", "categorical"],
        default="mse",
    )
    parser.add_argument("--n_genotype_classes", type=int, default=3)

    parser.add_argument("--study_name", default="poems_vae_calibration_aware_fixed_bayesopt")
    parser.add_argument("--storage", default="sqlite:///optuna_poems_calibration_aware_fixed.db")
    parser.add_argument("--out_csv", default="optuna_trials_calibration_aware_fixed.csv")
    parser.add_argument("--allow_study_mismatch", action="store_true")

    parser.add_argument("--regressor_type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--trait_likelihood", choices=["gaussian", "mse"], default="gaussian")

    # Fixed split configuration. phenotype_stratified is recommended for FT10.
    parser.add_argument(
        "--split_mode",
        choices=["phenotype_stratified", "label_stratified", "random", "existing"],
        default="phenotype_stratified",
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--phenotype_bins", type=int, default=10)
    parser.add_argument("--split_dir", default="bo_fixed_splits")
    parser.add_argument("--split_file", default=None)
    parser.add_argument(
        "--force_trait_outliers_train",
        action="store_true",
        help="Only applies to --split_mode existing. Not recommended for FT10 BO diagnostics.",
    )
    parser.add_argument("--outlier_quantile", type=float, default=0.05)

    # Search space.
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--wd_min", type=float, default=1e-7)
    # Expanded because the previous best was near 1e-3, the old upper boundary.
    parser.add_argument("--wd_max", type=float, default=1e-2)
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=30.0)
    parser.add_argument("--beta_min", type=float, default=1e-6)
    parser.add_argument("--beta_max", type=float, default=1e-3)
    parser.add_argument("--l1_min", type=float, default=1e-7)
    parser.add_argument("--l1_max", type=float, default=1e-2)
    parser.add_argument("--dropout_min", type=float, default=0.0)
    parser.add_argument("--dropout_max", type=float, default=0.25)
    parser.add_argument(
        "--early_stop_metrics",
        nargs="+",
        default=["trait"],
        choices=["trait", "total", "recon"],
        help="Use 'trait' alone for a focused refinement run; add 'total' to search both.",
    )

    # Objective configuration.
    parser.add_argument(
        "--objective_mode",
        choices=["balanced", "legacy"],
        default="balanced",
    )
    parser.add_argument("--lambda_sd", type=float, default=0.25)
    parser.add_argument("--lambda_bias", type=float, default=0.10)
    parser.add_argument("--lambda_cor", type=float, default=0.10)
    parser.add_argument("--lambda_ccc", type=float, default=0.20)
    parser.add_argument("--lambda_slope", type=float, default=0.10)
    parser.add_argument("--lambda_tail", type=float, default=0.10)
    parser.add_argument("--tail_quantile", type=float, default=0.10)
    parser.add_argument(
        "--recon_weight",
        type=float,
        default=0.05,
        help=(
            "Small tie-breaker preserving reconstruction for decoder interpretation. "
            "Set to 0 for pure point-prediction tuning."
        ),
    )

    # Efficiently re-test old top trials under the fair fixed design.
    parser.add_argument("--warm_start_csv", default=None)
    parser.add_argument("--warm_start_top_n", type=int, default=0)

    # TPE sampler controls.
    parser.add_argument("--n_startup_trials", type=int, default=12)

    args = parser.parse_args()

    if not (0.0 < args.test_size < 1.0 and 0.0 < args.val_size < 1.0):
        parser.error("--test_size and --val_size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1.0:
        parser.error("--test_size + --val_size must be less than 1.")
    if not (0.0 < args.tail_quantile < 0.5):
        parser.error("--tail_quantile must be between 0 and 0.5.")
    if args.n_trials < 1:
        parser.error("--n_trials must be at least 1.")

    allowed = {
        ("poems", "mse"),
        ("dense_mlp", "mse"),
        ("dense_mlp", "bernoulli"),
        ("dense_mlp", "categorical"),
    }
    if (args.decoder_type, args.genotype_likelihood) not in allowed:
        parser.error(
            "Supported combinations are poems+mse, dense_mlp+mse, "
            "dense_mlp+bernoulli, and dense_mlp+categorical."
        )
    if args.n_genotype_classes < 2:
        parser.error("--n_genotype_classes must be >= 2.")

    if args.dec_hidden_dim is None:
        args.dec_hidden_dim = (
            64 if args.decoder_type == "poems" else 256
        )

    if args.decoder_type == "dense_mlp":
        # The l1_min/l1_max CLI values are ignored for dense_mlp.
        pass

    if args.recon_weight != 0:
        print(
            "NOTE: reconstruction losses are likelihood-specific. "
            "Do not compare Optuna objective values across MSE, Bernoulli, "
            "and categorical studies. Compare final models with repeated CV."
        )

    return args


def main() -> None:
    args = parse_args()

    data_dir = Path(root_dir) / "data" / args.dataset
    split_inputs = load_split_inputs(data_dir, trait_name=args.trait_name)
    train_idx, val_idx, test_idx, split_file, split_metadata = get_or_create_fixed_split(
        args, split_inputs
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        load_if_exists=True,
    )

    study_config = build_study_config(args, split_file, split_metadata)
    initialize_or_validate_study(
        study,
        study_config,
        allow_mismatch=args.allow_study_mismatch,
    )

    enqueue_warm_start_trials(
        study=study,
        csv_path=args.warm_start_csv,
        top_n=args.warm_start_top_n,
        args=args,
    )

    objective = make_objective(
        args=args,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_file=split_file,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[save_callback(args.out_csv)],
        gc_after_trial=True,
        n_jobs=1,
    )

    save_trials(study, args.out_csv)
    print("\nBest value:", study.best_value)
    print("Best params:", study.best_params)
    print("Fixed split:", split_file)
    print("Fixed model seed:", args.model_seed)
    print(
        "\nImportant: select the configuration here, then estimate final predictive "
        "performance with repeated cross-validation and an ensemble."
    )


if __name__ == "__main__":
    main()
