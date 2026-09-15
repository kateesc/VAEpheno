#!/usr/bin/env python3
"""
Evaluation utilities for the supervised SNP-VAE project.

This module keeps the original clustering-evaluation API while adding
decoder-agnostic genotype reconstruction metrics for:

    * MSE reconstruction
    * Bernoulli reconstruction for binary 0/1 SNPs
    * Categorical reconstruction for discrete genotype states (e.g. 0/1/2)

The genotype functions are intentionally independent of PyTorch so they can
also be used on saved NumPy/CSV outputs.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp
from sklearn import metrics


# ---------------------------------------------------------------------
# Original clustering evaluation, made robust to non-contiguous labels.
# ---------------------------------------------------------------------

def _as_1d(a, name: str) -> np.ndarray:
    x = np.asarray(a).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} is empty.")
    return x


def evaluate(label, pred):
    """
    Return NMI, ARI, Fowlkes-Mallows, and Hungarian-matched accuracy.

    Backward-compatible replacement for the original evaluate().
    """
    y_true = _as_1d(label, "label")
    y_pred = _as_1d(pred, "pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("label and pred must have the same number of elements.")

    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    f = metrics.fowlkes_mallows_score(y_true, y_pred)

    pred_adjusted = get_y_preds(
        y_true,
        y_pred,
        n_clusters=max(len(np.unique(y_true)), len(np.unique(y_pred))),
    )
    acc = metrics.accuracy_score(y_true, pred_adjusted)
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters=None):
    """
    Backward-compatible cost matrix helper.

    The returned matrix is suitable for minimizing assignment cost.
    """
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError("C must be a 2D contingency/confusion matrix.")

    if n_clusters is None:
        n_clusters = max(C.shape)

    n_clusters = int(n_clusters)
    padded = np.zeros((n_clusters, n_clusters), dtype=float)
    padded[: C.shape[0], : C.shape[1]] = C

    # Maximizing agreement == minimizing max(C)-C.
    return padded.max() - padded


def get_cluster_labels_from_indices(indices):
    """
    Backward-compatible helper returning the assigned column for each row.
    """
    indices = list(indices)
    if not indices:
        return np.array([], dtype=int)

    n_rows = max(int(i) for i, _ in indices) + 1
    out = np.full(n_rows, -1, dtype=int)
    for row, col in indices:
        out[int(row)] = int(col)
    return out


def get_y_preds(y_true, cluster_assignments, n_clusters=None):
    """
    Map arbitrary cluster IDs to true labels with the Hungarian algorithm.

    Unlike the old implementation, this works when true labels or predicted
    cluster IDs are non-contiguous or do not start at zero.
    """
    y_true = _as_1d(y_true, "y_true")
    y_pred = _as_1d(cluster_assignments, "cluster_assignments")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and cluster_assignments must have equal length.")

    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)

    contingency = metrics.cluster.contingency_matrix(
        y_true,
        y_pred,
        sparse=False,
    )

    # Maximize the number of matched samples.
    row_ind, col_ind = linear_sum_assignment(-contingency)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[pred_values[c]] = true_values[r]

    # If there are more predicted clusters than true classes, map unmatched
    # clusters to the true class with maximal overlap.
    for c, pred_value in enumerate(pred_values):
        if pred_value in mapping:
            continue
        best_r = int(np.argmax(contingency[:, c]))
        mapping[pred_value] = true_values[best_r]

    return np.asarray([mapping[v] for v in y_pred])


# ---------------------------------------------------------------------
# Genotype reconstruction evaluation.
# ---------------------------------------------------------------------

def _prepare_mask(x_true: np.ndarray, obs_mask: Optional[np.ndarray]) -> np.ndarray:
    x_true = np.asarray(x_true)

    if obs_mask is None:
        mask = np.isfinite(x_true)
    else:
        mask = np.asarray(obs_mask).astype(bool)
        if mask.shape != x_true.shape:
            raise ValueError(
                f"obs_mask shape {mask.shape} does not match x_true shape {x_true.shape}."
            )
        mask &= np.isfinite(x_true)

    if not np.any(mask):
        raise ValueError("No observed finite genotype entries are available.")

    return mask


def _validate_bernoulli_targets(x_true: np.ndarray, mask: np.ndarray) -> None:
    vals = np.unique(x_true[mask])
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(
            "Bernoulli reconstruction requires observed genotype values in {0,1}; "
            f"found values such as {vals[:10]}."
        )


def _validate_categorical_targets(
    x_true: np.ndarray,
    mask: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    vals = x_true[mask]
    if not np.all(np.isclose(vals, np.round(vals))):
        raise ValueError("Categorical genotype targets must be integer-coded classes.")

    vals_int = np.round(vals).astype(int)
    if vals_int.min() < 0 or vals_int.max() >= n_classes:
        raise ValueError(
            f"Categorical targets must lie in [0,{n_classes - 1}], "
            f"but observed range is [{vals_int.min()},{vals_int.max()}]."
        )
    return vals_int


def evaluate_genotype_reconstruction(
    x_true,
    *,
    genotype_likelihood: str,
    x_hat=None,
    x_logits=None,
    x_probs=None,
    obs_mask=None,
    n_genotype_classes: Optional[int] = None,
    bernoulli_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate SNP reconstruction with metrics appropriate to the observation model.

    Parameters
    ----------
    x_true
        Observed genotype matrix [N, P].
    genotype_likelihood
        One of: "mse", "bernoulli", "categorical".
    x_hat
        [N, P] reconstruction or expected dosage.
    x_logits
        Bernoulli: [N, P] logits.
        Categorical: [N, P, C] logits.
    x_probs
        Optional probabilities with the same interpretation as x_logits after
        sigmoid/softmax.
    obs_mask
        Optional [N, P] mask, 1/True for observed entries.
    n_genotype_classes
        Required for categorical if it cannot be inferred from x_logits/x_probs.
    bernoulli_threshold
        Probability threshold used for hard-call accuracy.

    Returns
    -------
    dict
        Likelihood-appropriate reconstruction diagnostics.
    """
    likelihood = str(genotype_likelihood).lower()
    if likelihood not in {"mse", "bernoulli", "categorical"}:
        raise ValueError(
            "genotype_likelihood must be one of {'mse','bernoulli','categorical'}."
        )

    x_true = np.asarray(x_true, dtype=float)
    if x_true.ndim != 2:
        raise ValueError("x_true must have shape [n_samples, n_snps].")

    mask = _prepare_mask(x_true, obs_mask)
    result: Dict[str, float] = {
        "n_observed": int(mask.sum()),
        "observed_fraction": float(mask.mean()),
    }

    if likelihood == "mse":
        if x_hat is None:
            raise ValueError("MSE reconstruction requires x_hat.")

        pred = np.asarray(x_hat, dtype=float)
        if pred.shape != x_true.shape:
            raise ValueError(
                f"x_hat shape {pred.shape} does not match x_true shape {x_true.shape}."
            )

        ok = mask & np.isfinite(pred)
        err = pred[ok] - x_true[ok]

        result.update(
            recon_mse=float(np.mean(err ** 2)),
            recon_rmse=float(np.sqrt(np.mean(err ** 2))),
            recon_mae=float(np.mean(np.abs(err))),
        )

        # Descriptive discrete-call accuracy when targets are integer-coded.
        true_vals = x_true[ok]
        if np.all(np.isclose(true_vals, np.round(true_vals))):
            hard = np.rint(pred[ok])
            result["hard_call_accuracy"] = float(
                np.mean(hard == np.round(true_vals))
            )

        return result

    if likelihood == "bernoulli":
        _validate_bernoulli_targets(x_true, mask)

        if x_logits is None and x_probs is None:
            if x_hat is None:
                raise ValueError(
                    "Bernoulli reconstruction requires x_logits, x_probs, or x_hat."
                )
            x_probs = x_hat

        if x_logits is not None:
            logits = np.asarray(x_logits, dtype=float)
            if logits.shape != x_true.shape:
                raise ValueError(
                    f"x_logits shape {logits.shape} does not match "
                    f"x_true shape {x_true.shape}."
                )
            probs = expit(logits)
            ok = mask & np.isfinite(logits)
            y = x_true[ok]
            eta = logits[ok]

            # Stable BCEWithLogits equivalent:
            # max(eta,0) - eta*y + log(1 + exp(-abs(eta)))
            bce = np.maximum(eta, 0.0) - eta * y + np.log1p(np.exp(-np.abs(eta)))
            result["recon_nll"] = float(np.mean(bce))
        else:
            probs = np.asarray(x_probs, dtype=float)
            if probs.shape != x_true.shape:
                raise ValueError(
                    f"x_probs shape {probs.shape} does not match "
                    f"x_true shape {x_true.shape}."
                )
            ok = mask & np.isfinite(probs)
            y = x_true[ok]
            p = np.clip(probs[ok], 1e-12, 1.0 - 1e-12)
            result["recon_nll"] = float(
                np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            )

        p = probs[ok]
        y = x_true[ok]
        hard = (p >= bernoulli_threshold).astype(float)

        result.update(
            brier_score=float(np.mean((p - y) ** 2)),
            hard_call_accuracy=float(np.mean(hard == y)),
            mean_predicted_alt_probability=float(np.mean(p)),
            observed_alt_fraction=float(np.mean(y)),
        )
        return result

    # categorical
    if x_logits is None and x_probs is None:
        raise ValueError(
            "Categorical reconstruction requires x_logits or x_probs."
        )

    if x_logits is not None:
        logits = np.asarray(x_logits, dtype=float)
        if logits.ndim != 3 or logits.shape[:2] != x_true.shape:
            raise ValueError(
                "Categorical x_logits must have shape [n_samples,n_snps,n_classes]."
            )
        n_classes = logits.shape[-1]
        log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
        probs = np.exp(log_probs)
    else:
        probs = np.asarray(x_probs, dtype=float)
        if probs.ndim != 3 or probs.shape[:2] != x_true.shape:
            raise ValueError(
                "Categorical x_probs must have shape [n_samples,n_snps,n_classes]."
            )
        n_classes = probs.shape[-1]
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        log_probs = np.log(probs)

    if n_genotype_classes is not None and int(n_genotype_classes) != n_classes:
        raise ValueError(
            f"Configured n_genotype_classes={n_genotype_classes}, "
            f"but decoder output has {n_classes} classes."
        )

    true_classes = _validate_categorical_targets(x_true, mask, n_classes)

    flat_log_probs = log_probs[mask]
    flat_probs = probs[mask]

    selected_logp = flat_log_probs[
        np.arange(flat_log_probs.shape[0]),
        true_classes,
    ]
    hard = np.argmax(flat_probs, axis=-1)

    classes = np.arange(n_classes, dtype=float)
    expected = np.sum(flat_probs * classes[None, :], axis=-1)

    result.update(
        recon_nll=float(-np.mean(selected_logp)),
        hard_call_accuracy=float(np.mean(hard == true_classes)),
        expected_dosage_mse=float(
            np.mean((expected - true_classes.astype(float)) ** 2)
        ),
        expected_dosage_mae=float(
            np.mean(np.abs(expected - true_classes.astype(float)))
        ),
        n_genotype_classes=int(n_classes),
    )
    return result


def reconstruction_metric_for_model_selection(metrics: Dict[str, float]) -> float:
    """
    Return a lower-is-better reconstruction quantity suitable for summaries.

    MSE models -> recon_mse
    Bernoulli/categorical -> recon_nll
    """
    if "recon_nll" in metrics:
        return float(metrics["recon_nll"])
    if "recon_mse" in metrics:
        return float(metrics["recon_mse"])
    raise KeyError("No recognized reconstruction loss is present in metrics.")
