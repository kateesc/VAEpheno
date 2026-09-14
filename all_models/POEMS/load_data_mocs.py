from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _read_optional_sample_names(sample_path: str, n_samples: int):
    if os.path.exists(sample_path):
        sample_names = pd.read_csv(sample_path, header=None).iloc[:, 0].astype(str).tolist()
        if len(sample_names) != n_samples:
            raise ValueError("Mismatch between samples.txt and input rows.")
        return sample_names
    return [f"sample_{i}" for i in range(n_samples)]


def _read_optional_labels(label_path: str, n_samples: int):
    """
    labels_all.csv is now optional.

    If missing:
        labels = zeros
        has_labels = False
    """
    if os.path.exists(label_path):
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].astype(int).to_numpy()
        if len(labels) != n_samples:
            raise ValueError("Mismatch between labels_all.csv and input rows.")
        return labels, True

    return np.zeros(n_samples, dtype=int), False


def _read_optional_label_map(label_map_path: str):
    if not os.path.exists(label_map_path):
        return None

    tmp = pd.read_csv(label_map_path)
    if not {"label", "population"}.issubset(tmp.columns):
        return None

    return {int(r["label"]): str(r["population"]) for _, r in tmp.iterrows()}


def _fit_column_means_train_only(X_train_raw: np.ndarray):
    means = np.nanmean(X_train_raw, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means.astype(np.float32)


def _apply_column_means(X_raw: np.ndarray, col_means: np.ndarray):
    return np.where(np.isnan(X_raw), col_means[None, :], X_raw).astype(np.float32)


def _force_trait_outliers_into_train(
    train_idx,
    val_idx,
    test_idx,
    y_raw,
    outlier_quantile: float = 0.05,
):
    """
    Ensure extreme trait values are present in training.

    Extreme samples are defined as:
        y <= q_low or y >= q_high

    If an extreme sample is in val/test, it is moved to train.
    This is optional and intended for small biological datasets where rare
    flowering-time extremes should be represented during training.
    """
    y = np.asarray(y_raw)
    if y.ndim == 2:
        y1 = y[:, 0]
    else:
        y1 = y

    valid = np.isfinite(y1)
    if valid.sum() < 5:
        return train_idx, val_idx, test_idx

    q_low = np.nanquantile(y1[valid], outlier_quantile)
    q_high = np.nanquantile(y1[valid], 1.0 - outlier_quantile)

    outlier_idx = np.where(valid & ((y1 <= q_low) | (y1 >= q_high)))[0]

    train_set = set(map(int, train_idx))
    val_set = set(map(int, val_idx))
    test_set = set(map(int, test_idx))

    for i in map(int, outlier_idx):
        train_set.add(i)
        val_set.discard(i)
        test_set.discard(i)

    train_idx = np.array(sorted(train_set), dtype=int)
    val_idx = np.array(sorted(val_set), dtype=int)
    test_idx = np.array(sorted(test_set), dtype=int)

    return train_idx, val_idx, test_idx


def split_indices(
    n_samples: int,
    labels: np.ndarray,
    has_labels: bool,
    seed: int = 21,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    idx = np.arange(n_samples)

    stratify_1 = labels if has_labels and len(np.unique(labels)) > 1 else None

    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_1,
    )

    relative_val_size = val_size / (1.0 - test_size)

    stratify_2 = labels[trainval_idx] if stratify_1 is not None else None

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val_size,
        random_state=seed,
        stratify=stratify_2,
    )

    return train_idx, val_idx, test_idx


def preprocess_train_val_test(
    X_raw,
    Y_raw,
    train_idx,
    val_idx,
    test_idx,
    force_outliers_train: bool = False,
    outlier_quantile: float = 0.05,
):
    """
    Leakage-safe preprocessing.

    Important:
        SNP imputation means are fitted on training samples only.
        Trait scaler is fitted on training samples only.
        Validation/test are transformed using training-fitted parameters.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if force_outliers_train:
        train_idx, val_idx, test_idx = _force_trait_outliers_into_train(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y_raw=Y_raw,
            outlier_quantile=outlier_quantile,
        )

    M = (~np.isnan(X_raw)).astype(np.float32)
    Ymask = (~np.isnan(Y_raw)).astype(np.float32)

    # SNP imputation: fit means on train only
    x_col_means = _fit_column_means_train_only(X_raw[train_idx])
    X_filled = _apply_column_means(X_raw, x_col_means)

    # Trait filling before scaling: fit trait means on train only
    y_col_means = np.nanmean(Y_raw[train_idx], axis=0)
    y_col_means = np.where(np.isnan(y_col_means), 0.0, y_col_means).astype(np.float32)
    Y_filled = np.where(np.isnan(Y_raw), y_col_means[None, :], Y_raw).astype(np.float32)

    # Trait scaling: fit scaler on train only
    y_scaler = StandardScaler()
    y_scaler.fit(Y_filled[train_idx])
    Y_scaled = y_scaler.transform(Y_filled).astype(np.float32)

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_raw": X_raw,
        "X_filled": X_filled,
        "M": M,
        "Y_raw": Y_raw,
        "Y_filled": Y_filled,
        "Y_scaled": Y_scaled,
        "Ymask": Ymask,
        "x_col_means": x_col_means,
        "y_col_means": y_col_means,
        "y_scaler": y_scaler,
    }


def load_data_mocs(
    dataset: str = "plant_new",
    root_dir: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 21,
    trait_name: Optional[str] = None,
    train_idx_override: Optional[Sequence[int]] = None,
    val_idx_override: Optional[Sequence[int]] = None,
    test_idx_override: Optional[Sequence[int]] = None,
    force_outliers_train: bool = False,
    outlier_quantile: float = 0.05,
):
    """
    Leakage-safe replacement for the original load_data_mocs().

    Keeps the same input-file syntax:
        data/<dataset>/1_all.csv
        data/<dataset>/1_featname.csv
        data/<dataset>/2_all.csv
        data/<dataset>/2_featname.csv
        data/<dataset>/labels_all.csv      optional
        data/<dataset>/labels_map.csv      optional
        data/<dataset>/samples.txt         optional

    Returns the same broad object style as before, but with leakage-safe
    train-only SNP imputation and train-only trait scaling.
    """

    if root_dir is None:
        root_dir = os.path.abspath(os.path.dirname(__file__))

    data_dir = os.path.join(root_dir, "data", dataset)

    snp_path = os.path.join(data_dir, "1_all.csv")
    snp_name_path = os.path.join(data_dir, "1_featname.csv")
    trait_path = os.path.join(data_dir, "2_all.csv")
    trait_name_path = os.path.join(data_dir, "2_featname.csv")
    label_path = os.path.join(data_dir, "labels_all.csv")
    label_map_path = os.path.join(data_dir, "labels_map.csv")
    sample_path = os.path.join(data_dir, "samples.txt")

    X_df = pd.read_csv(snp_path, header=None, na_values=["NA", "NaN", "nan", ""])
    Y_df = pd.read_csv(trait_path, header=None, na_values=["NA", "NaN", "nan", ""])

    snp_names = pd.read_csv(snp_name_path, header=None).iloc[:, 0].astype(str).tolist()
    all_trait_names = pd.read_csv(trait_name_path, header=None).iloc[:, 0].astype(str).tolist()

    X_raw = X_df.to_numpy(dtype=np.float32)
    Y_raw_all = Y_df.to_numpy(dtype=np.float32)

    if len(snp_names) != X_raw.shape[1]:
        raise ValueError("Mismatch between 1_all.csv columns and 1_featname.csv rows.")
    if len(all_trait_names) != Y_raw_all.shape[1]:
        raise ValueError("Mismatch between 2_all.csv columns and 2_featname.csv rows.")
    if X_raw.shape[0] != Y_raw_all.shape[0]:
        raise ValueError("Mismatch between rows of 1_all.csv and 2_all.csv.")

    n_samples = X_raw.shape[0]

    labels, has_labels = _read_optional_labels(label_path, n_samples)
    label_map = _read_optional_label_map(label_map_path)
    sample_names = _read_optional_sample_names(sample_path, n_samples)

    if trait_name is None:
        Y_raw = Y_raw_all
        selected_trait_names = all_trait_names
    else:
        if trait_name not in all_trait_names:
            raise ValueError(
                f"Trait '{trait_name}' not found in 2_featname.csv. "
                f"Available traits: {all_trait_names}"
            )
        trait_idx = all_trait_names.index(trait_name)
        Y_raw = Y_raw_all[:, [trait_idx]]
        selected_trait_names = [trait_name]

    if train_idx_override is not None and val_idx_override is not None and test_idx_override is not None:
        train_idx = np.asarray(train_idx_override, dtype=int)
        val_idx = np.asarray(val_idx_override, dtype=int)
        test_idx = np.asarray(test_idx_override, dtype=int)
    else:
        train_idx, val_idx, test_idx = split_indices(
            n_samples=n_samples,
            labels=labels,
            has_labels=has_labels,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )

    prep = preprocess_train_val_test(
        X_raw=X_raw,
        Y_raw=Y_raw,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        force_outliers_train=force_outliers_train,
        outlier_quantile=outlier_quantile,
    )

    train_idx = prep["train_idx"]
    val_idx = prep["val_idx"]
    test_idx = prep["test_idx"]

    X_filled = prep["X_filled"]
    M = prep["M"]
    Y_scaled = prep["Y_scaled"]
    Ymask = prep["Ymask"]

    result = {
        "input_dim": X_raw.shape[1],
        "trait_dim": Y_raw.shape[1],

        "X_train_raw": X_raw[train_idx],
        "X_train_filled": X_filled[train_idx],
        "M_train": M[train_idx],
        "Y_train": Y_scaled[train_idx],
        "Ymask_train": Ymask[train_idx],
        "L_train": labels[train_idx],
        "S_train": np.asarray(sample_names)[train_idx].tolist(),

        "X_val_raw": X_raw[val_idx],
        "X_val_filled": X_filled[val_idx],
        "M_val": M[val_idx],
        "Y_val": Y_scaled[val_idx],
        "Ymask_val": Ymask[val_idx],
        "L_val": labels[val_idx],
        "S_val": np.asarray(sample_names)[val_idx].tolist(),

        "X_test_raw": X_raw[test_idx],
        "X_test_filled": X_filled[test_idx],
        "M_test": M[test_idx],
        "Y_test": Y_scaled[test_idx],
        "Ymask_test": Ymask[test_idx],
        "L_test": labels[test_idx],
        "S_test": np.asarray(sample_names)[test_idx].tolist(),

        "Y_train_raw": Y_raw[train_idx],
        "Y_val_raw": Y_raw[val_idx],
        "Y_test_raw": Y_raw[test_idx],

        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,

        "snp_names": snp_names,
        "selected_trait_names": selected_trait_names,
        "all_trait_names": all_trait_names,
        "label_map": label_map,
        "has_labels": has_labels,
        "y_scaler": prep["y_scaler"],
        "x_col_means": prep["x_col_means"],
        "y_col_means": prep["y_col_means"],

        # Full arrays for downstream PCA/structure plots if needed
        "X_raw": X_raw,
        "X_filled": X_filled,
        "M": M,
        "Y_raw": Y_raw,
        "Y_scaled": Y_scaled,
        "Ymask": Ymask,
        "labels": labels,
        "sample_names": sample_names,
    }

    return result
