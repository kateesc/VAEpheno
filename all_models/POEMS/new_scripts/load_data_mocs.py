# load_data_mocs.py
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
except Exception:
    torch = None


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



# -----------------------------
# large-SNP storage helpers
# -----------------------------

NA_VALUES = {"", "NA", "NaN", "nan", "NAN", "."}


def _count_nonempty_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _read_single_column_text(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def _parse_numeric_csv_row(line: str, expected_cols: int) -> np.ndarray:
    """Fast row parser for huge numeric SNP CSV files."""
    s = line.strip()
    if not s:
        raise ValueError("Encountered an empty genotype row.")

    s_fast = s.replace("NaN", "nan").replace("NAN", "nan").replace("NA", "nan")
    arr = np.fromstring(s_fast, sep=",", dtype=np.float32)
    if arr.size == expected_cols:
        return arr

    # Conservative fallback for empty CSV fields or unusual NA syntax.
    import csv
    fields = next(csv.reader([s]))
    if len(fields) != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} SNP values, found {len(fields)}."
        )
    out = np.empty(expected_cols, dtype=np.float32)
    for j, value in enumerate(fields):
        value = value.strip()
        out[j] = np.nan if value in NA_VALUES else np.float32(value)
    return out


def build_float32_npy_cache(
    csv_path: str,
    npy_path: str,
    n_samples: int,
    n_features: int,
    overwrite: bool = False,
    progress_every: int = 25,
):
    """
    Convert 1_all.csv to a row-major float32 .npy file one accession at a time.

    This is a one-time cost. Later runs can open the file with mmap_mode='r'
    instead of parsing the full CSV and materializing it in RAM.
    """
    if os.path.exists(npy_path) and not overwrite:
        mm = np.load(npy_path, mmap_mode="r")
        if mm.shape != (n_samples, n_features):
            raise ValueError(
                f"Existing cache shape {mm.shape}; expected {(n_samples, n_features)}."
            )
        return npy_path

    tmp_path = npy_path + ".building.npy"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    mm = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_features),
    )

    row_idx = 0
    with open(csv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if row_idx >= n_samples:
                raise ValueError(
                    f"CSV contains more than the expected {n_samples} samples."
                )
            mm[row_idx] = _parse_numeric_csv_row(line, n_features)
            row_idx += 1
            if progress_every and row_idx % progress_every == 0:
                print(f"[CSV->NPY] {row_idx}/{n_samples} samples converted", flush=True)

    if row_idx != n_samples:
        raise ValueError(
            f"CSV contained {row_idx} non-empty rows; expected {n_samples}."
        )

    mm.flush()
    del mm
    os.replace(tmp_path, npy_path)

    meta = {
        "dtype": "float32",
        "shape": [int(n_samples), int(n_features)],
        "source_csv": os.path.abspath(csv_path),
        "source_size_bytes": int(os.path.getsize(csv_path)),
        "source_mtime": float(os.path.getmtime(csv_path)),
    }
    with open(npy_path + ".meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"[CSV->NPY] cache written: {npy_path}")
    return npy_path


def _fit_column_means_train_only_memmap(
    X_memmap,
    train_idx,
    sample_chunk_size: int = 4,
):
    """
    Fit train-only SNP means without materializing X_train.

    For 1.8M SNPs and sample_chunk_size=4, the main temporary genotype block
    is about 4 * 1.8M * 4 bytes ~= 29 MB.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    n_features = X_memmap.shape[1]
    sums = np.zeros(n_features, dtype=np.float64)
    counts = np.zeros(n_features, dtype=np.int32)

    for start in range(0, len(train_idx), sample_chunk_size):
        idx = train_idx[start:start + sample_chunk_size]
        block = np.asarray(X_memmap[idx, :], dtype=np.float32)
        finite = np.isfinite(block)
        sums += np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
        counts += finite.sum(axis=0, dtype=np.int32)

    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    return means.astype(np.float32)


def _fit_trait_preprocessing_train_only(Y_raw, train_idx):
    y_col_means = np.nanmean(Y_raw[train_idx], axis=0)
    y_col_means = np.where(np.isnan(y_col_means), 0.0, y_col_means).astype(np.float32)
    Y_filled = np.where(np.isnan(Y_raw), y_col_means[None, :], Y_raw).astype(np.float32)
    y_scaler = StandardScaler()
    y_scaler.fit(Y_filled[train_idx])
    Y_scaled = y_scaler.transform(Y_filled).astype(np.float32)
    return Y_filled, Y_scaled, y_col_means, y_scaler


class LazySNPDataset:
    """
    Row-lazy PyTorch dataset backed by a memory-mapped .npy genotype matrix.

    Each __getitem__ loads one accession, computes its observed mask, and
    applies TRAIN-ONLY SNP imputation means. The full filled matrix and full
    missingness mask are never held in RAM.
    """
    def __init__(
        self,
        npy_path,
        indices,
        x_col_means,
        y_scaled,
        y_raw,
        trait_mask,
        labels,
        sample_names,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use LazySNPDataset.")
        self.npy_path = str(npy_path)
        self.indices = np.asarray(indices, dtype=int)
        self.x_col_means = np.asarray(x_col_means, dtype=np.float32)
        self.y_scaled = np.asarray(y_scaled, dtype=np.float32)
        self.y_raw = np.asarray(y_raw, dtype=np.float32)
        self.trait_mask = np.asarray(trait_mask, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=int)
        self.sample_names = np.asarray(sample_names)
        self._mm = None

    def _get_memmap(self):
        if self._mm is None:
            self._mm = np.load(self.npy_path, mmap_mode="r")
        return self._mm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        global_idx = int(self.indices[item])
        row = np.asarray(self._get_memmap()[global_idx], dtype=np.float32).copy()
        obs = np.isfinite(row)
        x_in = np.where(obs, row, self.x_col_means).astype(np.float32)
        return {
            "x_raw": torch.from_numpy(row),
            "x_in": torch.from_numpy(x_in),
            "obs_mask": torch.from_numpy(obs.astype(np.float32)),
            "y": torch.from_numpy(self.y_scaled[global_idx]),
            "y_raw": torch.from_numpy(self.y_raw[global_idx]),
            "trait_mask": torch.from_numpy(self.trait_mask[global_idx]),
            "label": torch.tensor(self.labels[global_idx], dtype=torch.long),
            "sample_name": str(self.sample_names[global_idx]),
            "global_idx": global_idx,
        }


def make_large_snp_dataloaders(
    loaded: Dict,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Build DataLoaders for the memmap backend."""
    if torch is None:
        raise ImportError("PyTorch is required to create DataLoaders.")
    if loaded.get("storage_backend") != "memmap":
        raise ValueError("Expected storage_backend='memmap'.")

    common = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
    )
    return {
        "train": torch.utils.data.DataLoader(
            loaded["train_dataset"], shuffle=True, drop_last=False, **common
        ),
        "val": torch.utils.data.DataLoader(
            loaded["val_dataset"], shuffle=False, drop_last=False, **common
        ),
        "test": torch.utils.data.DataLoader(
            loaded["test_dataset"], shuffle=False, drop_last=False, **common
        ),
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
    storage_backend: str = "auto",          # auto | memory | memmap
    large_snp_threshold: int = 250_000,
    build_memmap_cache: bool = True,
    overwrite_memmap_cache: bool = False,
    mean_sample_chunk_size: int = 4,
    load_snp_names: Optional[bool] = None,
):
    """
    Leakage-safe loader with an optional large-SNP memmap backend.

    memory backend
    --------------
    Preserves the original broad return structure and is recommended for
    moderate SNP counts such as the current ~74,933-SNP Arabidopsis matrix.

    memmap backend
    --------------
    Recommended for matrices on the order of 1.8M SNPs. It creates/uses
    data/<dataset>/1_all.float32.npy, fits train-only imputation means in
    chunks, and returns LazySNPDataset objects rather than materializing
    X_filled and M for the full matrix.

    explicit split overrides are authoritative: force_outliers_train is not
    allowed to move samples once train/val/test indices are supplied.
    """
    if storage_backend not in {"auto", "memory", "memmap"}:
        raise ValueError("storage_backend must be 'auto', 'memory', or 'memmap'.")

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
    cache_path = os.path.join(data_dir, "1_all.float32.npy")

    for required in [snp_path, snp_name_path, trait_path, trait_name_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required input file not found: {required}")

    # Traits are small; load them normally.
    Y_raw_all = pd.read_csv(
        trait_path, header=None, na_values=["NA", "NaN", "nan", ""]
    ).to_numpy(dtype=np.float32)
    n_samples = Y_raw_all.shape[0]

    all_trait_names = _read_single_column_text(trait_name_path)
    if len(all_trait_names) != Y_raw_all.shape[1]:
        raise ValueError("Mismatch between 2_all.csv columns and 2_featname.csv rows.")

    # Avoid loading 1.8M feature-name strings unless downstream code needs them.
    n_features = _count_nonempty_lines(snp_name_path)
    cache_exists = os.path.exists(cache_path)

    if storage_backend == "auto":
        use_memmap = cache_exists or n_features >= int(large_snp_threshold)
    else:
        use_memmap = storage_backend == "memmap"
    backend = "memmap" if use_memmap else "memory"

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

    explicit = (
        train_idx_override is not None
        and val_idx_override is not None
        and test_idx_override is not None
    )

    if explicit:
        train_idx = np.asarray(train_idx_override, dtype=int)
        val_idx = np.asarray(val_idx_override, dtype=int)
        test_idx = np.asarray(test_idx_override, dtype=int)
        if force_outliers_train:
            print(
                "Explicit split overrides supplied: force_outliers_train is ignored."
            )
    else:
        train_idx, val_idx, test_idx = split_indices(
            n_samples=n_samples,
            labels=labels,
            has_labels=has_labels,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )
        if force_outliers_train:
            train_idx, val_idx, test_idx = _force_trait_outliers_into_train(
                train_idx, val_idx, test_idx, Y_raw, outlier_quantile
            )

    Ymask = (~np.isnan(Y_raw)).astype(np.float32)
    Y_filled, Y_scaled, y_col_means, y_scaler = _fit_trait_preprocessing_train_only(
        Y_raw, train_idx
    )

    if backend == "memory":
        X_raw = pd.read_csv(
            snp_path,
            header=None,
            na_values=["NA", "NaN", "nan", ""],
        ).to_numpy(dtype=np.float32)

        if X_raw.shape != (n_samples, n_features):
            raise ValueError(
                f"1_all.csv shape {X_raw.shape} does not match expected "
                f"{(n_samples, n_features)}."
            )

        snp_names = _read_single_column_text(snp_name_path)
        x_col_means = _fit_column_means_train_only(X_raw[train_idx])
        X_filled = _apply_column_means(X_raw, x_col_means)
        M = (~np.isnan(X_raw)).astype(np.float32)

        return {
            "storage_backend": "memory",
            "input_dim": n_features,
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
            "snp_name_path": snp_name_path,
            "selected_trait_names": selected_trait_names,
            "all_trait_names": all_trait_names,
            "label_map": label_map,
            "has_labels": has_labels,
            "y_scaler": y_scaler,
            "x_col_means": x_col_means,
            "y_col_means": y_col_means,
            "X_raw": X_raw,
            "X_filled": X_filled,
            "M": M,
            "Y_raw": Y_raw,
            "Y_scaled": Y_scaled,
            "Ymask": Ymask,
            "labels": labels,
            "sample_names": sample_names,
        }

    # Large-data memmap backend.
    if not cache_exists:
        if not build_memmap_cache:
            raise FileNotFoundError(
                f"Memmap cache not found: {cache_path}. "
                "Set build_memmap_cache=True to create it."
            )
        build_float32_npy_cache(
            csv_path=snp_path,
            npy_path=cache_path,
            n_samples=n_samples,
            n_features=n_features,
            overwrite=overwrite_memmap_cache,
        )

    X_mm = np.load(cache_path, mmap_mode="r")
    if X_mm.shape != (n_samples, n_features):
        raise ValueError(
            f"Memmap shape {X_mm.shape} does not match expected {(n_samples, n_features)}."
        )

    print(
        f"[load_data_mocs] memmap backend: {n_samples} samples x {n_features:,} SNPs"
    )
    print(
        f"[load_data_mocs] fitting train-only SNP means in sample chunks of "
        f"{mean_sample_chunk_size}"
    )

    x_col_means = _fit_column_means_train_only_memmap(
        X_mm, train_idx, sample_chunk_size=mean_sample_chunk_size
    )

    if load_snp_names is None:
        load_snp_names = False
    snp_names = _read_single_column_text(snp_name_path) if load_snp_names else None

    train_dataset = LazySNPDataset(
        cache_path, train_idx, x_col_means, Y_scaled, Y_raw, Ymask, labels, sample_names
    )
    val_dataset = LazySNPDataset(
        cache_path, val_idx, x_col_means, Y_scaled, Y_raw, Ymask, labels, sample_names
    )
    test_dataset = LazySNPDataset(
        cache_path, test_idx, x_col_means, Y_scaled, Y_raw, Ymask, labels, sample_names
    )

    return {
        "storage_backend": "memmap",
        "input_dim": n_features,
        "trait_dim": Y_raw.shape[1],
        "n_samples": n_samples,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "X_memmap": X_mm,
        "X_memmap_path": cache_path,
        "X_raw": X_mm,
        "X_filled": None,
        "M": None,
        "Y_train": Y_scaled[train_idx],
        "Y_val": Y_scaled[val_idx],
        "Y_test": Y_scaled[test_idx],
        "Y_train_raw": Y_raw[train_idx],
        "Y_val_raw": Y_raw[val_idx],
        "Y_test_raw": Y_raw[test_idx],
        "Y_raw": Y_raw,
        "Y_scaled": Y_scaled,
        "Ymask": Ymask,
        "labels": labels,
        "sample_names": sample_names,
        "has_labels": has_labels,
        "label_map": label_map,
        "snp_names": snp_names,
        "snp_name_path": snp_name_path,
        "selected_trait_names": selected_trait_names,
        "all_trait_names": all_trait_names,
        "x_col_means": x_col_means,
        "y_col_means": y_col_means,
        "y_scaler": y_scaler,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare/load SNP data with optional large-SNP memmap caching."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--trait_name", default=None)
    parser.add_argument(
        "--storage_backend", choices=["auto", "memory", "memmap"], default="auto"
    )
    parser.add_argument("--large_snp_threshold", type=int, default=250000)
    parser.add_argument("--overwrite_memmap_cache", action="store_true")
    parser.add_argument("--mean_sample_chunk_size", type=int, default=4)
    parser.add_argument("--load_snp_names", action="store_true")
    args = parser.parse_args()

    data = load_data_mocs(
        dataset=args.dataset,
        root_dir=args.root_dir,
        trait_name=args.trait_name,
        storage_backend=args.storage_backend,
        large_snp_threshold=args.large_snp_threshold,
        build_memmap_cache=True,
        overwrite_memmap_cache=args.overwrite_memmap_cache,
        mean_sample_chunk_size=args.mean_sample_chunk_size,
        load_snp_names=args.load_snp_names,
    )

    print("\nLoaded dataset")
    print("--------------")
    print(f"backend: {data['storage_backend']}")
    print(f"samples: {len(data['sample_names'])}")
    print(f"SNPs: {data['input_dim']:,}")
    print(f"traits: {data['selected_trait_names']}")
    print(
        f"train/val/test: {len(data['train_idx'])}/"
        f"{len(data['val_idx'])}/{len(data['test_idx'])}"
    )
    if data["storage_backend"] == "memmap":
        print(f"memmap: {data['X_memmap_path']}")
