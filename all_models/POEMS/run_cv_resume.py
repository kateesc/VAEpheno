import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    GroupKFold,
    train_test_split,
)

from train import train_POEMS, load_plant_data, infer_snp_structure, root_dir
import util


def read_best_params(path):
    if path is None:
        return {}

    path = Path(path)

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        df = pd.read_csv(path)

        if "state" in df.columns:
            df = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()

        if "value" in df.columns:
            df = df[np.isfinite(pd.to_numeric(df["value"], errors="coerce"))].copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.sort_values("value", ascending=True)

        if df.empty:
            raise ValueError(f"No completed finite trials found in {path}")

        raw = df.iloc[0].to_dict()

    aliases = {
        "lr": "lr_in",
        "wd": "wd_in",
        "params_lr": "lr_in",
        "params_wd": "wd_in",
        "param_lr": "lr_in",
        "param_wd": "wd_in",
        "params_latent_dim": "latent_dim",
        "params_alpha_trait": "alpha_trait",
        "params_beta_kl": "beta_kl",
        "params_decoder_l1_lambda": "decoder_l1_lambda",
        "params_dropout": "dropout",
        "params_early_stop_metric": "early_stop_metric",
        "param_latent_dim": "latent_dim",
        "param_alpha_trait": "alpha_trait",
        "param_beta_kl": "beta_kl",
        "param_decoder_l1_lambda": "decoder_l1_lambda",
        "param_dropout": "dropout",
        "param_early_stop_metric": "early_stop_metric",
    }

    out = {}
    for k, v in raw.items():
        kk = aliases.get(k, k)

        if kk.startswith("params_"):
            kk = kk.replace("params_", "", 1)
        if kk.startswith("param_"):
            kk = kk.replace("param_", "", 1)

        kk = aliases.get(kk, kk)
        out[kk] = v

    return out


def make_cv_splits(args, data):
    idx = np.arange(data["x_raw"].shape[0])
    labels = data["labels"]

    if args.cv_strategy == "random":
        cv = RepeatedKFold(
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed,
        )

        for trainval_idx, test_idx in cv.split(idx):
            yield trainval_idx, test_idx, None

    elif args.cv_strategy == "label_stratified":
        if not data["has_labels"]:
            raise ValueError("label_stratified CV requested, but labels_all.csv is missing.")

        cv = RepeatedStratifiedKFold(
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed,
        )

        for trainval_idx, test_idx in cv.split(idx, labels):
            yield trainval_idx, test_idx, None

    elif args.cv_strategy == "pca_blocked":
        # Unsupervised structure groups from SNPs.
        # These are used as blocks, not as biological truth.
        structure_df, pve, clusters = infer_snp_structure(
            data["x_raw"],
            train_idx=None,
            n_pcs=args.n_structure_pcs,
            n_clusters=args.n_structure_clusters,
            random_state=args.seed,
        )

        os.makedirs(args.out_dir, exist_ok=True)

        structure_df.insert(0, "sample_name", data["sample_names"])
        structure_df.to_csv(
            os.path.join(args.out_dir, "global_SNP_PCA_clusters_for_blocked_CV.csv"),
            index=False,
        )

        pd.DataFrame(
            {
                "PC": np.arange(1, len(pve) + 1),
                "explained_variance_ratio": pve,
            }
        ).to_csv(
            os.path.join(args.out_dir, "global_SNP_PCA_variance_for_blocked_CV.csv"),
            index=False,
        )

        # GroupKFold has no repeats by itself.
        # To keep blocks intact, repeats are not randomized.
        cv = GroupKFold(n_splits=min(args.n_splits, len(np.unique(clusters))))

        for trainval_idx, test_idx in cv.split(idx, groups=clusters):
            yield trainval_idx, test_idx, clusters

    else:
        raise ValueError(f"Unknown cv_strategy: {args.cv_strategy}")


def save_csv_atomic(df, path):
    """
    Write CSV safely. This avoids leaving a half-written CSV if the job dies
    during the write step.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def load_completed_runs(running_csv):
    """
    Load previous completed fold/ensemble runs from cv_metrics_running.csv.

    Completion is defined by the presence of a row with both:
        fold
        ensemble_member

    This is appropriate because run_cv writes one row only after train_POEMS()
    finishes for that fold/ensemble member.
    """
    running_csv = Path(running_csv)

    if not running_csv.exists():
        return [], set()

    old_df = pd.read_csv(running_csv)

    if old_df.empty:
        return [], set()

    required_cols = {"fold", "ensemble_member"}

    if not required_cols.issubset(old_df.columns):
        print(
            f"Resume requested, but {running_csv} does not contain "
            f"required columns {required_cols}. Starting with no completed runs."
        )
        return [], set()

    # If there are accidental duplicate rows, keep the last one.
    old_df["fold"] = old_df["fold"].astype(int)
    old_df["ensemble_member"] = old_df["ensemble_member"].astype(int)

    old_df = old_df.drop_duplicates(
        subset=["fold", "ensemble_member"],
        keep="last",
    ).reset_index(drop=True)

    completed = set(
        zip(
            old_df["fold"].astype(int),
            old_df["ensemble_member"].astype(int),
        )
    )

    rows = old_df.to_dict("records")

    return rows, completed


def print_resume_status(args, completed):
    """
    Print expected and missing fold/ensemble combinations.
    """
    if args.cv_strategy in ["random", "label_stratified"]:
        expected_n_folds = args.n_splits * args.n_repeats
    else:
        # pca_blocked uses GroupKFold without repeats.
        expected_n_folds = args.n_splits

    expected = set()

    for fold in range(1, expected_n_folds + 1):
        for ens in range(1, args.ensemble_seeds + 1):
            expected.add((fold, ens))

    missing = sorted(expected - completed)

    print("Resume status")
    print("-------------")
    print(f"Expected fold/ensemble runs: {len(expected)}")
    print(f"Completed fold/ensemble runs found: {len(completed)}")
    print(f"Missing fold/ensemble runs: {len(missing)}")

    if missing:
        print("Missing combinations:")
        print(missing)
    else:
        print("No missing combinations detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Repeated CV / ensemble evaluation for uncertainty VAE."
    )

    parser.add_argument("--dataset", default="plant_new")
    parser.add_argument("--trait_name", default="DTF")
    parser.add_argument("--experiment_note", default="plant_supervised_vae_cv_uncertainty")
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--enc_hidden_dim", type=int, default=128)
    parser.add_argument("--dec_hidden_dim", type=int, default=64)

    parser.add_argument("--best_params", default=None)
    parser.add_argument("--out_dir", default="cv_outputs_uncertainty")

    parser.add_argument(
        "--cv_strategy",
        choices=["random", "label_stratified", "pca_blocked"],
        default="random",
    )

    parser.add_argument(
        "--regressor_type",
        choices=["mlp", "linear"],
        default="mlp",
    )

    parser.add_argument(
        "--trait_likelihood",
        choices=["gaussian", "mse"],
        default="gaussian",
    )

    parser.add_argument("--n_structure_pcs", type=int, default=10)
    parser.add_argument("--n_structure_clusters", type=int, default=3)

    parser.add_argument("--no_force_trait_outliers_train", action="store_true")
    parser.add_argument("--outlier_quantile", type=float, default=0.05)

    parser.add_argument(
        "--ensemble_seeds",
        type=int,
        default=1,
        help="Number of independent seeds per CV split.",
    )

    # New argument for resuming interrupted CV.
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume CV by reading cv_metrics_running.csv in out_dir and "
            "skipping completed fold/ensemble runs."
        ),
    )

    args = parser.parse_args()

    defaults = {
        "lr_in": 1e-3,
        "wd_in": 1e-5,
        "latent_dim": 16,
        "alpha_trait": 1.0,
        "beta_kl": 0.001,
        "decoder_l1_lambda": 1e-4,
        "dropout": 0.2,
        "early_stop_metric": "trait",
    }

    defaults.update(
        {
            k: v
            for k, v in read_best_params(args.best_params).items()
            if k in defaults
        }
    )

    data_dir = os.path.join(root_dir, "data", args.dataset)
    data = load_plant_data(data_dir, trait_name=args.trait_name, require_labels=False)
    labels = data["labels"]

    os.makedirs(args.out_dir, exist_ok=True)

    running_csv = os.path.join(args.out_dir, "cv_metrics_running.csv")
    final_csv = os.path.join(args.out_dir, "cv_metrics.csv")
    summary_csv = os.path.join(args.out_dir, "cv_metrics_summary.csv")

    rows = []
    completed = set()

    if args.resume:
        rows, completed = load_completed_runs(running_csv)

        print(f"Resume mode is ON.")
        print(f"Reading previous runs from: {running_csv}")
        print(f"Loaded previous completed rows: {len(rows)}")
        print_resume_status(args, completed)

    else:
        if os.path.exists(running_csv):
            print(
                f"Warning: {running_csv} already exists, but --resume was not used. "
                "This run may duplicate or overwrite previous results."
            )

    fold_id = 0

    for trainval_idx, test_idx, groups in make_cv_splits(args, data):
        fold_id += 1

        # Inner validation split from training data.
        # Use labels only if they exist and are usable.
        strat = None

        if data["has_labels"] and len(np.unique(labels[trainval_idx])) > 1:
            counts = np.bincount(labels[trainval_idx])

            if counts.min() >= 2:
                strat = labels[trainval_idx]

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.15,
            random_state=args.seed + fold_id,
            shuffle=True,
            stratify=strat,
        )

        for ens in range(args.ensemble_seeds):
            ensemble_member = ens + 1

            if args.resume and (fold_id, ensemble_member) in completed:
                print(f"Skipping completed fold {fold_id}, ensemble {ensemble_member}")
                continue

            run_seed = args.seed + 1000 * fold_id + ens

            print("")
            print("=" * 80)
            print(f"Running fold {fold_id}, ensemble {ensemble_member}")
            print(f"Seed: {run_seed}")
            print("=" * 80)

            result = train_POEMS(
                lr_in=float(defaults["lr_in"]),
                wd_in=float(defaults["wd_in"]),
                batch_size_in=args.batch_size,
                nepoch_in=args.epochs,
                experiment_note=f"{args.experiment_note}_fold_{fold_id}_ens_{ensemble_member}",
                dataset=args.dataset,
                trait_name=args.trait_name,
                latent_dim=int(defaults["latent_dim"]),
                enc_hidden_dim=args.enc_hidden_dim,
                dec_hidden_dim=args.dec_hidden_dim,
                dropout=float(defaults["dropout"]),
                beta_kl=float(defaults["beta_kl"]),
                alpha_trait=float(defaults["alpha_trait"]),
                decoder_l1_lambda=float(defaults["decoder_l1_lambda"]),
                seed=run_seed,
                early_stop_metric=str(defaults["early_stop_metric"]),
                patience=args.patience,
                regressor_type=args.regressor_type,
                trait_likelihood=args.trait_likelihood,
                train_idx_override=train_idx,
                val_idx_override=val_idx,
                test_idx_override=test_idx,
                force_trait_outliers_train=not args.no_force_trait_outliers_train,
                outlier_quantile=args.outlier_quantile,
                infer_structure=False,
            )

            result["fold"] = fold_id
            result["ensemble_member"] = ensemble_member
            result["seed"] = run_seed
            result["cv_strategy"] = args.cv_strategy

            rows.append(result)
            completed.add((fold_id, ensemble_member))

            running_df = pd.DataFrame(rows)
            save_csv_atomic(running_df, running_csv)

            print(f"Saved running metrics to: {running_csv}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No CV rows were produced or loaded. "
            "Check cv_metrics_running.csv and your --resume settings."
        )

    save_csv_atomic(df, final_csv)

    metric_cols = [
        c
        for c in [
            "test_mse_mean",
            "test_r2_mean",
            "test_pearson_mean",
            "test_trait_loss",
            "test_recon_loss",
        ]
        if c in df.columns
    ]

    if metric_cols:
        summary = (
            df[metric_cols]
            .agg(["mean", "std", "min", "max"])
            .T
            .reset_index()
            .rename(columns={"index": "metric"})
        )

        save_csv_atomic(summary, summary_csv)

        print("")
        print("CV summary")
        print("----------")
        print(summary)

    else:
        print("Warning: No expected metric columns found for summary.")
        summary = pd.DataFrame()

    # This may depend on what train_POEMS stores in the result dictionary.
    # If it fails after resume because old prediction arrays were loaded from CSV,
    # the metric files are still safely written.
    try:
        ensemble_summary = util.summarize_ensemble_predictions(df, out_dir=args.out_dir)

        if not ensemble_summary.empty:
            util.plot_ensemble_uncertainty(
                ensemble_summary,
                out_dir=args.out_dir,
                filename="Ensemble_Uncertainty.pdf",
            )

    except Exception as e:
        print("")
        print("Warning: ensemble uncertainty summary/plot failed.")
        print("This does not affect cv_metrics.csv or cv_metrics_summary.csv.")
        print(f"Error was: {repr(e)}")

    print("")
    print("Finished CV.")
    print(f"Final metrics saved to: {final_csv}")

    if metric_cols:
        print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
