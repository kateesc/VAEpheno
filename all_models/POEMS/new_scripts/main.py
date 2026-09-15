#!/usr/bin/env python3
"""
Single-run entry point for the supervised SNP VAE.

This launcher is intentionally compatible with the legacy POEMS+MSE setup,
while exposing the new decoder / genotype-likelihood options.

Supported experimental combinations:
    poems     + mse          : frozen/current reference
    dense_mlp + mse          : decoder-architecture ablation
    dense_mlp + bernoulli    : binary-SNP likelihood (e.g. current Arabidopsis 0/1)
    dense_mlp + categorical  : discrete genotype classes (e.g. future 0/1/2)

IMPORTANT
---------
The current legacy train.py does not yet accept decoder_type,
genotype_likelihood, dec_hidden_dim2, or n_genotype_classes.
This script detects that case:
  * legacy poems+mse still runs;
  * any new decoder/likelihood request fails loudly instead of being ignored.
After train.py is upgraded, the same CLI works without changes.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from train import train_POEMS


SUPPORTED_COMBINATIONS = {
    ("poems", "mse"),
    ("dense_mlp", "mse"),
    ("dense_mlp", "bernoulli"),
    ("dense_mlp", "categorical"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train supervised SNP-only VAE with selectable decoder architecture "
            "and genotype reconstruction likelihood."
        )
    )

    # Optimizer / training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--early_stop_metric",
        choices=["total", "recon", "trait"],
        default="trait",
    )
    parser.add_argument("--is_test", action="store_true")

    # Representation / architecture
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--enc_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--dec_hidden_dim",
        type=int,
        default=None,
        help=(
            "First decoder hidden width. If omitted: 64 for POEMS and "
            "256 for dense_mlp."
        ),
    )
    parser.add_argument(
        "--dec_hidden_dim2",
        type=int,
        default=512,
        help="Second dense-decoder hidden width; ignored by POEMS.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--decoder_type",
        choices=["poems", "dense_mlp"],
        default="poems",
    )
    parser.add_argument(
        "--genotype_likelihood",
        choices=["mse", "bernoulli", "categorical"],
        default="mse",
        help=(
            "mse: real-valued reconstruction; bernoulli: binary 0/1 SNPs; "
            "categorical: discrete classes such as 0/1/2."
        ),
    )
    parser.add_argument(
        "--n_genotype_classes",
        type=int,
        default=3,
        help="Used only for categorical reconstruction.",
    )
    parser.add_argument(
        "--decoder_l1_lambda",
        type=float,
        default=None,
        help=(
            "POEMS gate L1 coefficient. If omitted: 1e-4 for POEMS, "
            "0 for dense_mlp."
        ),
    )

    # VAE / trait objective
    parser.add_argument("--beta_kl", type=float, default=0.001)
    parser.add_argument("--alpha_trait", type=float, default=1.0)
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

    # Data / experiment naming
    parser.add_argument(
        "--experiment_note",
        type=str,
        default="plant_supervised_vae_uncertainty",
    )
    parser.add_argument("--dataset", type=str, default="plant_new")
    parser.add_argument("--trait_name", type=str, default=None)

    # Standalone train/val/test split behavior.
    parser.add_argument(
        "--split_strategy",
        choices=["auto", "random", "stratified"],
        default="auto",
    )
    parser.add_argument("--no_force_trait_outliers_train", action="store_true")
    parser.add_argument("--outlier_quantile", type=float, default=0.05)

    # Optional genotype-structure diagnostics.
    parser.add_argument("--no_infer_structure", action="store_true")
    parser.add_argument("--n_structure_pcs", type=int, default=10)
    parser.add_argument("--n_structure_clusters", type=int, default=3)

    # Reproducibility helper.
    parser.add_argument(
        "--save_cli_config",
        type=str,
        default=None,
        help="Optional JSON path to save the resolved command-line configuration.",
    )

    return parser


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    combo = (args.decoder_type, args.genotype_likelihood)
    if combo not in SUPPORTED_COMBINATIONS:
        allowed = ", ".join(f"{a}+{b}" for a, b in sorted(SUPPORTED_COMBINATIONS))
        raise ValueError(
            f"Unsupported combination {args.decoder_type}+{args.genotype_likelihood}. "
            f"Supported combinations: {allowed}"
        )

    if args.genotype_likelihood == "categorical" and args.n_genotype_classes < 2:
        raise ValueError("--n_genotype_classes must be >= 2.")

    if args.dec_hidden_dim is None:
        args.dec_hidden_dim = 64 if args.decoder_type == "poems" else 256

    if args.decoder_l1_lambda is None:
        args.decoder_l1_lambda = 1e-4 if args.decoder_type == "poems" else 0.0

    if args.decoder_type == "poems" and args.decoder_l1_lambda < 0:
        raise ValueError("--decoder_l1_lambda must be non-negative.")

    if args.decoder_type == "dense_mlp" and args.decoder_l1_lambda != 0:
        print(
            "WARNING: dense_mlp has no POEMS gate W. "
            "decoder_l1_lambda should normally be 0 unless train.py implements "
            "an explicit dense-decoder regularizer."
        )

    return args


def save_cli_config(args: argparse.Namespace) -> None:
    if args.save_cli_config is None:
        return
    path = Path(args.save_cli_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    print(f"Saved resolved CLI configuration to: {path}")


def call_train(args: argparse.Namespace):
    """
    Call train_POEMS while protecting against a partially upgraded pipeline.

    Legacy train.py supports only the original POEMS+MSE model.
    If the new train.py arguments are unavailable, a non-legacy request raises
    an explicit error rather than silently running the wrong experiment.
    """
    train_signature = inspect.signature(train_POEMS)
    accepted = set(train_signature.parameters)

    base_kwargs = dict(
        lr_in=args.lr,
        wd_in=args.wd,
        batch_size_in=args.batch_size,
        nepoch_in=args.epochs,
        experiment_note=args.experiment_note,
        dataset=args.dataset,
        trait_name=args.trait_name,
        latent_dim=args.latent_dim,
        enc_hidden_dim=args.enc_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        dropout=args.dropout,
        beta_kl=args.beta_kl,
        alpha_trait=args.alpha_trait,
        decoder_l1_lambda=args.decoder_l1_lambda,
        seed=args.seed,
        early_stop_metric=args.early_stop_metric,
        patience=args.patience,
        is_test=args.is_test,
        regressor_type=args.regressor_type,
        trait_likelihood=args.trait_likelihood,
        split_strategy=args.split_strategy,
        force_trait_outliers_train=not args.no_force_trait_outliers_train,
        outlier_quantile=args.outlier_quantile,
        infer_structure=not args.no_infer_structure,
        n_structure_pcs=args.n_structure_pcs,
        n_structure_clusters=args.n_structure_clusters,
    )

    new_kwargs = dict(
        decoder_type=args.decoder_type,
        genotype_likelihood=args.genotype_likelihood,
        dec_hidden_dim2=args.dec_hidden_dim2,
        n_genotype_classes=args.n_genotype_classes,
    )

    missing_new = [key for key in new_kwargs if key not in accepted]
    is_legacy_reference = (
        args.decoder_type == "poems"
        and args.genotype_likelihood == "mse"
    )

    if missing_new and not is_legacy_reference:
        raise RuntimeError(
            "Your current train.py has not yet been upgraded for the requested "
            "decoder/likelihood experiment. Missing train_POEMS arguments: "
            + ", ".join(missing_new)
            + ". Update train.py before running this configuration."
        )

    kwargs = dict(base_kwargs)
    for key, value in new_kwargs.items():
        if key in accepted:
            kwargs[key] = value

    print("\nResolved model configuration")
    print("----------------------------")
    print(f"decoder_type          : {args.decoder_type}")
    print(f"genotype_likelihood   : {args.genotype_likelihood}")
    print(f"n_genotype_classes    : {args.n_genotype_classes}")
    print(f"encoder hidden        : {args.enc_hidden_dim}")
    print(f"decoder hidden 1      : {args.dec_hidden_dim}")
    print(f"decoder hidden 2      : {args.dec_hidden_dim2}")
    print(f"decoder L1 lambda     : {args.decoder_l1_lambda}")
    print(f"latent_dim            : {args.latent_dim}")
    print(f"trait head            : {args.regressor_type}/{args.trait_likelihood}")
    print()

    return train_POEMS(**kwargs)


def main():
    parser = build_parser()
    args = resolve_defaults(parser.parse_args())
    save_cli_config(args)
    call_train(args)


if __name__ == "__main__":
    main()
