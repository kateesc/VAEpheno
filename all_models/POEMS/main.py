import argparse
from train import train_POEMS


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised SNP-only VAE with Gaussian trait uncertainty and no posthoc baselines."
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--enc_hidden_dim", type=int, default=128)
    parser.add_argument("--dec_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.001)
    parser.add_argument("--alpha_trait", type=float, default=1.0)
    parser.add_argument("--decoder_l1_lambda", type=float, default=1e-4)
    parser.add_argument("--experiment_note", type=str, default="plant_supervised_vae_uncertainty")
    parser.add_argument("--dataset", type=str, default="plant_new")
    parser.add_argument("--trait_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--early_stop_metric", type=str, default="trait", choices=["total", "recon", "trait"])
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--is_test", action="store_true")

    parser.add_argument("--regressor_type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--trait_likelihood", choices=["gaussian", "mse"], default="gaussian")

    parser.add_argument("--split_strategy", choices=["auto", "random", "stratified"], default="auto")
    parser.add_argument("--no_force_trait_outliers_train", action="store_true")
    parser.add_argument("--outlier_quantile", type=float, default=0.05)

    parser.add_argument("--no_infer_structure", action="store_true")
    parser.add_argument("--n_structure_pcs", type=int, default=10)
    parser.add_argument("--n_structure_clusters", type=int, default=3)
    parser.add_argument("--cv_strategy",type=str,default="stratified",choices=["stratified", "random", "pca_blocked"])

    args = parser.parse_args()

    train_POEMS(
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


if __name__ == "__main__":
    main()
