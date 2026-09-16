Decoder / genotype-likelihood ablation pipeline

This bundle upgrades the existing supervised SNP-VAE so decoder architecture
and genotype reconstruction likelihood are explicit and reproducible.

Files

Core:

models.py

train.py

run_cv_resume.py

bayes_opt_corrpenalty_calibration_aware_fixed.py

util.py

Companion:

main.py

evaluation.py

Keep your existing:

setup_seed.py

load_data_mocs.py / other project-specific helpers if used elsewhere

requirements.txt

data/...

Supported model combinations

poems + mse

frozen/current reference

POEMS SNP-specific latent gate W

POEMS gate L1 regularization

dense_mlp + mse

Andreas-style dense decoder

architecture-only comparison against POEMS+MSE

dense_mlp + bernoulli

current binary Arabidopsis 0/1 likelihood

BCE-with-logits reconstruction

dense_mlp + categorical

future discrete genotype classes, e.g. 0/1/2

categorical cross-entropy

not recommended for the present binary Arabidopsis matrix

Controlled Arabidopsis experiment

Use the exact same C4 split/hyperparameters initially.

A. POEMS + MSE

python run_cv_resume.py \
  --dataset arabidopsis_FT10 \
  --trait_name FT10 \
  --best_params PATH_TO_C4_BEST_PARAMS.csv \
  --decoder_type poems \
  --genotype_likelihood mse \
  --dec_hidden_dim 64 \
  --out_dir FT10_C4_POEMS_MSE

B. Dense MLP + MSE

python run_cv_resume.py \
  --dataset arabidopsis_FT10 \
  --trait_name FT10 \
  --best_params PATH_TO_C4_BEST_PARAMS.csv \
  --decoder_type dense_mlp \
  --genotype_likelihood mse \
  --dec_hidden_dim 256 \
  --dec_hidden_dim2 512 \
  --out_dir FT10_C4_DENSE_MSE

C. Dense MLP + Bernoulli

python run_cv_resume.py \
  --dataset arabidopsis_FT10 \
  --trait_name FT10 \
  --best_params PATH_TO_C4_BEST_PARAMS.csv \
  --decoder_type dense_mlp \
  --genotype_likelihood bernoulli \
  --dec_hidden_dim 256 \
  --dec_hidden_dim2 512 \
  --out_dir FT10_C4_DENSE_BERNOULLI

run_cv_resume.py automatically overrides the old POEMS gate L1 coefficient
to zero for dense decoders.

Future 0/1/2 data

python run_cv_resume.py \
  --dataset future_species \
  --trait_name DTF \
  --decoder_type dense_mlp \
  --genotype_likelihood categorical \
  --n_genotype_classes 3 \
  --dec_hidden_dim 256 \
  --dec_hidden_dim2 512 \
  --out_dir future_species_dense_categorical

Architecture-specific Bayesian optimization

Do not restart BO until the controlled architecture/likelihood ablations have
shown that a new model is worth tuning.

Example dense-Bernoulli BO:

python bayes_opt_corrpenalty_calibration_aware_fixed.py \
  --dataset arabidopsis_FT10 \
  --trait_name FT10 \
  --decoder_type dense_mlp \
  --genotype_likelihood bernoulli \
  --dec_hidden_dim 256 \
  --dec_hidden_dim2 512 \
  --study_name FT10_dense_bernoulli_BO \
  --storage sqlite:///FT10_dense_bernoulli_BO.db \
  --out_csv FT10_dense_bernoulli_BO.csv

For dense decoders, decoder_l1_lambda is fixed to 0 and is not optimized.
For POEMS, it remains part of the BO search space.

IMPORTANT: the BO objective's reconstruction term is likelihood-specific.
Do not compare raw Optuna objective values between MSE and Bernoulli studies.
Use the final repeated-CV metrics and reconstruction diagnostics.

New safety / reproducibility behavior

Bernoulli runs fail early if observed SNP values are not in {0,1}.

Categorical runs fail early if genotype values are not integer classes in
[0, n_genotype_classes-1].

CV resume checks cv_run_config.json and refuses to mix incompatible runs.

Split overrides in train_POEMS() remain untouched; when explicit
train/validation/test indices are supplied, forced-trait-outlier movement is
not applied.

SNP structure inference with a supplied train_idx now fits imputation,
scaling, PCA, and KMeans on training samples and projects the remaining
samples into that basis.

Dense decoder runs do not inherit the POEMS gate-W L1 penalty.

Run names/checkpoints record decoder type, genotype likelihood, dense decoder
widths, and genotype-class count.

Decoder gradients later

For MSE:

differentiate the scalar reconstructed SNP output.

For Bernoulli:

primary geometry recommendation: differentiate decode_logits(z) for the SNP
logit direction.

probability gradients decode_probs(z) can also be reported, but they are
attenuated near probabilities 0 and 1.

For categorical:

decide explicitly whether the scientific quantity is:

expected dosage,

a specific class logit/probability,

or a contrast such as ALT-dosage direction.
