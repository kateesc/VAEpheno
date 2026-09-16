POEMS + Bernoulli patch

Files that actually required logic changes:

models.py

allows poems+bernoulli

POEMS raw scalar output is treated as Bernoulli logit eta_j(z)

POEMS+MSE keeps the old [-10,10] clamp exactly

POEMS+Bernoulli leaves logits unclamped and uses sigmoid only for x_hat

run_cv_resume.py

allows ("poems", "bernoulli")

main.py

allows ("poems", "bernoulli") for single runs

bayes_opt_corrpenalty_calibration_aware_fixed.py

allows POEMS+Bernoulli for a future architecture-specific BO study

No substantive changes were needed in train.py:

it already validates binary data for Bernoulli

it already dispatches to masked BCEWithLogitsLoss

it already retains model.decoder_regularization()

POEMS therefore keeps its W-gate L1 penalty

No changes were needed in evaluation.py or util.py.

Recommended first POEMS+Bernoulli run:
python all_models/POEMS/run_cv_resume.py 
--dataset arabidopsis_FT10 
--trait_name FT10 
--best_params optuna_arabidopsis_FT10_balanced_trait_total_trials.csv 
--decoder_type poems 
--genotype_likelihood bernoulli 
--dec_hidden_dim 64 
--batch_size 16 
--epochs 150 
--patience 20 
--cv_strategy random 
--n_splits 5 
--n_repeats 3 
--ensemble_seeds 1 
--seed 21 
--out_dir FT10_C4_POEMS_Bernoulli

For this controlled first comparison, retain the locked C4 POEMS L1 value from Trial 22.
If POEMS+Bernoulli later gets its own BO, then tune the POEMS L1 within that Bernoulli-specific study.
