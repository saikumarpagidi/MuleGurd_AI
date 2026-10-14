# MuleGuard.AI — Phase-2 Augmented LightGBM Executive Summary
Generated: 2026-09-09T09:39:02.330710+00:00

## Business Result
- **PR-AUC:** 0.6917
- **ROC-AUC:** 0.9905
- **80% Recall mandate:** **MET**
- **Model complexity:** 5996 trees (genuine convergence via early stopping)
- **Feature dimensionality:** 104 features (66 baseline tabular + 38 Autoencoder latent signals)

## Alert-Capacity Metrics (the numbers that matter operationally)
| Alert Budget | Transactions Flagged/Day | Recall (Fraud Caught) | Precision | Lift |
|---|---|---|---|---|
| Top 0.1% | 6,380 | 52.68% | 87.85% | 526.8x |
| Top 0.5% | 31,899 | 79.26% | 26.44% | 158.5x |
| Top 1.0% | 63,797 | 85.63% | 14.28% | 85.6x |
| Top 5.0% | 318,983 | 92.19% | 3.08% | 18.4x |

## Data
- Fit set: 22,966,731 rows | Validation set: 2,551,859 rows | OOT test set: 6,379,648 rows (never used until final evaluation)
- Class imbalance (raw ratio, legit:fraud): 1036.76:1

## Tuned Hyperparameters
```json
{
  "objective": "binary",
  "metric": [
    "average_precision",
    "auc"
  ],
  "boosting_type": "gbdt",
  "is_unbalance": false,
  "scale_pos_weight": 70.68775502296954,
  "learning_rate": 0.013119860213858855,
  "num_leaves": 120,
  "max_depth": 10,
  "min_data_in_leaf": 1402,
  "min_child_samples": 50,
  "feature_fraction": 0.5392661283932232,
  "bagging_fraction": 0.5011908323800748,
  "bagging_freq": 1,
  "lambda_l1": 1.1009700812828425,
  "lambda_l2": 0.3048500789351395,
  "min_gain_to_split": 0.6910753690578371,
  "verbosity": -1,
  "seed": 42,
  "n_jobs": 4,
  "max_bin": 127,
  "force_row_wise": true
}
```

## Top 15 Features (of 104 total — full ranking in shap_feature_importance_full.csv)
```
 rank                        feature  mean_abs_shap
    1 sender_receiver_pair_txn_count       1.288062
    2   days_since_account_first_txn       0.480333
    3   days_since_own_last_activity       0.403955
    4                 is_ach_payment       0.373562
    5   consecutive_cross_bank_count       0.353090
    6              sender_avg_amount       0.331974
    7                   day_of_month       0.322926
    8           currency_pair_rarity       0.313560
    9                    ae_latent_0       0.303350
   10                   txn_count_1h       0.281934
   11         payment_format_encoded       0.276662
   12              sender_amount_std       0.274907
   13           amount_robust_zscore       0.269791
   14    sender_unique_receivers_all       0.237089
   15         weekend_activity_ratio       0.223762
```

## Interpretation for Non-Technical Stakeholders
This model was trained strictly on past transactions and evaluated only on
future, unseen transactions — the same constraint it will face in
production. The augmented feature set integrates both explicit transactional
graph behaviors (velocity, bursts, pair counts) and 38 deep representation
signals from Autoencoder v3 (latent embeddings and feature reconstruction errors),
providing dual-layered protection against complex money muling and dispersion syndicates.

## Staging Isolation Guarantee
- Baseline model at `ml/models/lightgbm/lgbm_best.pkl` remains **100% UNTOUCHED**.
- All Phase-2 artifacts reside exclusively in `ml/models/lightgbm_phase2/` and `ml/reports/lightgbm_phase2/`.
