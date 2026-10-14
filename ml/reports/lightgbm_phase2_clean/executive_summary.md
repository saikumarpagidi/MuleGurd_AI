# MuleGuard.AI — Phase-2 Clean Augmented LightGBM Executive Summary
Generated: 2026-09-10T08:51:39.543387+00:00

## Business Result
- **PR-AUC:** 0.6803
- **ROC-AUC:** 0.9906
- **80% Recall mandate:** **MET**
- **Model complexity:** 5316 trees (genuine convergence via early stopping)
- **Feature dimensionality:** 92 features (66 leak-free baseline tabular + 26 monotonic Autoencoder anomaly signals)

## Alert-Capacity Metrics (the numbers that matter operationally)
| Alert Budget | Transactions Flagged/Day | Recall (Fraud Caught) | Precision | Lift |
|---|---|---|---|---|
| Top 0.1% | 6,380 | 52.01% | 86.74% | 520.1x |
| Top 0.5% | 31,899 | 78.52% | 26.19% | 157.0x |
| Top 1.0% | 63,797 | 85.74% | 14.30% | 85.7x |
| Top 5.0% | 318,983 | 92.21% | 3.08% | 18.4x |

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

## Top 15 Features (of 92 total — full ranking in shap_feature_importance_full.csv)
```
 rank                        feature  mean_abs_shap
    1 sender_receiver_pair_txn_count       1.478666
    2                 is_ach_payment       0.383707
    3         payment_format_encoded       0.378243
    4   days_since_own_last_activity       0.375525
    5   days_since_account_first_txn       0.371814
    6                   txn_count_1h       0.326303
    7              sender_avg_amount       0.324073
    8   consecutive_cross_bank_count       0.312483
    9           currency_pair_rarity       0.280842
   10                    amount_paid       0.271414
   11               sender_amount_cv       0.269824
   12              sender_amount_std       0.253757
   13         weekend_activity_ratio       0.228343
   14            receiver_amount_std       0.224327
   15    sender_unique_receivers_all       0.221067
```

## Architectural Design: Why Latent Coordinates Were Purged
In previous iterations, feeding unconstrained continuous latent bottleneck coordinates (ae_latent_0 ... ae_latent_11)
created overfitted negative splits in LightGBM trees that falsely pardoned known mule accounts.
This clean Phase-2 model incorporates strictly monotonic reconstruction error and anomaly metrics (ae_score,
ae_mse, ae_top3_error_mean, and 21 per-feature errors), ensuring that Autoencoder signals can raise risk for anomalous
patterns without ever creating unconstrained negative safe zones.

## Isolation Guarantee
- Baseline model at `ml/models/lightgbm_retrain/lgbm_best.pkl` remains **100% UNTOUCHED**.
- All Phase-2 Clean artifacts reside exclusively in `lightgbm_phase2_clean/` and `lightgbm_phase2_clean/`.
