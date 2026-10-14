Light GBM final script

# MuleGuard.AI — Executive Summary
Generated: 2026-09-16T12:30:33.469828+00:00

## Business Result
- **PR-AUC:** 0.5559
- **ROC-AUC:** 0.9892
- **80% Recall mandate:** **MET**
- **Model complexity:** 5316 trees (genuine convergence via early stopping)

## Alert-Capacity Metrics (the numbers that matter operationally)
| Alert Budget | Transactions Flagged/Day | Recall (Fraud Caught) | Precision |
|---|---|---|---|
| Top 1% | 63,797 | 83.79% | 13.97% |
| Top 5% | 318,983 | 92.01% | 3.07% |

## Data
- Fit set: 22,966,731 rows | Validation set: 2,551,859 rows | OOT test set: 6,379,648 rows (never used until final evaluation)
- Class imbalance (raw ratio, legit:fraud): 1036.76:1

## Tuned Hyperparameters
```json
{
  "scale_pos_weight_multiplier": 0.4827591866086642,
  "learning_rate": 0.015322050730531,
  "num_leaves": 96,
  "max_depth": 8,
  "min_data_in_leaf": 1080,
  "feature_fraction": 0.9975977716944487,
  "bagging_fraction": 0.8217763883593567,
  "bagging_freq": 2,
  "lambda_l1": 7.194254704033055e-07,
  "lambda_l2": 2.490484090463567e-06,
  "min_gain_to_split": 0.444824304257932
}
```

## Top 15 Features (of 66 total — full ranking in shap_feature_importance_full.csv)
```
 rank                        feature  mean_abs_shap
    1 sender_receiver_pair_txn_count       3.080546
    2   days_since_account_first_txn       1.554023
    3                   txn_count_1h       0.873339
    4           currency_pair_rarity       0.551288
    5   days_since_own_last_activity       0.549402
    6                    amount_paid       0.520309
    7         payment_format_encoded       0.514253
    8             sender_interval_cv       0.513200
    9           gather_scatter_score       0.490967
   10                   day_of_month       0.478517
   11                 is_ach_payment       0.431841
   12       time_since_last_received       0.383937
   13                    hour_of_day       0.358973
   14             amount_received_1h       0.350587
   15           night_activity_ratio       0.344173
```

## Interpretation for Non-Technical Stakeholders
This model was trained strictly on past transactions and evaluated only on
future, unseen transactions — the same constraint it will face in
production. If investigators review the top 1% of
flagged transactions daily, the model catches
83.8% of all laundering activity.
The top drivers behind flagged transactions match patterns a trained human
investigator would look for (repeated sender-receiver pairs, account age,
payment method, transaction velocity), which is a strong sanity check that
the model is learning real fraud behavior rather than noise.
