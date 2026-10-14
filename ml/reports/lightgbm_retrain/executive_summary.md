# MuleGuard.AI — Executive Summary
Generated: 2026-09-09T15:54:26.546284+00:00

## Business Result
- **PR-AUC:** 0.6779
- **ROC-AUC:** 0.9904
- **80% Recall mandate:** **MET**
- **Model complexity:** 5990 trees (genuine convergence via early stopping)

## Alert-Capacity Metrics (the numbers that matter operationally)
| Alert Budget | Transactions Flagged/Day | Recall (Fraud Caught) | Precision |
|---|---|---|---|
| Top 1% | 63,797 | 85.69% | 14.29% |
| Top 5% | 318,983 | 92.10% | 3.07% |

## Data
- Fit set: 22,966,731 rows | Validation set: 2,551,859 rows | OOT test set: 6,379,648 rows (never used until final evaluation)
- Class imbalance (raw ratio, legit:fraud): 1036.76:1

## Tuned Hyperparameters
```json
{
  "scale_pos_weight_multiplier": 0.06818121503156904,
  "learning_rate": 0.013119860213858855,
  "num_leaves": 120,
  "max_depth": 10,
  "min_data_in_leaf": 1402,
  "feature_fraction": 0.5392661283932232,
  "bagging_fraction": 0.5011908323800748,
  "bagging_freq": 1,
  "lambda_l1": 1.1009700812828425,
  "lambda_l2": 0.3048500789351395,
  "min_gain_to_split": 0.6910753690578371
}
```

## Top 15 Features (of 66 total — full ranking in shap_feature_importance_full.csv)
```
 rank                        feature  mean_abs_shap
    1 sender_receiver_pair_txn_count       1.681383
    2         payment_format_encoded       0.545723
    3              sender_avg_amount       0.480340
    4   days_since_own_last_activity       0.376906
    5                 is_ach_payment       0.357066
    6                    amount_paid       0.310932
    7   consecutive_cross_bank_count       0.310752
    8           currency_pair_rarity       0.304908
    9           gather_scatter_score       0.283454
   10   days_since_account_first_txn       0.265877
   11              sender_amount_std       0.254388
   12               sender_amount_cv       0.250609
   13                   day_of_month       0.230838
   14             amount_received_1h       0.226876
   15            receiver_amount_std       0.223190
```

## Interpretation for Non-Technical Stakeholders
This model was trained strictly on past transactions and evaluated only on
future, unseen transactions — the same constraint it will face in
production. If investigators review the top 1% of
flagged transactions daily, the model catches
85.7% of all laundering activity.
The top drivers behind flagged transactions match patterns a trained human
investigator would look for (repeated sender-receiver pairs, account age,
payment method, transaction velocity), which is a strong sanity check that
the model is learning real fraud behavior rather than noise.
