# MuleGuard.AI — Autoencoder Final Executive Summary

## Business Result
- **OOT ROC-AUC:** 0.3005
- **OOT PR-AUC:** 0.000475 (vs. random baseline 0.000498)
- **PR-AUC lift over random:** 1.0x

## Alert-Capacity Metrics
| Alert Budget | Transactions Flagged | Recall | Precision | Lift over random |
|---|---|---|---|---|
| Top 0.1% | 6,380 | 0.25% | 0.13% | 2.5x |
| Top 0.5% | 31,899 | 0.44% | 0.04% | 0.9x |
| Top 1.0% | 63,797 | 0.66% | 0.03% | 0.7x |
| Top 2.0% | 127,593 | 1.07% | 0.03% | 0.5x |
| Top 5.0% | 318,983 | 3.81% | 0.04% | 0.8x |
| Top 10.0% | 637,965 | 8.81% | 0.04% | 0.9x |

## Feature Attribution (Top 10)
Mean squared reconstruction error per feature, normalized against the validation set.
Higher values mean the model reconstructed that feature unusually poorly.

```
1. payment_format_risk_score: 0.490492
2. hour_of_day_norm: 0.297756
3. txn_count_1h: 0.278367
4. amount_sent_1h: 0.224894
5. day_of_week_norm: 0.213976
6. txn_count_24h: 0.200775
7. is_cross_bank: 0.188296
8. log_amount_received: 0.187075
9. log_amount_paid: 0.182250
10. is_night: 0.179749
```
