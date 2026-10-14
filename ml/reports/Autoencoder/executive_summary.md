# MuleGuard.AI — Autoencoder v6 Executive Summary

## Isolation Status
- **v6 model**: `ml/models/autoencoder_v6/autoencoder_v6.pt` (staging)
- **v2 model**: `ml/models/autoencoder/autoencoder_oot_v2.pt` (production — untouched)

## Business Result
- **OOT ROC-AUC:** 0.6717
- **OOT PR-AUC:** 0.008543  (random baseline 0.001668)
- **PR-AUC lift over random:** 5.1x
- **Scoring method:** per-feature validation-normalized anomaly score

Comparison with raw unweighted MSE score: ROC-AUC=0.6989, PR-AUC=0.009833

## Architecture (v6 upgrades)
- **Input → Hidden → Bottleneck:** 21 → 128 → 64 → **12** → 64 → 128 → 21
- **Compression ratio:** 1.75x (21 → 12 bottleneck enforcing genuine latent compression)
- **Normalization:** LayerNorm (replaces BatchNorm — stable at batch_size=1, safe at inference)
- **Regularization:** Dropout(0.15) on encoder
- **Feature source:** `features_autoencoder.parquet` (21 pre-engineered AML features)

## Cold-Start Guard (fixes score=1 on new transactions)
- **Cold-start rows in OOT test:** 883,412 (13.85%)
- **Cold-start fraud rows:** 3,838
- **Action:** These rows are assigned ae_score=0.0 (not flagged by AE alone)

## Data
- Train rows (legit only):      22,944,867
- Validation rows (legit only): 2,549,133
- OOT test rows (all labels):   6,379,648  (10,640 fraud, 0.1668%)
- Split: last 20% of chronologically-sorted rows (identical to LightGBM OOT split)

## Alert-Capacity Metrics
| Alert Budget | Flagged | Recall | Precision | Lift over random |
|---|---|---|---|---|
| Top 0.1% | 6,380 | 3.37% | 5.63% | 33.7x |
| Top 0.5% | 31,899 | 9.45% | 3.15% | 18.9x |
| Top 1.0% | 63,797 | 14.23% | 2.37% | 14.2x |
| Top 5.0% | 318,983 | 19.83% | 0.66% | 4.0x |

## Feature Attribution (Top 10 of 21 — full ranking in feature_attribution.csv)
Mean squared reconstruction error per feature in scaled feature space.
Higher = the model finds this feature harder to reconstruct = stronger anomaly signal.

```
 1. fan_in_24h                          1.713061
 2. amount_received_1h                  0.365645
 3. time_since_last_received            0.104117
 4. amount_ratio                        0.035870
 5. receiver_txn_count_1h               0.016112
 6. txn_count_1h                        0.007855
 7. account_degree_ratio                0.007442
 8. fan_out_24h                         0.006650
 9. amount_sent_1h                      0.006570
10. hour_of_day_norm                    0.006335
```

## Next Steps
1. 38 AE signals are already extracted in `features_ae_signals_v6.parquet`.
2. Run `train_lgbm_aml.py` with AE signals merged (Phase 2).
3. Promote v6 model to production by copying `ml/models/autoencoder_v6/` → `ml/models/autoencoder/`.

## Model Governance Note
This model is trained on legitimate transactions only and never shown confirmed fraud during
training. It complements the supervised LightGBM model by flagging structurally unusual
transactions. Both models evaluate on the identical OOT test window for valid comparison.
