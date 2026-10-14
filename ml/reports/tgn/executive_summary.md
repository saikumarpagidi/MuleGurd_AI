# MuleGuard.AI — Temporal Graph Network Executive Summary

## Business Result
- **OOT ROC-AUC:** 0.9872
- **OOT PR-AUC:** 0.512391 (vs. random baseline 0.001668)
- **PR-AUC lift over random:** 307.2x
- **Reported validation AUC at training time:** 0.984903682477314

## Data
- OOT test rows (untouched, full population -- not a reservoir sample): 6,379,648 (10,642 fraud, 0.1668%)
- Split methodology: last 20% of chronologically-sorted rows, row-fraction split (matches train_lgbm_aml.py's OOT_SPLIT_FRACTION exactly, so this model's test population is the SAME population LightGBM and the Autoencoder are evaluated on)

## Alert-Capacity Metrics
| Alert Budget | Transactions Flagged | Recall | Precision | Lift over random |
|---|---|---|---|---|
| Top 0.1% | 6,380 | 40.49% | 67.54% | 404.9x |
| Top 0.5% | 31,899 | 68.05% | 22.70% | 136.1x |
| Top 1.0% | 63,797 | 78.45% | 13.09% | 78.5x |
| Top 5.0% | 318,983 | 91.09% | 3.04% | 18.2x |

## Edge-Feature Ablation
Not run for this evaluation. To produce the documented evidence required for
the Sprint 9 validation report and Sprint 11 explainability documentation,
train a second checkpoint with `--ablate-edge-features` and re-run this
script with `--ablation-model-path` pointing to it.

## Interpretation for Non-Technical Stakeholders
This model reasons about relationships between accounts over time, not just
a single transaction's own attributes -- it maintains a running memory for
every account and updates it as new transactions arrive, then scores a new
transaction using both accounts' current memory plus the graph structure
around them. Because it is evaluated on the identical held-out time window
as LightGBM and the Autoencoder, its output can be safely combined with
theirs (e.g. flag a transaction if any model scores it highly) without
double-counting or mismatched test populations.

**Caveat on the role-classification numbers:** this model also produces a
per-account role label (mule / bridge / hub / none), but that head is not
part of the trained loss and is currently checked only against a heuristic
pseudo-label, not investigator-validated ground truth. Do not use the
role-accuracy number as a production quality gate until real labels are
available.
