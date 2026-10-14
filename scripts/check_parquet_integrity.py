"""
Diagnostic: checks features_lgbm.parquet for corruption, row-group by
row-group, so we know if it's fully unusable or just partially damaged.
Run this BEFORE deciding whether to regenerate the whole file.

Usage: python check_parquet_integrity.py
"""
import pyarrow.parquet as pq
from pathlib import Path

path = Path("ml/data/processed/features_lgbm.parquet")

print(f"File: {path}")
print(f"File size on disk: {path.stat().st_size / 1e9:.2f} GB")

pf = pq.ParquetFile(path)
print(f"Metadata OK. Row groups: {pf.num_row_groups}, Total rows (per metadata): {pf.metadata.num_rows:,}")

bad_groups = []
for i in range(pf.num_row_groups):
    try:
        pf.read_row_group(i)
        print(f"  Row group {i}: OK")
    except Exception as e:
        print(f"  Row group {i}: CORRUPT -- {e}")
        bad_groups.append(i)

print()
if not bad_groups:
    print("No corruption found at the row-group level (odd, given the earlier "
          "full-file read failure — try re-running the training script once "
          "more; it may have been a transient read, e.g. antivirus lock).")
else:
    print(f"CONFIRMED: {len(bad_groups)} of {pf.num_row_groups} row groups are corrupt: {bad_groups}")
    print("Recommendation: regenerate features_lgbm.parquet from Step 4 "
          "(feature engineering) rather than trying to patch this file.")
