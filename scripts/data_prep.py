import os
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_transactions(raw_path: str, processed_path: str):
    logger.info(f"Loading raw data from {raw_path}...")
    
    # Read the dataset
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} rows.")
    
    # 1. Rename columns
    column_map = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount_received_raw",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid_raw",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }
    df = df.rename(columns=column_map)
    logger.info("Renamed columns.")
    
    # 2. Parse timestamps to UTC
    # Format appears to be YYYY/MM/DD HH:MM
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M").dt.tz_localize("UTC")
    df = df.sort_values("timestamp")
    logger.info("Parsed and sorted timestamps (UTC).")
    
    # 3. Currency Normalization to USD
    # Rates sourced from September 2022 averages to match the dataset timeframe.
    # EUR/USD was near parity; GBP crashed due to the UK mini-budget crisis;
    # BTC traded in the $19-20K range post-Terra/Luna collapse.
    CURRENCY_TO_USD = {
        "US Dollar": 1.0,
        "Euro": 1.00,
        "UK Pound": 1.12,
        "Bitcoin": 19500.0,
        "Swiss Franc": 1.03,
        "Yen": 0.0069,
        "Yuan": 0.14,
        "Ruble": 0.016,
        "Australian Dollar": 0.65,
        "Canadian Dollar": 0.74,
        "Saudi Riyal": 0.27,
        "Mexican Peso": 0.05,
        "Brazil Real": 0.19,
        "Rupee": 0.012,
        "Shekel": 0.29,
    }
    
    # Use fallback 1.0 if currency not found (though all should match)
    df["amount_paid"] = df["amount_paid_raw"] * df["payment_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    df["amount_received"] = df["amount_received_raw"] * df["receiving_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    
    # Drop raw amounts as they are now normalized
    df = df.drop(columns=["amount_paid_raw", "amount_received_raw"])
    logger.info("Normalized amounts to USD.")

    # Check for unmapped currencies that silently fell back to 1.0
    paid_unmapped = set(df["payment_currency"].unique()) - set(CURRENCY_TO_USD.keys())
    recv_unmapped = set(df["receiving_currency"].unique()) - set(CURRENCY_TO_USD.keys())
    if paid_unmapped:
        logger.warning(f"Unmapped payment currencies (treated as 1:1 USD): {paid_unmapped}")
    if recv_unmapped:
        logger.warning(f"Unmapped receiving currencies (treated as 1:1 USD): {recv_unmapped}")

    # 4. Data Validation Gate
    # Enterprise pipelines must validate data integrity between transform and load.
    # These assertions halt the pipeline early if upstream data is corrupted.
    assert df["timestamp"].isna().sum() == 0, "Null timestamps detected after parsing"
    assert df["amount_paid"].min() >= 0, "Negative paid amounts detected after normalization"
    assert df["amount_received"].min() >= 0, "Negative received amounts detected after normalization"
    assert df["is_laundering"].isin([0, 1]).all(), "Invalid label values — expected 0 or 1"
    assert len(paid_unmapped) == 0, f"Unmapped payment currencies found: {paid_unmapped}"
    assert len(recv_unmapped) == 0, f"Unmapped receiving currencies found: {recv_unmapped}"

    fraud_rate = df["is_laundering"].mean()
    logger.info(
        f"Validation PASSED — Rows: {len(df):,} | "
        f"Fraud rate: {fraud_rate:.4%} | "
        f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}"
    )

    # 5. Save to Parquet
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / "ml" / "data" / "raw" / "HI-Small_Trans.csv"
    processed_file = project_root / "ml" / "data" / "processed" / "transactions_clean.parquet"
    
    if not raw_file.exists():
        logger.error(f"Raw data file not found: {raw_file}")
    else:
        clean_transactions(str(raw_file), str(processed_file))
