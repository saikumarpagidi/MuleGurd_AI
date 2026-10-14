"""
scripts/test_live_endpoint.py
==============================
Tests the live AI Engine endpoint:
    http://localhost:8002/score

Sends:
  - 5 Real Mule transactions (from dataset, stored in Redis feature store)
  - 5 Legitimate transactions (from dataset, stored in Redis feature store)
  - 3 Brand New transactions (not in Redis, computed on-the-fly)

Prints detailed scores, SHAP explanations, and saves request/response logs.
"""

import json
import urllib.request
import pandas as pd
import redis
from pathlib import Path

ENDPOINT = "http://localhost:8002/score"

def main():
    print("=" * 100)
    print("TESTING LIVE AI ENGINE ENDPOINT: http://localhost:8002/score")
    print("=" * 100)

    # 1. Connect to Redis and raw transactions to pull exact matching historical events
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    df_chunk = pd.read_csv("ml/data/raw/HI-Medium_Trans.csv", nrows=300000)

    fmt_map = {
        "Reinvestment": "Reinvestment",
        "Cheque": "Cheque",
        "Credit Card": "Credit Card",
        "Debit Card": "Debit Card",
        "ACH": "ACH",
        "Wire": "Wire",
        "Cash": "Cash",
        "Bitcoin": "Bitcoin"
    }

    # Find 5 real mules with accounts in Redis
    mules = []
    for _, row in df_chunk[df_chunk["Is Laundering"] == 1].iterrows():
        acc = str(row["Account"])
        if r.exists(f"muleguard:features:account:{acc}"):
            mules.append(row)
            if len(mules) == 5:
                break

    # Find 5 legit with accounts in Redis
    legit = []
    for _, row in df_chunk[df_chunk["Is Laundering"] == 0].iterrows():
        acc = str(row["Account"])
        if r.exists(f"muleguard:features:account:{acc}"):
            legit.append(row)
            if len(legit) == 5:
                break

    def make_payload(row, tid):
        raw_fmt = str(row["Payment Format"])
        fmt = fmt_map.get(raw_fmt, "Other")
        ts = str(row["Timestamp"]).replace("/", "-") + ":00Z"
        ts = ts.replace(" ", "T")
        return {
            "transaction_id": tid,
            "from_account": str(row["Account"]),
            "to_account": str(row["Account.1"]),
            "amount": float(row["Amount Paid"]),
            "timestamp": ts,
            "payment_format": fmt,
            "currency": str(row["Payment Currency"])
        }

    test_cases = []

    # Add 5 Real Mules
    for i, m in enumerate(mules, 1):
        test_cases.append({
            "category": "REAL_MULE",
            "payload": make_payload(m, f"TXN-LIVE-MULE-{i:02d}"),
            "expected_fraud": True
        })

    # Add 5 Legit
    for i, l in enumerate(legit, 1):
        test_cases.append({
            "category": "LEGITIMATE",
            "payload": make_payload(l, f"TXN-LIVE-LEGIT-{i:02d}"),
            "expected_fraud": False
        })

    # Add 3 Brand New
    new_cases = [
        {
            "transaction_id": "TXN-LIVE-NEW-01",
            "from_account": "NEW_ACCOUNT_8801",
            "to_account": "MERCHANT_GROCERY_99",
            "amount": 42.50,
            "timestamp": "2026-09-08T11:20:00Z",
            "payment_format": "Debit Card",
            "currency": "USD"
        },
        {
            "transaction_id": "TXN-LIVE-NEW-02",
            "from_account": "NEW_ACCOUNT_8802",
            "to_account": "PEER_TRANSFER_77",
            "amount": 120.00,
            "timestamp": "2026-09-08T15:45:00Z",
            "payment_format": "ACH",
            "currency": "USD"
        },
        {
            "transaction_id": "TXN-LIVE-NEW-03",
            "from_account": "NEW_ACCOUNT_8803",
            "to_account": "UTILITY_ELECTRIC_66",
            "amount": 85.10,
            "timestamp": "2026-09-08T09:10:00Z",
            "payment_format": "Credit Card",
            "currency": "USD"
        },
    ]
    for c in new_cases:
        test_cases.append({
            "category": "NEW_TRANSACTION",
            "payload": c,
            "expected_fraud": False
        })

    results = []
    print(f"\nSending requests to {ENDPOINT} ...\n")

    for i, tc in enumerate(test_cases, 1):
        p = tc["payload"]
        cat = tc["category"]
        data_bytes = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[{i:02d}] FAILED: {exc}")
            continue

        lgbm = resp_data.get("lgbm_score", 0.0)
        ae = resp_data.get("ae_score", 0.0)
        recon_err = resp_data.get("ae_reconstruction_error", 0.0)
        decision = resp_data.get("decision_preliminary", "UNKNOWN")
        source = resp_data.get("feature_source", "UNKNOWN")
        latency = resp_data.get("inference_latency_ms", 0.0)
        model_ver = resp_data.get("model_versions", {})

        print(f"[{i:02d}] {cat:<16} | TxnId: {p['transaction_id']}")
        print(f"     Payload : From={p['from_account']} -> To={p['to_account']} | Amt=${p['amount']:.2f} | Format={p['payment_format']}")
        print(f"     Response: Decision={decision:<7} | LightGBM Score={lgbm:.4f} | AE Score={ae:.4f} (ReconErr={recon_err:.2f})")
        print(f"     Metadata: Source={source} | Latency={latency:.1f}ms | AE Model={model_ver.get('autoencoder')}")
        if cat == "NEW_TRANSACTION":
            print(f"     * NOTE  : AE Score is {ae:.4f} because the running ai-engine currently uses {model_ver.get('autoencoder')} (v2)!")
        print("-" * 100)

        results.append({
            "test_id": i,
            "category": cat,
            "request_payload": p,
            "response": resp_data
        })

    # Save to JSON
    out_file = Path("ml/reports/live_api_test_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll 13 live responses saved to: {out_file}")


if __name__ == "__main__":
    main()
