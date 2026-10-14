"""
scripts/mock_redis_features.py
================================
Utility script to mock Redis feature data for testing the AI Engine's 
ability to retrieve historical features directly from the Feature Store
(instead of relying on caller overrides).

Usage:
  # Ensure Redis is running locally: docker run -p 6379:6379 -d redis
  .venv\Scripts\python.exe scripts/mock_redis_features.py
"""

import sys

try:
    import redis
except ImportError:
    print("Error: 'redis' package not installed. Run: pip install redis")
    sys.exit(1)

def main():
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    try:
        r.ping()
        print("✅ Connected to Redis on localhost:6379")
    except redis.exceptions.ConnectionError:
        print("❌ Could not connect to Redis. Is it running? (docker run -p 6379:6379 -d redis)")
        sys.exit(1)

    account_id = "ACC_mule_suspect"
    redis_key = f"muleguard:features:account:{account_id}"

    # Features simulating a highly active mule account
    mule_features = {
        "sender_avg_amount": "9200.0",
        "txn_count_24h": "45",
        "fan_out_24h": "40",
        "amount_vs_sender_mean": "1.05",
        "sender_lifetime_txn_count": "50",
        "sender_log_past_transaction_count": "3.9",
        "sender_log_amount_vs_past_mean": "0.05",
        "sender_amount_sum_24h": "414000.0",
        "sender_receiver_pair_txn_count": "0"
    }

    # Write to Redis
    r.hset(redis_key, mapping=mule_features)
    
    print(f"✅ Injected mock mule features into Redis key: {redis_key}")
    print("\nYou can now test the AI Engine with:")
    print("""
$body = '{
  "transaction_id": "TEST-MULE-REDIS-001",
  "from_account": "ACC_mule_suspect",
  "to_account": "ACC_receiver",
  "amount": 9500.0,
  "timestamp": "2026-08-24T03:15:00Z",
  "payment_format": "ACH",
  "currency": "USD"
}'
Invoke-RestMethod -Uri "http://localhost:8002/score" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
    """)
    print("The response should show `feature_source: redis` and a high risk score.")

if __name__ == "__main__":
    main()
