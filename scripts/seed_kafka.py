"""
MuleGuard.AI — Data Replayer

Publishes synthetic transaction events to Kafka for local testing and load simulation.
The 5% default fraud rate mimics realistic mule transaction patterns:
  - Dormancy burst (round large amounts)
  - Fan-in topology (money converging through a ring)
  - Night-time transfers (3 AM)
  - Reused device fingerprints (bot scripts)

Usage:
    python scripts/seed_kafka.py --count 100
    python scripts/seed_kafka.py --count 1000 --broker localhost:9092
"""

import json
import uuid
import random
import hashlib
import argparse
import sys
from datetime import datetime, timezone, timedelta

TOPIC = "muleguard.transactions"
CHANNELS = ["mobile_app", "web", "atm", "upi_qr", "pos", "net_banking"]

# Mule ring accounts (deterministic tokens for testing)
RING_ACCOUNTS = [
    hashlib.sha256(f"ring_account_{i}".encode()).hexdigest()
    for i in range(5)
]


def generate_token(seed: str) -> str:
    """Generate a deterministic SHA-256 token from a seed string.

    In production, tokenisation is performed at the bank perimeter.
    Raw account numbers never enter MuleGuard.AI.
    """
    return hashlib.sha256(seed.encode()).hexdigest()


def generate_normal_transaction(index: int) -> dict:
    """Normal transaction — legitimate payment pattern."""
    amount = round(random.uniform(50, 5000), 2)
    ts = datetime.now(timezone.utc) - timedelta(seconds=random.randint(0, 3600))
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:16]}",
        "source_account_token": generate_token(f"normal_src_{index % 2000}"),
        "destination_account_token": generate_token(f"normal_dst_{random.randint(0, 10000)}"),
        "amount": amount,
        "channel": random.choice(CHANNELS),
        "timestamp": ts.isoformat(),
        "ja3_fingerprint": hashlib.md5(f"device_{index % 500}".encode()).hexdigest(),
        "_synthetic_label": "normal"
    }


def generate_mule_transaction(index: int) -> dict:
    """Mule transaction — exhibits characteristic patterns:
    - Dormancy burst (account dormant, then large sudden inflow)
    - Fan-in pattern (multiple sources → single account)
    - Round large amounts
    - Night-time transfers
    """
    amount = float(random.choice([10000, 25000, 50000, 100000, 200000]))
    # Simulate 3AM transfer (suspicious time)
    ts = datetime.now(timezone.utc).replace(hour=3, minute=random.randint(0, 59))
    
    # Use ring accounts to simulate money flowing through the ring
    ring_idx = index % len(RING_ACCOUNTS)
    next_ring_idx = (ring_idx + 1) % len(RING_ACCOUNTS)
    
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:16]}",
        "source_account_token": RING_ACCOUNTS[ring_idx],
        "destination_account_token": RING_ACCOUNTS[next_ring_idx],
        "amount": amount,
        "channel": random.choice(["upi_qr", "atm"]),
        "timestamp": ts.isoformat(),
        "ja3_fingerprint": hashlib.md5("bot_script_001".encode()).hexdigest(),
        "_synthetic_label": "mule"
    }


def generate_transaction(index: int, fraud_rate: float = 0.05) -> dict:
    """Generate a transaction — 5% are simulated mule transactions."""
    if random.random() < fraud_rate:
        return generate_mule_transaction(index)
    return generate_normal_transaction(index)


def main():
    parser = argparse.ArgumentParser(
        description="MuleGuard.AI — Kafka Data Replayer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_kafka.py --count 100
  python scripts/seed_kafka.py --count 1000 --broker kafka:29092 --fraud-rate 0.1
        """
    )
    parser.add_argument("--count", type=int, default=100,
                        help="Number of events to publish (default: 100)")
    parser.add_argument("--broker", type=str, default="localhost:9092",
                        help="Kafka bootstrap server (default: localhost:9092)")
    parser.add_argument("--fraud-rate", type=float, default=0.05,
                        help="Fraction of mule transactions (default: 0.05 = 5%%)")
    parser.add_argument("--topic", type=str, default=TOPIC,
                        help=f"Kafka topic (default: {TOPIC})")
    args = parser.parse_args()

    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
    except ImportError:
        print("ERROR: kafka-python not installed.")
        print("Install with: pip install kafka-python")
        sys.exit(1)

    print(f"MuleGuard.AI - Kafka Data Replayer")
    print(f"  Broker:    {args.broker}")
    print(f"  Topic:     {args.topic}")
    print(f"  Events:    {args.count}")
    print(f"  Fraud Rate: {args.fraud_rate * 100:.1f}%")
    print()

    try:
        producer = KafkaProducer(
            bootstrap_servers=args.broker,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",          # Wait for all replicas (durability)
            retries=5,
            request_timeout_ms=30000,
        )
    except NoBrokersAvailable:
        print(f"ERROR: Cannot connect to Kafka at {args.broker}")
        print("Is Docker Compose running? Try: docker-compose ps")
        sys.exit(1)

    mule_count = 0
    normal_count = 0

    print(f"Publishing {args.count} events...")
    for i in range(args.count):
        event = generate_transaction(i, args.fraud_rate)
        label = event.pop("_synthetic_label")  # Remove test metadata before publishing
        
        if label == "mule":
            mule_count += 1
        else:
            normal_count += 1

        producer.send(
            args.topic,
            value=event,
            key=event["source_account_token"]
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{args.count}] Published - "
                  f"normal: {normal_count}, mule: {mule_count}")

    producer.flush()
    producer.close()

    print()
    print(f"Done.")
    print(f"  Total events published: {args.count}")
    print(f"  Normal transactions:    {normal_count}")
    print(f"  Mule transactions:      {mule_count}")
    print()
    print(f"Verify at: http://localhost:8090")
    print(f"  -> Navigate to Cluster: muleguard-local")
    print(f"  -> Topics -> {args.topic}")
    print(f"  -> Messages tab")


if __name__ == "__main__":
    main()
