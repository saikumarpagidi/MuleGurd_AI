#!/bin/bash
# ============================================================
# MuleGuard.AI — Kafka Topic Creation (Linux / macOS / WSL2)
#
# Usage: bash infrastructure/kafka/topics.sh
# Prerequisite: mg_kafka container must be healthy
# ============================================================

set -e

KAFKA_CONTAINER="mg_kafka"
BOOTSTRAP="localhost:9092"

echo "================================================"
echo "MuleGuard.AI — Creating Kafka Topics"
echo "Bootstrap: ${BOOTSTRAP}"
echo "================================================"

echo "Waiting for Kafka broker..."
for i in $(seq 1 30); do
  if docker exec ${KAFKA_CONTAINER} kafka-broker-api-versions \
       --bootstrap-server ${BOOTSTRAP} &>/dev/null; then
    echo "Kafka is ready."
    break
  fi
  echo "  Attempt $i/30 — retrying in 3s..."
  sleep 3
done

echo ""
echo "Creating topics..."

# Transaction events — 12 partitions for parallel consumption
docker exec ${KAFKA_CONTAINER} kafka-topics \
  --create --bootstrap-server ${BOOTSTRAP} \
  --topic muleguard.transactions \
  --partitions 12 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config max.message.bytes=1048576 \
  --if-not-exists
echo "  OK muleguard.transactions (12 partitions)"

# Scoring alerts — REVIEW and BLOCK decisions
docker exec ${KAFKA_CONTAINER} kafka-topics \
  --create --bootstrap-server ${BOOTSTRAP} \
  --topic muleguard.alerts \
  --partitions 6 --replication-factor 1 \
  --config retention.ms=2592000000 \
  --if-not-exists
echo "  OK muleguard.alerts (6 partitions)"

# Identity and device fingerprint events
docker exec ${KAFKA_CONTAINER} kafka-topics \
  --create --bootstrap-server ${BOOTSTRAP} \
  --topic muleguard.identity \
  --partitions 6 --replication-factor 1 \
  --config retention.ms=604800000 \
  --if-not-exists
echo "  OK muleguard.identity (6 partitions)"

# Analyst feedback for model retraining pipeline
docker exec ${KAFKA_CONTAINER} kafka-topics \
  --create --bootstrap-server ${BOOTSTRAP} \
  --topic muleguard.model-feedback \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=7776000000 \
  --if-not-exists
echo "  OK muleguard.model-feedback (3 partitions)"

echo ""
echo "================================================"
echo "Verifying topics..."
docker exec ${KAFKA_CONTAINER} kafka-topics --list --bootstrap-server ${BOOTSTRAP}

echo ""
echo "Done. All 4 topics created."
echo "Inspect at: http://localhost:8090"
