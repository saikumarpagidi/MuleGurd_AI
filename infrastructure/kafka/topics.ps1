# ============================================================
# MuleGuard.AI - Kafka Topic Creation (Windows PowerShell)
#
# Usage: .\infrastructure\kafka\topics.ps1
# Prerequisite: mg_kafka container must be healthy
# ============================================================

$KafkaContainer = "mg_kafka"
$Bootstrap      = "localhost:9092"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "MuleGuard.AI - Creating Kafka Topics" -ForegroundColor Cyan
Write-Host "Bootstrap: $Bootstrap" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "`nWaiting for Kafka broker..." -ForegroundColor Yellow

$maxAttempts = 30
$attempt     = 0
$ready       = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    $attempt++
    try {
        docker exec $KafkaContainer kafka-broker-api-versions --bootstrap-server $Bootstrap 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Kafka is ready." -ForegroundColor Green
            $ready = $true
        }
    } catch {
    }

    if (-not $ready) {
        Write-Host "  Attempt $attempt/$maxAttempts - retrying in 3s..."
        Start-Sleep -Seconds 3
    }
}

if (-not $ready) {
    Write-Host "ERROR: Kafka not reachable after $maxAttempts attempts." -ForegroundColor Red
    exit 1
}

Write-Host "`nCreating topics..." -ForegroundColor Yellow

# Transaction events - 12 partitions for parallel consumption by the scoring orchestrator
docker exec $KafkaContainer kafka-topics --create --bootstrap-server $Bootstrap --topic muleguard.transactions --partitions 12 --replication-factor 1 --config retention.ms=604800000 --config max.message.bytes=1048576 --if-not-exists
Write-Host "  OK muleguard.transactions (12 partitions)" -ForegroundColor Green

# REVIEW and BLOCK decisions trigger alert events
docker exec $KafkaContainer kafka-topics --create --bootstrap-server $Bootstrap --topic muleguard.alerts --partitions 6 --replication-factor 1 --config retention.ms=2592000000 --if-not-exists
Write-Host "  OK muleguard.alerts (6 partitions)" -ForegroundColor Green

# Identity and device fingerprint correlation events
docker exec $KafkaContainer kafka-topics --create --bootstrap-server $Bootstrap --topic muleguard.identity --partitions 6 --replication-factor 1 --config retention.ms=604800000 --if-not-exists
Write-Host "  OK muleguard.identity (6 partitions)" -ForegroundColor Green

# Analyst feedback consumed by the model retraining pipeline
docker exec $KafkaContainer kafka-topics --create --bootstrap-server $Bootstrap --topic muleguard.model-feedback --partitions 3 --replication-factor 1 --config retention.ms=7776000000 --if-not-exists
Write-Host "  OK muleguard.model-feedback (3 partitions)" -ForegroundColor Green

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "Verifying topics..." -ForegroundColor Yellow
docker exec $KafkaContainer kafka-topics --list --bootstrap-server $Bootstrap

Write-Host "`nDone. All 4 topics created." -ForegroundColor Green
Write-Host "Inspect at: http://localhost:8090" -ForegroundColor Cyan
