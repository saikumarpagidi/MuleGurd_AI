# ============================================================
# MuleGuard.AI - Local Environment Startup & Release Gate Verifier
#
# Usage:
#   .\scripts\start_local.ps1              # Start + verify
#   .\scripts\start_local.ps1 -VerifyOnly  # Verification only
#   .\scripts\start_local.ps1 -Stop        # Stop all containers
# ============================================================

param(
    [switch]$VerifyOnly,
    [switch]$Stop,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"

# -- COLORS ------------------------------------------------
function Write-Header($msg)  { Write-Host "`n$msg" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg)    { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Info($msg)    { Write-Host "  [INFO] $msg" -ForegroundColor Yellow }
function Write-Step($msg)    { Write-Host "`n>> $msg" -ForegroundColor Magenta }

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# -- STOP --------------------------------------------------
if ($Stop) {
    Write-Header "Stopping MuleGuard.AI containers..."
    Set-Location $ProjectRoot
    docker-compose down
    Write-Ok "All containers stopped. Data volumes preserved."
    exit 0
}

# -- RESET (NUCLEAR) ---------------------------------------
if ($Reset) {
    Write-Host "`n[WARNING] This will DELETE ALL DATA VOLUMES. Are you sure? (y/N)" -ForegroundColor Red
    $confirm = Read-Host
    if ($confirm -ne 'y') { Write-Info "Reset cancelled."; exit 0 }
    Set-Location $ProjectRoot
    docker-compose down -v
    Write-Ok "All containers and volumes destroyed."
    exit 0
}

# -- START -------------------------------------------------
if (-not $VerifyOnly) {
    Write-Header "MuleGuard.AI - Starting Local Environment"
    Write-Info "Project root: $ProjectRoot"
    Set-Location $ProjectRoot

    Write-Step "Starting all containers..."
    docker-compose up -d --build
    
    Write-Step "Waiting 60 seconds for services to become healthy..."
    Write-Info "Neo4j takes ~60s, Spring Boot takes ~90s on first run."
    Start-Sleep -Seconds 60
}

# -- RELEASE GATE VERIFICATION -----------------------------
Write-Header "---------------------------------------------------"
Write-Header "RELEASE GATE VERIFICATION"
Write-Header "---------------------------------------------------"

$passed = 0
$failed = 0

function Test-Gate($description, $scriptBlock) {
    try {
        $result = & $scriptBlock
        if ($result) {
            Write-Ok $description
            $script:passed++
            return $true
        } else {
            Write-Fail $description
            $script:failed++
            return $false
        }
    } catch {
        Write-Fail "$description - Error: $($_.Exception.Message)"
        $script:failed++
        return $false
    }
}

# -- Gate 1: Container Status ------------------------------
Write-Step "Gate 1: Docker Container Health"

Test-Gate "mg_postgres is healthy" {
    $status = docker inspect mg_postgres --format '{{.State.Health.Status}}' 2>&1
    $status -eq "healthy"
}

Test-Gate "mg_redis is healthy" {
    $status = docker inspect mg_redis --format '{{.State.Health.Status}}' 2>&1
    $status -eq "healthy"
}

Test-Gate "mg_kafka is healthy" {
    $status = docker inspect mg_kafka --format '{{.State.Health.Status}}' 2>&1
    $status -eq "healthy"
}

Test-Gate "mg_neo4j is healthy" {
    $status = docker inspect mg_neo4j --format '{{.State.Health.Status}}' 2>&1
    $status -eq "healthy"
}

Test-Gate "mg_flink_jobmanager is healthy" {
    $status = docker inspect mg_flink_jobmanager --format '{{.State.Health.Status}}' 2>&1
    $status -eq "healthy"
}

# -- Gate 2: Service Health Endpoints ---------------------
Write-Step "Gate 2: Microservice Health Endpoints"

$services = @(
    @{name="transaction-ingest"; port=8001; expected="transaction-ingest"},
    @{name="ai-engine";          port=8002; expected="ai-engine"},
    @{name="graph-service";      port=8003; expected="graph-service"},
    @{name="feature-service";    port=8004; expected="feature-service"}
)

foreach ($svc in $services) {
    Test-Gate "$($svc.name) /health - status:ok" {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$($svc.port)/health" -TimeoutSec 5
            $response.status -eq "ok" -and $response.service -eq $svc.expected
        } catch { $false }
    }
}

Test-Gate "scoring-orchestrator /api/v1/health - status:ok" {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/health" -TimeoutSec 10
        $response.status -eq "ok"
    } catch { $false }
}

# -- Gate 3: PostgreSQL Schema -----------------------------
Write-Step "Gate 3: PostgreSQL Schema & Immutability"

Test-Gate "PostgreSQL has 4 tables" {
    $result = docker exec mg_postgres psql -U muleguard -d muleguard -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" 2>&1
    $countStr = $result -join ""
    $countMatch = [regex]::Match($countStr, "\b4\b")
    $countMatch.Success
}

Test-Gate "audit_log INSERT succeeds" {
    $query = "INSERT INTO audit_log (event_type, entity_type, entity_id, payload) VALUES ('GATE_TEST','test','verify','{}');"
    $result = docker exec mg_postgres psql -U muleguard -d muleguard -c $query 2>&1
    ($result -join "") -match "INSERT 0 1"
}

Test-Gate "audit_log DELETE blocked (returns 0 rows)" {
    $result = docker exec mg_postgres psql -U muleguard -d muleguard -c "DELETE FROM audit_log WHERE event_type='GATE_TEST';" 2>&1
    $result -match "DELETE 0"
}

Test-Gate "audit_log UPDATE blocked (returns 0 rows)" {
    $result = docker exec mg_postgres psql -U muleguard -d muleguard -c "UPDATE audit_log SET actor='hacker' WHERE event_type='GATE_TEST';" 2>&1
    $result -match "UPDATE 0"
}

# -- Gate 4: Redis -----------------------------------------
Write-Step "Gate 4: Redis"

Test-Gate "Redis PING - PONG" {
    $result = docker exec mg_redis redis-cli PING 2>&1
    $result.Trim() -eq "PONG"
}

# -- Gate 5: Kafka Topics ----------------------------------
Write-Step "Gate 5: Kafka Topics"

$topics = @("muleguard.transactions", "muleguard.alerts", "muleguard.identity", "muleguard.model-feedback")
Test-Gate "All 4 Kafka topics exist" {
    $result = docker exec mg_kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    $allFound = $true
    $resultStr = $result -join "`n"
    foreach ($topic in $topics) {
        if ($resultStr -notmatch [regex]::Escape($topic)) { $allFound = $false }
    }
    $allFound
}

# -- Gate 6: Flink -----------------------------------------
Write-Step "Gate 6: Flink UI"

Test-Gate "Flink JobManager UI accessible" {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8081" -TimeoutSec 5 -UseBasicParsing
        $response.StatusCode -eq 200
    } catch { $false }
}

# -- SUMMARY -----------------------------------------------
Write-Header "---------------------------------------------------"
Write-Header "RELEASE GATE SUMMARY"
Write-Header "---------------------------------------------------"

Write-Host ""
Write-Host "  Passed: $passed" -ForegroundColor Green
Write-Host "  Failed: $failed" -ForegroundColor Red
Write-Host ""

if ($failed -eq 0) {
    Write-Host "  ALL GATES PASSED. Sprint 1 is complete!" -ForegroundColor Green
    Write-Host "  You can now proceed to Sprint 2." -ForegroundColor Cyan
} else {
    Write-Host "  $failed gate(s) FAILED. Fix them before proceeding to Sprint 2." -ForegroundColor Red
    Write-Host "  Run this script again after fixes: .\scripts\start_local.ps1 -VerifyOnly" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Dashboard URLs:" -ForegroundColor Cyan
Write-Host "    Kafka UI:    http://localhost:8090"
Write-Host "    Neo4j:       http://localhost:7474  (neo4j / muleguard_neo4j)"
Write-Host "    Flink:       http://localhost:8081"
Write-Host "    Grafana:     http://localhost:3001  (admin / admin_local)"
Write-Host "    Prometheus:  http://localhost:9090"
