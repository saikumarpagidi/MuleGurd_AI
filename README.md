# MuleGuard.AI

> **AI-Driven Mule Account Detection System**
>
> Detects mule accounts and money-muling networks in real time using three parallel AI models — LightGBM (velocity scoring), Deep Autoencoder (anomaly detection), and Temporal Graph Network (network topology analysis). Decisions are fused by the Scoring Orchestrator and persisted to an immutable audit trail.

---

## Architecture

```
Transaction Event
      │
      ▼
  transaction-ingest  ──────────────────► Kafka (muleguard.transactions)
                                                    │
                                          ┌─────────▼──────────┐
                                          │  scoring-orchestrator│
                                          │  (Spring Boot WebFlux)
                                          └──────┬──────┬───────┘
                                                 │      │
                                    ┌────────────┘      └────────────┐
                                    ▼                                ▼
                              ai-engine                      graph-service
                         (LightGBM + Autoencoder)        (TGN + Neo4j GDS)
                                    │                                │
                                    └───────────────┬────────────────┘
                                                    ▼
                                            feature-service
                                           (Feast + Redis)
                                                    │
                                                    ▼
                                     Risk Fusion → Decision
                                     ┌──────────────────────┐
                                     │  < 0.45  → APPROVE   │
                                     │  0.45–0.74 → REVIEW  │
                                     │  ≥ 0.75  → BLOCK     │
                                     └──────────────────────┘
                                                    │
                                          PostgreSQL · Audit Log
                                          Kafka · muleguard.alerts
```

**Risk fusion formula:** `Final Score = 0.50 × TGN + 0.30 × LightGBM + 0.15 × Autoencoder + 0.05 × Device`

---

## Infrastructure

| Component | Image | Port | Purpose |
|-----------|-------|------|---------|
| PostgreSQL 16 | `postgres:16-alpine` | 5432 | System of record — scoring decisions, alerts, audit log, model registry |
| Redis 7 | `redis:7-alpine` | 6379 | Online feature store cache |
| Apache Kafka | `confluentinc/cp-kafka:7.5.0` | 9092 | Central message bus |
| Apache Flink | `flink:1.18-java17` | 8081 | Real-time stream processing |
| Neo4j 5.13 | `neo4j:5.13-community` | 7474 / 7687 | Graph database — ring detection, network analysis |

---

## Services

| Service | Stack | Port | Status |
|---------|-------|------|--------|
| `transaction-ingest` | FastAPI / Python 3.11 | 8001 | Foundation skeleton |
| `ai-engine` | FastAPI / Python 3.11 | 8002 | Foundation skeleton |
| `graph-service` | FastAPI / Python 3.11 | 8003 | Foundation skeleton |
| `feature-service` | FastAPI / Python 3.11 | 8004 | Foundation skeleton |
| `scoring-orchestrator` | Spring Boot WebFlux / Java 21 | 8080 | Foundation skeleton |

---

## Developer Onboarding & Installation Guide

This guide covers the end-to-end setup for this  project. The architecture is heavily containerized to ensure strict environment parity and minimal local dependencies.

### Step 1: System Prerequisites

Ensure the following are installed and running on your host machine prior to initialization:
- **Docker Desktop** (v24.x or higher) — Must be actively running.
- **Git** — For version control operations.

### Step 2: Environment Configuration

The repository ships with an environment template. You must initialize your local configuration before booting the cluster.

```powershell
# Copy the environment template to your local environment file
cp .env.example .env
```
*Note: The default template values are pre-configured to work seamlessly for local development. Never commit your `.env` file to version control.*

### Step 3: Launching the Cluster

Start the comprehensive microservices and infrastructure stack in detached mode:

```powershell
docker-compose up -d
```
All containers are configured to initialize in strict dependency order. Core infrastructure (PostgreSQL, Redis, Kafka, Neo4j, Flink) will achieve a healthy state prior to the application services booting. Allow approximately 60–90 seconds for the initial startup sequence and image downloads.

### Step 4: System Verification

Once the boot sequence concludes, manually verify the cluster's operational status using the exposed local endpoints:

1. **Verify Microservices Health APIs:**
   Navigate to the standard health endpoints. Each must return an HTTP 200 OK status containing `{"status":"ok"}`.
   - Transaction Ingest: [http://localhost:8001/health](http://localhost:8001/health)
   - AI Engine: [http://localhost:8002/health](http://localhost:8002/health)
   - Graph Service: [http://localhost:8003/health](http://localhost:8003/health)
   - Feature Service: [http://localhost:8004/health](http://localhost:8004/health)
   - Scoring Orchestrator: [http://localhost:8080/api/v1/health](http://localhost:8080/api/v1/health)

2. **Verify Graph Database (Neo4j):**
   - Access the Neo4j Browser at **[http://localhost:7474](http://localhost:7474)**.
   - Authenticate using the default development credentials: `neo4j` / `muleguard_neo4j`.

3. **Verify Stream Processing (Apache Flink):**
   - Access the Flink Dashboard at **[http://localhost:8081](http://localhost:8081)** to confirm the processing cluster is responsive.

### Step 5: Data Seeding & Validation

To validate the end-to-end ingestion pipeline, publish a batch of synthetic transaction events directly to the Kafka cluster:

```powershell
python scripts\seed_kafka.py --count 100
```
*Validation:* Navigate to the Kafka UI at **[http://localhost:8090](http://localhost:8090)**, select **Topics** → **muleguard.transactions** → **Messages**, and verify the JSON payloads were successfully ingested into the topic.

### Cluster Management

To safely halt the environment while preserving all database volumes and cached data:
```powershell
docker-compose down
```

To execute a hard reset (destroys all persisted data volumes and yields a clean slate on the next boot):
```powershell
docker-compose down -v
```


---

## Build Progress

| Phase | Scope | Weeks | Status |
|-------|-------|-------|--------|
| **Phase 1** | Foundation — infrastructure, CI/CD, observability | 1–4 | In Progress|
| Phase 2 | Data ingestion, feature engineering, first models | 5–10 | 🔒 Locked |
| Phase 3 | Temporal GNN, full scoring pipeline | 11–14 | 🔒 Locked |
| Phase 4 | Explainability, analyst dashboard | 15–21 | 🔒 Locked |
| Phase 5 | Hardening, Kubernetes, production readiness | 22–28 | 🔒 Locked |

---



## Repository Structure

```
MuleGurd_AI/
├── docker-compose.yml           All infrastructure and service containers
├── .env                         Local secrets — never committed
├── .env.example                 Template for new developers
├── .pre-commit-config.yaml      Credential scanning and lint hooks
│
├── services/
│   ├── transaction-ingest/      FastAPI — ingestion boundary
│   ├── ai-engine/               FastAPI — LightGBM + Autoencoder
│   ├── graph-service/           FastAPI — TGN + Neo4j
│   ├── feature-service/         FastAPI — Feast + Redis
│   └── scoring-orchestrator/    Spring Boot WebFlux — risk fusion
│
├── infrastructure/
│   ├── postgres/schema.sql      Database schema with immutability rules
│   ├── kafka/topics.ps1         Topic creation script
│   ├── neo4j/seed_graph.cypher  Graph constraints and seed data
│   ├── prometheus.yml           Scrape configuration
│   ├── prometheus/alerts.yml    Alerting rules
│   └── grafana/                 Dashboard and provisioning config
│
├── scripts/
│   ├── seed_kafka.py            Synthetic event publisher
│   └── install_hooks.ps1        Developer hook setup
│
├── ml/                          Model training — populated in Phase 2
├── flink/                       Streaming jobs — populated in Phase 2
├── k8s/                         Kubernetes manifests — populated in Phase 5
└── .github/workflows/           CI/CD pipelines (ci.yml, build.yml)
```

---

*MuleGuard.AI — Built for financial crime detection at scale.*
