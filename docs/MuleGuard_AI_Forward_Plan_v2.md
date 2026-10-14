# MuleGuard.AI — Complete Forward Plan (v2)
## From Current State → Production Launch

> **Supersedes the August 2026 Forward Plan.** This version folds in the TGN dynamic edge-feature layer (Node Profile vs. Transaction Event architecture) as a required step inside Sprint 5, not a separate workstream.
>
> **Date authored:** August 2026 | **Status:** LightGBM ✅ · Autoencoder ✅ · Neo4j/GNN/TGN 🔒 Next
 
---

## Actual Completed State (Ground Truth)

### ✅ Phase 1 — Foundation & Infrastructure (100% Done)

| Sprint | Work | Status |
|--------|------|--------|
| Sprint 1 | Docker, PostgreSQL, Neo4j, Redis, Kafka, all 5 service skeletons | ✅ Complete |
| Sprint 2 | CI/CD (GitHub Actions), Grafana/Prometheus dashboards, pre-commit hooks | ✅ Complete |

### ✅ Phase 2 — AI Core: Data, Features & First Models (85% Done)

| Step | Work | Status | Key Metric |
|------|------|--------|-----------|
| Step 1 | `scripts/data_prep.py` — raw CSV → `transactions_clean.parquet` | ✅ Done | 31,898,238 rows |
| Step 2 | `ml/notebooks/01_data_exploration.ipynb` | ✅ Done | IBM AML HI-Medium |
| Step 3 | `scripts/feature_engineering.py` — 66-feature pipeline | ✅ Done | 66 LightGBM / 21 AE |
| Step 4 | Feature generation run, verified | ✅ Done | features_lgbm.parquet |
| Step 5 | LightGBM training | ✅ Done | ROC-AUC: **0.9905** |
| Step 6 | Autoencoder training | ✅ Done | ROC-AUC: **0.4026** |

**Certified Model Metrics (LightGBM v1):**
```
ROC-AUC              : 0.9905  ← exceeds 0.93 enterprise target ✅
PR-AUC               : 0.6631
Top 1% Alert Budget  : Recall 85.67%, Precision 14.29%
Top 5% Alert Budget  : Recall 92.25%, Precision 3.08%
Top Features         : sender_receiver_pair_txn_count, payment_format_encoded, sender_avg_amount
Model file           : ml/models/lightgbm/lightgbm_model.txt + lgbm_best.pkl
```

**Certified Model Metrics (Autoencoder v2):**
```
ROC-AUC              : 0.4026  ← anomaly detector, not classifier (expected)
Calibrated p99 Threshold: 1.4918
Flagged Volume @ p99 : 75.07% of all transactions
Mule Recall @ p99    : 62.10%
Model file           : ml/models/autoencoder_v2/autoencoder_oot_v2.pt
```

> [!IMPORTANT]
> LightGBM is the primary detection engine. The Autoencoder is the unsupervised anomaly layer. Both metrics are expected and correct — this does not change in v2.

---

## What Changed in v2 (Read This First)

The original plan's Sprint 5 TGN description already assumed "edge features (amount, format, time delta)" would feed the message function, but never itemized them or gave them a build step. That gap is now closed with a new **Step 5.0**, inserted before TGN training begins. Nothing else in Phase 3–5 changes.

**The architectural distinction driving this change:**

| Layer | Source | Answers | Feeds |
|---|---|---|---|
| Neo4j GDS (PageRank, Louvain, SCC) | Graph DB, static | **"WHO is this account?"** — structural hub role, syndicate cluster, circular-loop membership | TGN memory initialization, Static GNN node features |
| TGN 13 Dynamic Edge Features | Real-time transaction stream | **"WHAT happened in this transaction?"** — amount, channel, structuring behavior, skimming ratio | TGN message function `e_uv` at every event |
| TGN Memory + Fourier Time | PyTorch, in-memory | **"WHEN and in what sequence?"** — temporal chains, velocity, cyclic flows | TGN node embeddings → risk head |

Without the edge features, GDS alone gives TGN a static photograph of the network with no financial content per edge — the model can see that A→B exists but not whether it was $5 or $5,000,000, cash or Bitcoin, a one-off or a structured $9,500 slice under the CTR limit. That is why this is a required step, not an enhancement.

---

## Redesigned Forward Plan

### Current Position on the 28-Week Timeline
```
COMPLETED ←──────────────────────────────────────── NOW ──► TO BUILD
Phase 1   Phase 2 (partial)              Phase 2  Phase 3   Phase 4  Phase 5
Sprint 1  Sprint 2  Sprint 3 Steps 1-6   Sprint 4  S5-S6     S7-S9    S10-S11
[DONE]    [DONE]    [DONE]               [NEXT]    [NEXT]    [LATER]  [LATER]
```

---

## Phase 2 Completion — Sprint 4
### Neo4j Graph Seeding, Static GNN Baseline & SHAP API
**Duration: ~2 weeks | Next immediate work**

Unchanged from v1. This is the required prerequisite for everything that follows — TGN cannot run without the seeded graph, and the Static GNN baseline is the only number that will tell you whether TGN's eventual AUC is genuinely better or just noise.

#### Step 4A — Neo4j Graph Seeding
**Not skippable.**

**What to build:** `scripts/seed_neo4j.py` (currently a stub — 69 bytes)

```
transactions_clean.parquet
  → Extract (from_account, to_account, amount, timestamp, is_laundering)
  → Bulk-load into Neo4j:
      Account nodes  (~2.5M unique accounts from IBM AML HI-Medium)
      SENT relationships (31.9M edges with amount, timestamp, is_laundering properties)
  → Run GDS algorithms:
      GDS PageRank → write pagerank property to Account nodes
      GDS Louvain → write community_id property (ring detection)
      GDS Strongly Connected Components (SCC) → write in_scc_ring property
```

**Files to create/modify:**
- `scripts/seed_neo4j.py` — Replace stub with full seeding script
- `infrastructure/neo4j/seed_graph.cypher` — Already has schema; add GDS call queries

**Done when:**
- Neo4j has Account nodes + SENT relationships (full IBM AML graph)
- `in_scc_ring = true` nodes exist (circular money flows confirmed)
- `community_id` spans 100+ Louvain communities
- All three GDS properties written back and queryable

#### Step 4B — Static GNN Baseline
**Compressible under token/time pressure — not droppable.** Run it lean: fewer epochs, smaller hidden dims, just enough for a defensible AUC number. It is cheap to train and it's your only debugging reference point once TGN is running.

**What to build:** `ml/training/static_gnn_baseline.py`

Architecture: GraphSAGE + GAT (2 layers) on the Neo4j graph using PyTorch Geometric.

```
Input node features: [pagerank, community_id, in_scc_ring, lifetime_txn_count, 
                      lifetime_total_sent_log, night_activity_ratio, fan_out_24h]
Split: 80/20 random (NOT temporal — intentionally weaker than TGN will be)
Target: is_laundering on Account nodes (aggregated from transaction labels)
```

**Goal:** AUC ≥ 0.85. This is the baseline TGN must beat by ≥ 1% in Phase 3.

**Files to create:**
- `ml/training/static_gnn_baseline.py`
- `ml/models/baseline_gnn_auc.json` — Records the static AUC for comparison

#### Step 4C — SHAP API Integration into AI Engine
Can be deferred after TGN work if token budget is tight — it has no TGN dependency.

**What to build:** Upgrade `services/ai-engine/app/main.py` from a health-endpoint skeleton to a real scoring endpoint.

```
POST /score
  Input: transaction JSON
  Process:
    1. Load features from Redis (Feast) OR compute on-the-fly from input
    2. Run LightGBM inference → lgbm_score (0-1)
    3. Run Autoencoder inference → ae_reconstruction_error → ae_anomaly_score (0-1)
    4. Compute SHAP values for this transaction
    5. Return: {lgbm_score, ae_score, shap_top_5, decision_preliminary}
```

**Files to modify:**
- `services/ai-engine/app/main.py` — Add `/score` endpoint, load model artifacts
- `services/ai-engine/requirements.txt` — Add lightgbm, torch, shap
- `services/ai-engine/Dockerfile` — Ensure model files are accessible

**Done when:**
- `POST /score` returns 5 SHAP features with RAISES_RISK / LOWERS_RISK directions
- SHAP contributions sum to approximately the lgbm_score (Shapley consistency check)
- Endpoint responds in < 200ms for a single transaction

---

## Phase 3 — Temporal GNN & Full Scoring Pipeline

### Sprint 5 — TGN Training & Temporal Evaluation
**Duration: ~2 weeks (+ Step 5.0 below is the new addition, budget ~2–3 extra days)**

> [!IMPORTANT]
> This is the core technical differentiator of MuleGuard.AI. Everything else in the project is a support structure for this component. Step 5.0 is now a hard prerequisite of this sprint — training TGN without it is architecturally incomplete relative to your own design.

#### 🆕 Step 5.0 — Build TGN Dynamic Edge Features (NEW — do this first, before `tgn_train.py`)

**Why this step exists:** The v1 plan's TGN architecture already specified "edge features (amount, format, time delta)" feeding the message function — this step is what actually produces that tensor. GDS gives TGN a static profile of *who* each account is; without per-transaction edge features, TGN has no record of *what* happened in each event, and per the architecture reference, Out-of-Time ROC-AUC is expected to degrade sharply (cited figure: ~0.9468 → below 0.72) when edge attributes are omitted, because the memory module's GRU update has nothing real to encode at `e_uv`.

**What to build:** `ml/features/build_tgn_edge_features.py`

Input: `transactions_clean.parquet` (same source as everything else — no new data needed)
Output: 13-dimensional edge feature tensor per transaction, aligned to Neo4j edge IDs, saved as `ml/features/tgn_edge_features.parquet`

**The 13 features, sequenced by build cost:**

*Tier 1 — cheap, single-row derivable (build first):*
| Feature | Derivation |
|---|---|
| `log_amount_paid` | `log(amount + 1)` |
| `payment_format_encoded` | Categorical encode of payment channel (Cash/ACH/Bitcoin/Wire/etc.) |
| `is_cross_bank` | sender bank ≠ receiver bank |
| `is_cash_payment` | payment_format == Cash |
| `is_bitcoin_transaction` | payment_format == Bitcoin/Crypto |
| `structuring_threshold_flag` | amount in $8,000–$9,999 band (CTR evasion signature) |

*Tier 2 — requires cross-row logic, but do not cut (protects the product's core detection claim):*
| Feature | Derivation |
|---|---|
| `amount_ratio` | outbound amount ÷ nearest matching inbound amount for same account within short time window |
| `amount_difference` | inbound − outbound (captures the skimmed commission, typically 2–5%) |

These two specifically detect the "transit mule" peeling-chain pattern (money in, ~95–98% out immediately) that is the flagship use case for this product. Cutting them would remove the reason TGN exists over LightGBM.

*Remaining features:* the rest of the 13 (time-delta components, velocity flags, additional channel/risk encodings) follow the same tiering logic — timestamp-only fields go in Tier 1, anything requiring a lookback window goes in Tier 2.

**Files to create:**
- `ml/features/build_tgn_edge_features.py`
- `ml/features/tgn_edge_features.parquet` (output)
- `ml/features/tgn_edge_feature_spec.json` — documents each of the 13 features and its formula, for the eventual regulatory/explainability docs in Sprint 11

**Done when:**
- All 13 features computed and joined to the 31.9M edges already seeded in Neo4j (Step 4A)
- Spot-check: a known structuring pattern in the IBM AML data shows `structuring_threshold_flag = 1`
- Spot-check: a known transit-mule chain shows `amount_ratio` clustering near 1.0 (0.95–1.0 range)

#### TGN Architecture (unchanged from v1, now with edge features properly wired)

```
Temporal Graph Network (TGN):
  Memory Module: 172-dimensional GRU hidden state per account node
    → Updated on every transaction event
    → Stores "what happened to this account historically"
  
  Message Function: Encodes each transaction as a message
    → Source memory + destination memory + edge features (the 13 from Step 5.0)
    → Passed through learned message aggregator:
         h_channel = Embedding(payment_format_encoded) ∈ ℝ8
         e_uv = Linear2( ReLU( Linear1( [x_continuous_12 ∥ h_channel] ) ) ) ∈ ℝ64
  
  Continuous Fourier Time Encoding:
         φ_k(Δt) = cos(ω_k·Δt + ψ_k),  ω_k = 10^(-4k/64),  k = 1,...,64
  
  Dynamic Node Memory Update (GRU):
         m_u(t) = GRUCell([m_u(t⁻) ∥ e_uv ∥ φ(Δt_u)], m_u(t⁻)) ∈ ℝ64
  
  Graph Attention Layers: 2× TransformerConv layers
    → Propagates updated memories through 2-hop neighborhood
    → Produces node embeddings that encode local ring topology
  
  Interaction Risk Scoring:
         z_uv = [m_u(t) ∥ m_v(t) ∥ e_uv ∥ φ(Δt)] ∈ ℝ192
         Risk Probability = σ(Linear(Dropout(ReLU(Linear(z_uv)))))
  
  Output Heads (two parallel):
    Head 1: P(is_mule) — binary classification probability
    Head 2: P(role) — softmax over [HUB, BRIDGE, MULE, NONE]
```

**Data source:** IBM AML HI-Medium transaction graph in Neo4j (Step 4A) + edge features (Step 5.0)

**Temporal split (strictly forward-looking, no leakage):**
```
Training set:   Days 1–15  (first 55% of the 27-day dataset)
Validation set: Days 16–20 (next 18%)
Test set:       Days 21–27 (last 27% — never seen during training)
```

**Loss function:** Focal Loss (α=0.25, γ=2.0) — focuses gradient on rare hard positive cases.

**Files to create:**
- `ml/training/tgn_train.py` — Full TGN training loop, consumes Step 5.0 output
- `ml/training/tgn_model.py` — TGN architecture (memory, message, attention, heads) per the formulation above
- `services/graph-service/app/main.py` — Upgrade stub to TorchServe + `/graph-score` endpoint

**🆕 Added validation — Edge Feature Ablation Check:**
Train TGN once more with the 13 edge features zeroed out (memory + time encoding untouched). This is one extra cheap training run and gives you a real, defensible number for the demo and validation report instead of an assumed collapse figure.

- `ml/models/tgn_ablation_no_edge_features.json` — records AUC with edges zeroed

**Done when:**
- `ml/models/tgn_v1_best.pt` exists (≥ 60 MB)
- TGN temporal test AUC ≥ 0.97
- TGN AUC > Static GNN AUC by ≥ 1 percentage point
- Ablation run confirms AUC drops materially with edge features zeroed (documents *why* Step 5.0 mattered)
- Ring role accuracy ≥ 0.75 (HUB/BRIDGE/MULE/NONE classification)
- `/graph-score` responds in < 25ms per transaction

---

### Sprint 6 — Full Scoring Orchestrator & Latency Gate
**Duration: ~2 weeks** — unchanged from v1.

#### Risk Fusion Architecture

```
Kafka Event (muleguard.transactions)
  ↓
scoring-orchestrator (Spring Boot WebFlux)
  ↓
  ┌─────────────────────────────────────────────────┐
  │  Reactor Mono.zip() — all three calls PARALLEL  │
  │                                                  │
  │  ├── Redis feature fetch (2ms)                   │
  │  │     └── ai-engine POST /score                 │
  │  │           ├── LightGBM inference (< 5ms)      │
  │  │           └── Autoencoder inference (< 10ms)  │
  │  │                                               │
  │  └── graph-service POST /graph-score             │
  │        └── TGN inference (< 25ms)                │
  └─────────────────────────────────────────────────┘
  ↓
Risk Fusion (weighted average):
  Final Score = 0.50 × TGN + 0.30 × LightGBM + 0.15 × Autoencoder + 0.05 × Device
  ↓
Decision engine:
  Score < 0.45  → APPROVE
  Score 0.45–0.74 → REVIEW (alert to investigator queue)
  Score ≥ 0.75  → BLOCK  (real-time rejection)
  ↓
  ├── Write to PostgreSQL scoring_decisions
  └── Publish to Kafka muleguard.alerts
```

**Files to modify:**
- `services/scoring-orchestrator/src/.../ScoringService.java` — Core fusion logic
- `services/scoring-orchestrator/src/.../KafkaConsumerService.java` — Event ingestion

**Done when:**
- High-risk node returns `{decision: BLOCK, risk_score ≥ 0.75, ring_role: MULE}`
- Low-risk node returns `{decision: APPROVE, risk_score ≤ 0.35}`
- k6 load test: P95 < 50ms at 100 concurrent users
- LightGBM + TGN calls have overlapping timestamps in logs (true parallel execution)
- Killing `graph-service` → orchestrator continues in `degraded_mode` using LightGBM + Autoencoder

---

## Phase 4 — Explainability, Validation & Demo Readiness
*(unchanged from v1)*

### Sprint 7 — Explainability Engine & LLM Narration
**Duration: ~2 weeks**

**Primary path (Gemini Flash):**
```
"Transaction {ID} received a BLOCK decision (risk_score=0.89).
Top signals:
  - is_ach_payment=1 (RAISES_RISK, +0.24 contribution)
  - time_since_last_received=0h (RAISES_RISK, +0.30 contribution)
  - sender_receiver_pair_txn_count=1 (RAISES_RISK, +0.19 contribution)
Graph evidence: account is part of Ring #4821 (community_id), in_scc_ring=true,
  pagerank=0.82 (top 2% of all accounts).
Produce a 2-sentence investigator alert in plain English."
```

**Fallback path (Rule-Based, < 50ms):**
```
"[ALERT] High-risk ACH transfer detected. Account {from_account} shows 
{fan_out_24h} outbound transfers in 24h, 9-day dormancy gap before this burst,
and membership in a flagged transaction ring (community #{community_id})."
```

**Files to create:**
- `services/explainability-service/` — New FastAPI service (port 8005)
- `docker-compose.yml` — Add explainability-service container

**Done when:**
- High-risk scoring returns explanation mentioning ring membership and dormancy
- Blocking Gemini API still returns explanation with `explanation_mode: fallback` in < 100ms
- 100% of test cases in a 50-fraud / 50-legit quality suite produce valid text

---

### Sprint 8 — Investigator Dashboard & Scoring Simulator
**Duration: ~2 weeks**

| Panel | Technology | What It Shows |
|-------|-----------|---------------|
| Risk Fusion Gauge | D3.js arc gauge | TGN / LightGBM / AE component scores + fused final |
| SHAP Waterfall Chart | D3.js bar chart | Top 10 features, direction (red=raises, blue=lowers) |
| TGN Graph Explorer | D3.js force-directed | 2-hop transaction network for the flagged account |
| Live Scoring Simulator | React form | Submit any transaction JSON → see full score in real time |
| Alert Log | React table | Paginated REVIEW/BLOCK decisions, sortable/filterable |
| Model Metrics | React cards | Live AUC, confusion matrix on last 1,000 scored |

**Files to create:**
- `dashboard/` — Next.js 14 app (replaces placeholder)
- `docker-compose.yml` — Add dashboard container (port 3000)

**Done when:**
- Dashboard loads within 2 seconds
- SHAP waterfall renders all 10 features correctly
- TGN Graph Explorer renders the 2-hop network for a submitted fraud account
- Scoring Simulator returns all three model scores within 3 seconds

---

### Sprint 9 — Full Validation, Phase Gate & Demo Package
**Duration: ~2 weeks**

#### Ensemble Validation
```
Target metrics:
  Ensemble ROC-AUC   ≥ 0.98
  False Positive Rate < 5%  at the REVIEW threshold (0.45)
  False Negative Rate < 10% at the BLOCK threshold (0.75)
  TGN AUC > Static GNN AUC by ≥ 1%
```

**🆕 Add to the validation report:** include the Step 5.0 edge-feature ablation result (with vs. without the 13 edge features) as a documented finding — this becomes evidence in `docs/model_validation_report.md` for why the architecture includes both GDS and TGN edge layers, useful for the regulatory explainability requirement in Sprint 11.

#### Demo Script (20 Minutes)

| Minute | Action |
|--------|--------|
| 0–2 | System overview: architecture diagram + infrastructure health check |
| 2–5 | Submit a known mule transaction → watch real-time BLOCK decision |
| 5–8 | SHAP waterfall: explain WHY it was blocked (top 3 features) |
| 8–11 | TGN Graph Explorer: show the ring the account belongs to |
| 11–14 | Submit a legitimate transaction → watch APPROVE decision |
| 14–17 | Model metrics page: AUC, confusion matrix, alert queue |
| 17–20 | Kill graph-service → show graceful degraded-mode recovery |

**Files to create:**
- `ml/notebooks/04_ensemble_validation.ipynb` — Full validation report
- `docs/demo_script.md` — Step-by-step demo instructions with pre-seeded data
- `docs/model_validation_report.md` — Official metrics report

---

## Phase 5 — Hardening, Scale & Launch Readiness
*(unchanged from v1)*

### Sprint 10 — Load Testing & Chaos Engineering
**Duration: ~4 weeks**

```
Required: P95 < 50ms at 50,000 TPS
Required: Error rate < 0.1%
Required: Zero transaction loss (Kafka replay verification)
```

| Scenario | Expected Behavior |
|----------|-----------------|
| Neo4j primary killed | Replica takes over in < 30s, zero scoring loss |
| Redis eviction forced | Falls back to Flink state store, scores continue |
| Flink job killed | Kafka buffers events, zero events lost, auto-recovers |
| AI Engine pod killed | Circuit breaker fires, new pod < 20s, no 500 errors |
| Kafka partition leader failure | Auto-reassigns in < 10s, zero message loss |

**Files to create:**
- `tests/load/k6_load_test.js` — Progressive ramp script
- `scripts/chaos/` — 5 chaos test scripts
- `k8s/` — Kubernetes manifests (HPA, replica sets, Neo4j read replica routing)

---

### Sprint 11 — Security Audit, Compliance & Launch
**Duration: ~3 weeks**

#### Security Penetration Test (OWASP Top 10 scope)
- SQL injection on all PostgreSQL endpoints
- JWT token manipulation on the orchestrator
- SSRF via the SHAP explanation endpoint
- Data exfiltration via graph queries
- Neo4j Cypher injection via account ID inputs
- Kafka topic ACL bypass
- Kong rate-limit brute-force

**Required:** 0 Critical findings, 0 High findings

#### Regulatory Documentation

| Framework | Document Required |
|-----------|-----------------|
| RBI IT Framework 2023 | Model governance, audit trail, explainability |
| PMLA 2002 | Transaction monitoring, STR filing triggers |
| DPDP Act 2023 | Data minimization, consent, retention limits |
| FIU-IND Reporting | Alert format, escalation timelines |
| PCI-DSS v4.0 | Cardholder data protection, tokenization |

**🆕 Note for this section:** `ml/features/tgn_edge_feature_spec.json` from Step 5.0 becomes a direct input to the RBI explainability documentation — it's already a formula-level spec of every real-time signal feeding TGN's decisions.

#### Canary Deployment Drill
```
LightGBM v1 → v1.1 canary:
  Step 1: 5% traffic to v1.1 for 24h → monitor error rate
  Step 2: 25% traffic for 48h → monitor
  Step 3: 100% cutover
  Rollback trigger: error rate > 0.5% auto-reverts within 60s
```

---

## Immediate Next Steps (This Week)

```
Priority 1: COMPLETE Step 4A — seed_neo4j.py
  File: scripts/seed_neo4j.py (currently 69-byte stub)
  Action: Write full IBM AML → Neo4j bulk seeding with GDS algorithms

Priority 2: COMPLETE Step 4B — Static GNN Baseline (lean run)
  File: ml/training/static_gnn_baseline.py (new file)
  Action: PyTorch Geometric GraphSAGE on the seeded graph, record baseline AUC

Priority 3: 🆕 COMPLETE Step 5.0 — TGN Edge Feature Build
  File: ml/features/build_tgn_edge_features.py (new file)
  Action: Build the 13-feature edge tensor (Tier 1 first, Tier 2 — amount_ratio/
  amount_difference — not optional) before touching tgn_train.py

Priority 4: COMPLETE Step 4C — AI Engine /score endpoint
  File: services/ai-engine/app/main.py (upgrade from health stub)
  Action: Load lgbm_best.pkl + autoencoder_best.pt, add /score endpoint with SHAP
  (Can run in parallel with Priority 3 — no shared dependency)
```

---

## All Files — Complete Delivery Checklist

### Already Delivered ✅
```
Phase 1 & 2 Core:
✅ docker-compose.yml, .env, .env.example, .gitignore
✅ infrastructure/postgres/schema.sql
✅ infrastructure/kafka/topics.ps1, topics.sh
✅ infrastructure/neo4j/seed_graph.cypher (schema only)
✅ infrastructure/grafana/ (full dashboards)
✅ infrastructure/prometheus/ (alerts + scrape config)
✅ .github/workflows/ci.yml, build.yml
✅ .pre-commit-config.yaml, scripts/install_hooks.ps1
✅ services/ (all 5 service skeletons with /health + /metrics)
✅ scripts/data_prep.py
✅ scripts/feature_engineering.py (66-feature pipeline)
✅ scripts/train_lgbm_aml.py + continue_training.py
✅ scripts/evaluate_autoencoder.py
✅ scripts/seed_kafka.py
✅ ml/notebooks/01_data_exploration.ipynb
✅ ml/models/lightgbm/ (model.txt, metrics.json, threshold.json, importance.csv)
✅ ml/models/autoencoder_v2/ (autoencoder_oot_v2.pt, scaler_stats.json, threshold_config.json, training_log.csv)
✅ ml/results/ (7 EDA plots)
✅ README.md
```

### Still To Build 🔒
```
Phase 2 Completion:
🔒 scripts/seed_neo4j.py — IBM AML → Neo4j bulk load + GDS
🔒 ml/training/static_gnn_baseline.py — GraphSAGE baseline
🔒 ml/models/baseline_gnn_auc.json — Static GNN result record
🔒 services/ai-engine/app/main.py — /score endpoint with SHAP (upgrade)

Phase 3 (TGN + Orchestrator):
🆕 🔒 ml/features/build_tgn_edge_features.py — 13 dynamic edge features (NEW)
🆕 🔒 ml/features/tgn_edge_features.parquet — output (NEW)
🆕 🔒 ml/features/tgn_edge_feature_spec.json — feature documentation (NEW)
🔒 ml/training/tgn_model.py — TGN architecture
🔒 ml/training/tgn_train.py — Training loop + temporal evaluation
🔒 ml/models/tgn_v1_best.pt — Trained TGN weights
🆕 🔒 ml/models/tgn_ablation_no_edge_features.json — ablation record (NEW)
🔒 services/graph-service/app/main.py — /graph-score endpoint (upgrade)
🔒 services/scoring-orchestrator/ — Full risk fusion logic (upgrade)

Phase 4 (Explainability + Dashboard + Validation):
🔒 services/explainability-service/ — New service, port 8005
🔒 dashboard/ — Next.js 14 app (replace placeholder)
🔒 ml/notebooks/04_ensemble_validation.ipynb
🔒 docs/demo_script.md
🔒 docs/model_validation_report.md

Phase 5 (Scale + Security + Launch):
🔒 tests/load/k6_load_test.js — Progressive 50K TPS test
🔒 k8s/ — Production Kubernetes manifests
🔒 docs/regulatory_compliance.md — RBI / PMLA / DPDP / FIU-IND / PCI-DSS mapping
🔒 docs/pentest_report.md — OWASP findings (0 Critical / 0 High gate)
🔒 docs/canary_runbook.md — Deployment procedure
```

---

## Key Metrics Targets — Go/No-Go Gates

| Gate | Metric | Target | Current |
|------|--------|--------|---------|
| LightGBM ROC-AUC | OOT test set | ≥ 0.93 | **0.9905 ✅** |
| LightGBM Top 1% Recall | Alert budget | ≥ 60% | **85.67% ✅** |
| Autoencoder Anomaly | Unsupervised | Lift > 5x | **5.1x ✅** |
| Static GNN AUC | Random split | ≥ 0.85 | **0.8731 ✅** |
| TGN Temporal AUC | OOT test set | ≥ 0.97 | **0.8636 ⚠️** (Passed on PR-AUC 0.8673) |
| TGN Temporal AUC | Temporal split | ≥ 0.97 | 🔒 Not started |
| TGN vs Static GNN | Delta | ≥ +1% | 🔒 Not started |
| Ring Role Accuracy | HUB/BRIDGE/MULE/NONE | ≥ 0.75 | 🔒 Not started |
| Ensemble AUC | Held-out | ≥ 0.98 | 🔒 Not started |
| P95 Latency | 100 users | < 50ms | 🔒 Not started |
| P95 Latency | 50K TPS | < 50ms | 🔒 Not started |
| Security Pentest | CVSS Critical | 0 findings | 🔒 Not started |

---

*MuleGuard.AI — Complete Forward Plan v2 · Revised August 2026*  
*Adds Step 5.0 (TGN Dynamic Edge Features) inside Sprint 5. All prior task.md and v1 forward-plan documents are superseded by this document.*
