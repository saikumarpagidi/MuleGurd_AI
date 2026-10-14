/* ============================================================
   TypeScript Types — MuleGuard.AI Dashboard
   Mirrors the exact wire contracts of the real backend services.

   Sources:
     - services/ai-engine/app/schemas.py     → AIEngineScore
     - services/graph-service/app/api/endpoints/graph_score.py → GraphScore
     - services/graph-service/app/api/endpoints/subgraph.py   → SubgraphResponse
     - services/scoring-orchestrator/.../  → FusedScore (orchestrator output)
   ============================================================ */

// ─── Primitive Enums ───────────────────────────────────────
export type Decision     = 'APPROVE' | 'REVIEW' | 'BLOCK';
export type RingRole     = 'HUB' | 'BRIDGE' | 'MULE' | 'NONE';
export type ServiceStatus = 'healthy' | 'degraded' | 'offline' | 'unknown';
export type ExplanationMode = 'llm' | 'fallback';
export type ShapDirection = 'RAISES_RISK' | 'LOWERS_RISK';
export type AeAnomalyDirection = 'NO_ANOMALY_CONTRIBUTION' | 'SMALL_ANOMALY_CONTRIBUTION' | 'RAISES_ANOMALY';

// ─── AI Engine — POST /score response ─────────────────────
// maps directly to services/ai-engine/app/schemas.py::ScoreResponse
export interface AIEngineShapFeature {
  feature: string;
  value: number;             // observed feature value
  shap_contribution: number; // signed SHAP value (logit-space)
  direction: ShapDirection;
}

/** Per-feature Autoencoder reconstruction anomaly.
 *  Unlike SHAP, direction is always positive (anomaly severity only).
 *  reconstruction_z_score: 0 = normal, >=1.0 = strongly anomalous.
 */
export interface AIEngineAeFeature {
  feature: string;
  value: number;                         // observed (pre-standardization) value
  reconstruction_z_score: number;        // feature-level squared error Z-score
  direction: AeAnomalyDirection;
}

export interface AIEngineScore {
  transaction_id: string;
  lgbm_score: number;        // 0-1
  ae_score: number;          // 0-1 normalized anomaly score
  ae_reconstruction_error: number;
  shap_top_5: AIEngineShapFeature[];
  ae_top_5?: AIEngineAeFeature[];        // empty list on cold-start or normal tx
  decision_preliminary: Decision;
  feature_source: 'redis' | 'on_the_fly';
  model_versions: Record<string, string>;
  inference_latency_ms: number;
}

// ─── Graph Service — POST /graph-score response ────────────
// maps to services/graph-service/app/api/endpoints/graph_score.py::GraphScoreResponse
export interface GraphScoreResult {
  tgn_score: number | null;  // TGN risk probability (null if heuristic mode)
  is_mule_probability: number | null;
  ring_role: RingRole;
  ring_probability: number;
  in_scc_ring: boolean;
  community_id: number;      // -1 if unknown
  pagerank: number;
  processing_time_ms: number;
  degraded_mode: boolean;
  model_version: string;
}

// ─── Graph Service — GET /subgraph/{account_id} response ──
// maps to services/graph-service/app/api/endpoints/subgraph.py::SubgraphResponse
export interface SubgraphEdge {
  src: string;
  dst: string;
  amount: number;
  timestamp_s: number;       // unix timestamp
  payment_format: string;
  is_laundering: boolean;
}

export interface SubgraphResponse {
  account_id: string;
  account_properties: AccountProperties;
  edges: SubgraphEdge[];
  edge_count: number;
  neo4j_connected: boolean;
}

export interface AccountProperties {
  pagerank?: number;
  community_id?: number;
  in_scc_ring?: boolean;
  ring_role?: string;
  // any other GDS fields returned by fetch_gds_signals
  [key: string]: unknown;
}

// ─── Dashboard Fused Score (computed by Next.js proxy) ────
// The orchestrator does the real fusion — but when directly calling
// ai-engine + graph-service separately, we fuse here with correct weights:
//   0.50 × TGN + 0.30 × LightGBM + 0.15 × Autoencoder
// (device weight 0.05 removed — no device fingerprinting implemented)
export interface ComponentScores {
  lgbm_score: number;  // from ai-engine
  ae_score: number;    // from ai-engine
  tgn_score: number;   // from graph-service tgn_score (or 0 if degraded)
}

// Normalized SHAP feature for dashboard display
export interface SHAPFeature {
  feature: string;
  value: number;
  contribution: number;        // same as shap_contribution
  direction: ShapDirection;
  displayName: string;         // human-readable label
}

/** Normalized AE anomaly feature for dashboard display. */
export interface AEFeature {
  feature: string;
  value: number;
  reconstruction_z_score: number;
  direction: AeAnomalyDirection;
  displayName: string;         // human-readable label
}

export interface Explanation {
  text: string;
  mode: ExplanationMode;
  ring_membership?: string;
  dormancy_gap_days?: number;
  fan_out_24h?: number;
}

// Dashboard unified score block (returned by /api/score proxy)
export interface ScoreResponse {
  transaction_id: string;
  final_score: number;
  decision: Decision;
  component_scores: ComponentScores;
  shap_top_5: SHAPFeature[];
  ae_top_5?: AEFeature[];      // Autoencoder anomaly features (empty on cold-start)
  explanation: Explanation;
  ring_role: RingRole;
  ring_probability: number;
  in_scc_ring: boolean;
  is_mule_probability?: number | null;
  community_id: number;
  pagerank: number;
  latency_ms: number;
  feature_source: string;
  model_versions: Record<string, string>;
  degraded_mode: boolean;   // true if graph service was in heuristic mode
  timestamp: string;
}

// ─── Graph Explorer Types (built from subgraph response) ───
export interface GraphNode {
  id: string;            // unique for D3
  account_id: string;
  ring_role: RingRole;
  pagerank: number;
  community_id: number;
  in_scc_ring: boolean;
  risk_score: number;    // derived from tgn_score if available, else heuristic
  is_mule_probability?: number | null;
  lifetime_txn_count: number;
  total_sent: number;
}

export interface GraphEdge {
  id: string;            // unique for D3
  source: string;        // node id
  target: string;        // node id
  amount: number;
  timestamp: string;
  payment_format: string;
  is_laundering: boolean;
  structuring_flag: boolean;
}

export interface GraphResponse {
  account_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  tgn_score: number;
  is_mule_probability?: number | null;
  ring_role: RingRole;
  community_id: number;
  neo4j_connected: boolean;
  degraded_mode: boolean;
}

// ─── Alerts ────────────────────────────────────────────────
// Populated from Postgres scoring_decisions table via orchestrator or
// direct DB query (when orchestrator is not accessible)
export interface Alert {
  id: string;
  transaction_id: string;
  account_id: string;
  decision: Decision;
  final_score: number;
  component_scores: ComponentScores;
  ring_role: RingRole;
  amount: number;
  payment_format: string;
  timestamp: string;
  explanation_summary: string;
  acknowledged: boolean;
}

export interface AlertsResponse {
  alerts: Alert[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

// ─── Model Metrics ─────────────────────────────────────────
export interface ROCPoint {
  fpr: number;
  tpr: number;
}

export interface PRPoint {
  recall: number;
  precision: number;
}

export interface ConfusionMatrixData {
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
}

export interface ModelMetrics {
  model_name: string;
  roc_auc: number;
  pr_auc: number;
  roc_curve: ROCPoint[];
  pr_curve: PRPoint[];
  confusion_matrix: ConfusionMatrixData;
  top_features: { feature: string; importance: number }[];
  threshold: number;
  recall_at_threshold: number;
  precision_at_threshold: number;
  last_updated: string;
  scored_last_1000: number;
}

export interface MetricsResponse {
  lgbm: ModelMetrics;
  autoencoder: ModelMetrics;
  tgn: ModelMetrics;
  ensemble: ModelMetrics;
  total_scored_today: number;
  block_rate: number;
  review_rate: number;
  approve_rate: number;
  avg_latency_ms: number;
}

// ─── Live Stats & Health ───────────────────────────────────
export interface LiveStats {
  transactions_per_second: number;
  block_rate_pct: number;
  review_rate_pct: number;
  approve_rate_pct: number;
  avg_latency_ms: number;
  active_alerts: number;
  total_today: number;
}

export interface ServiceHealth {
  name: string;
  status: ServiceStatus;
  latency_ms: number | null;
  url: string;
  last_checked: string;
}

// ─── Transaction Input (Simulator) ─────────────────────────
// Matches services/ai-engine/app/schemas.py::TransactionInput
// Required: transaction_id, from_account, to_account, amount, timestamp
export interface TransactionInput {
  transaction_id: string;
  from_account: string;
  to_account: string;
  amount: number;
  timestamp: string;        // ISO 8601
  payment_format: string;   // PaymentFormat enum value
  currency: string;         // default "USD"
}

// ─── Filter / Pagination ───────────────────────────────────
export interface AlertFilters {
  decision?: Decision | 'ALL';
  min_score?: number;
  max_score?: number;
  date_from?: string;
  date_to?: string;
  page?: number;
  limit?: number;
}

// ─── AI Engine Direct Telemetry & Batch Types ──────────────
export interface AIEngineHealth {
  status: 'ok' | 'degraded' | 'offline';
  lgbm_model_loaded: boolean;
  autoencoder_model_loaded: boolean;
  shap_explainer_ready: boolean;
  redis_connected: boolean;
  uptime_seconds: number;
  endpoint_url: string;
  ping_ms?: number;
}

export interface BatchSimulationItem {
  index: number;
  transaction_id: string;
  from_account: string;
  to_account: string;
  amount: number;
  payment_format: string;
  lgbm_score: number;
  ae_score: number;
  ae_reconstruction_error: number;
  combined_score: number;
  decision: Decision;
  feature_source: 'redis' | 'on_the_fly';
  latency_ms: number;
  sla_breached: boolean;
}

export interface BatchSimulationSummary {
  total_count: number;
  approved_count: number;
  review_count: number;
  blocked_count: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  sla_breach_count: number;
  sla_compliance_pct: number;
  redis_source_pct: number;
  throughput_tps: number;
  items: BatchSimulationItem[];
}

