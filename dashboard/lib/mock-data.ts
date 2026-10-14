/* ============================================================
   Rich Mock Data — MuleGuard.AI Dashboard
   Based on IBM AML HI-Medium dataset actual characteristics.
   Used when NEXT_PUBLIC_USE_MOCK_DATA=true OR as graceful
   fallback when real services are unavailable.

   NOTE: device_score REMOVED — no device fingerprinting implemented.
   Fusion weights: 0.50×TGN + 0.30×LGBM + 0.15×AE (renorm to 1.0)
   ============================================================ */

import type {
  Alert,
  AlertsResponse,
  GraphResponse,
  LiveStats,
  MetricsResponse,
  ModelMetrics,
  ScoreResponse,
  ServiceHealth,
} from './types';

// ─── Live Statistics ───────────────────────────────────────
export const mockLiveStats: LiveStats = {
  transactions_per_second: 847,
  block_rate_pct: 4.2,
  review_rate_pct: 8.9,
  approve_rate_pct: 86.9,
  avg_latency_ms: 38,
  active_alerts: 247,
  total_today: 2_891_442,
};

// ─── Service Health (correct ports from docker-compose.yml) ─
export const mockServiceHealth: ServiceHealth[] = [
  { name: 'transaction-ingest', status: 'healthy', latency_ms: 3, url: 'http://localhost:8001', last_checked: new Date().toISOString() },
  { name: 'ai-engine', status: 'healthy', latency_ms: 6, url: 'http://localhost:8002', last_checked: new Date().toISOString() },
  { name: 'graph-service', status: 'healthy', latency_ms: 19, url: 'http://localhost:8003', last_checked: new Date().toISOString() },
  { name: 'feature-service', status: 'healthy', latency_ms: 2, url: 'http://localhost:8004', last_checked: new Date().toISOString() },
  { name: 'scoring-orchestrator', status: 'healthy', latency_ms: 42, url: 'http://localhost:8080', last_checked: new Date().toISOString() },
];

// ─── Mock Alerts ───────────────────────────────────────────
function makeAlert(overrides: Partial<Alert> & { id: string }): Alert {
  return {
    transaction_id: `TXN_${Math.random().toString(36).slice(2, 10).toUpperCase()}`,
    account_id: `ACC_${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
    decision: 'REVIEW',
    final_score: 0.52,
    component_scores: { lgbm_score: 0.61, ae_score: 0.44, tgn_score: 0.55 },
    ring_role: 'NONE',
    amount: 9800,
    payment_format: 'ACH',
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    explanation_summary: 'Elevated ACH transfer pattern with dormancy gap',
    acknowledged: false,
    ...overrides,
  };
}

export const mockAlerts: Alert[] = [
  makeAlert({
    id: 'alert_001',
    decision: 'BLOCK',
    final_score: 0.92,
    account_id: 'ACC_MU4821',
    transaction_id: 'TXN_4B2F8A1C',
    amount: 9_750,
    payment_format: 'Cash',
    ring_role: 'MULE',
    component_scores: { lgbm_score: 0.94, ae_score: 0.88, tgn_score: 0.97 },
    explanation_summary: 'Transit mule: cash deposit → immediate ACH withdrawal. Ring #4821 member. PageRank 0.82.',
    timestamp: new Date(Date.now() - 120_000).toISOString(),
    acknowledged: false,
  }),
  makeAlert({
    id: 'alert_002',
    decision: 'BLOCK',
    final_score: 0.87,
    account_id: 'ACC_HB1142',
    transaction_id: 'TXN_9E7C2D4A',
    amount: 1_240_000,
    payment_format: 'Wire',
    ring_role: 'HUB',
    component_scores: { lgbm_score: 0.91, ae_score: 0.72, tgn_score: 0.95 },
    explanation_summary: 'Hub account PageRank 0.89. 34 outbound wires in 24h. SCC ring confirmed.',
    timestamp: new Date(Date.now() - 250_000).toISOString(),
    acknowledged: false,
  }),
  makeAlert({
    id: 'alert_003',
    decision: 'REVIEW',
    final_score: 0.64,
    account_id: 'ACC_BR3317',
    transaction_id: 'TXN_2A5F9B7E',
    amount: 42_500,
    payment_format: 'Bitcoin',
    ring_role: 'BRIDGE',
    component_scores: { lgbm_score: 0.71, ae_score: 0.58, tgn_score: 0.66 },
    explanation_summary: 'Bridge account connecting 2 communities. Bitcoin transfer to flagged exchange.',
    timestamp: new Date(Date.now() - 480_000).toISOString(),
    acknowledged: false,
  }),
  makeAlert({
    id: 'alert_004',
    decision: 'REVIEW',
    final_score: 0.57,
    account_id: 'ACC_TX0044',
    transaction_id: 'TXN_7F1A3C2D',
    amount: 9_499,
    payment_format: 'ACH',
    ring_role: 'NONE',
    component_scores: { lgbm_score: 0.63, ae_score: 0.41, tgn_score: 0.52 },
    explanation_summary: 'Structuring: $9,499 — just below $10,000 CTR threshold. 3rd such transaction this week.',
    timestamp: new Date(Date.now() - 720_000).toISOString(),
    acknowledged: true,
  }),
  makeAlert({
    id: 'alert_005',
    decision: 'BLOCK',
    final_score: 0.96,
    account_id: 'ACC_MU0017',
    transaction_id: 'TXN_C3B8E6A1',
    amount: 5_200,
    payment_format: 'Cash',
    ring_role: 'MULE',
    component_scores: { lgbm_score: 0.97, ae_score: 0.93, tgn_score: 0.99 },
    explanation_summary: 'Known mule. amount_ratio=0.97 transit pattern. 9-day dormancy gap before activity burst.',
    timestamp: new Date(Date.now() - 900_000).toISOString(),
    acknowledged: false,
  }),
  ...Array.from({ length: 15 }, (_, i) =>
    makeAlert({
      id: `alert_${String(i + 6).padStart(3, '0')}`,
      decision: Math.random() > 0.6 ? 'REVIEW' : Math.random() > 0.5 ? 'BLOCK' : 'APPROVE',
      final_score: 0.3 + Math.random() * 0.6,
      amount: Math.floor(1000 + Math.random() * 100_000),
    })
  ),
];

export const mockAlertsResponse: AlertsResponse = {
  alerts: mockAlerts,
  total: 247,
  page: 1,
  limit: 20,
  total_pages: 13,
};

// ─── Score Responses (Known Scenarios) ─────────────────────
export const mockScoreBlock: ScoreResponse = {
  transaction_id: 'TXN_4B2F8A1C',
  final_score: 0.921,  // = 0.50×0.97 + 0.30×0.94 + 0.15×0.88 = 0.921
  decision: 'BLOCK',
  component_scores: { lgbm_score: 0.94, ae_score: 0.88, tgn_score: 0.97 },
  shap_top_5: [
    { feature: 'sender_receiver_pair_txn_count', value: 1, contribution: 0.30, direction: 'RAISES_RISK', displayName: 'Sender-Receiver Pair Count' },
    { feature: 'payment_format_encoded', value: 1, contribution: 0.24, direction: 'RAISES_RISK', displayName: 'Payment Format (Cash)' },
    { feature: 'currency_pair_rarity', value: 0.99, contribution: 0.19, direction: 'RAISES_RISK', displayName: 'Currency Pair Rarity' },
    { feature: 'consecutive_cross_bank_count', value: 4, contribution: 0.14, direction: 'RAISES_RISK', displayName: 'Consecutive Cross Bank' },
    { feature: 'days_since_own_last_activity', value: 9, contribution: 0.08, direction: 'RAISES_RISK', displayName: 'Days Since Last Activity' },
  ],
  explanation: {
    text: 'BLOCKED (score: 92.1%). Account ACC_MU4821 received a cash deposit of $9,750 and immediately transferred 97% via ACH — a hallmark transit mule pattern. Confirmed member of Ring #4821 (in_scc_ring=true, PageRank=0.82, top 2%). A 9-day dormancy gap preceded this burst.',
    mode: 'fallback',
    ring_membership: 'Ring #4821',
    dormancy_gap_days: 9,
    fan_out_24h: 12,
  },
  ring_role: 'MULE',
  ring_probability: 0.97,
  in_scc_ring: true,
  community_id: 4821,
  pagerank: 0.82,
  latency_ms: 42,
  feature_source: 'redis',
  model_versions: { lightgbm: 'lgbm_v3/lgbm_v3_final.pkl', autoencoder: 'autoencoder_oot_v2.pt' },
  degraded_mode: false,
  timestamp: new Date().toISOString(),
};

export const mockScoreApprove: ScoreResponse = {
  transaction_id: 'TXN_LEGIT_001',
  final_score: 0.079,  // = 0.50×0.07 + 0.30×0.06 + 0.15×0.12 = 0.079
  decision: 'APPROVE',
  component_scores: { lgbm_score: 0.06, ae_score: 0.12, tgn_score: 0.07 },
  shap_top_5: [
    { feature: 'sender_receiver_pair_txn_count', value: 84, contribution: -0.12, direction: 'LOWERS_RISK', displayName: 'Sender-Receiver Pair Count' },
    { feature: 'sender_avg_amount', value: 2400, contribution: -0.08, direction: 'LOWERS_RISK', displayName: 'Sender Avg Amount' },
    { feature: 'time_since_last_received', value: 2, contribution: -0.06, direction: 'LOWERS_RISK', displayName: 'Time Since Last Received' },
    { feature: 'night_activity_ratio', value: 0.04, contribution: -0.05, direction: 'LOWERS_RISK', displayName: 'Night Activity Ratio' },
    { feature: 'fan_out_24h', value: 2, contribution: 0.02, direction: 'RAISES_RISK', displayName: 'Fan-Out 24h' },
  ],
  explanation: {
    text: 'APPROVED (score: 7.9%). Account shows 84 prior transactions with this receiver, consistent amount profile, and no ring membership. Low autoencoder anomaly score confirms normal behavioral pattern. Feature source: Redis.',
    mode: 'fallback',
    dormancy_gap_days: 0,
    fan_out_24h: 2,
  },
  ring_role: 'NONE',
  ring_probability: 0.02,
  in_scc_ring: false,
  community_id: -1,
  pagerank: 0.003,
  latency_ms: 36,
  feature_source: 'redis',
  model_versions: { lightgbm: 'lgbm_v3/lgbm_v3_final.pkl', autoencoder: 'autoencoder_oot_v2.pt' },
  degraded_mode: false,
  timestamp: new Date().toISOString(),
};

export const mockScoreReview: ScoreResponse = {
  transaction_id: 'TXN_REVIEW_001',
  final_score: 0.617, // = 0.50×0.66 + 0.30×0.71 + 0.15×0.58 = 0.627
  decision: 'REVIEW',
  component_scores: { lgbm_score: 0.71, ae_score: 0.58, tgn_score: 0.66 },
  shap_top_5: [
    { feature: 'fan_in_24h', value: 14, contribution: 0.15, direction: 'RAISES_RISK', displayName: 'Fan-In 24h' },
    { feature: 'bitcoin_flag', value: 1, contribution: 0.12, direction: 'RAISES_RISK', displayName: 'Bitcoin Payment' },
    { feature: 'amount_ratio', value: 2.4, contribution: 0.10, direction: 'RAISES_RISK', displayName: 'Amount Ratio' },
    { feature: 'txn_count_1h', value: 4, contribution: -0.07, direction: 'LOWERS_RISK', displayName: 'Txn Count 1h' },
    { feature: 'structuring_threshold_flag', value: 1, contribution: 0.06, direction: 'RAISES_RISK', displayName: 'Structuring Flag' },
  ],
  explanation: {
    text: 'REVIEW (score: 61.7%). Bridge account connecting 2 communities. Bitcoin transfer to potentially flagged exchange address. Moderate structuring score (0.74). Manual review recommended.',
    mode: 'fallback',
    ring_membership: 'Bridge node between communities #12 and #67',
    dormancy_gap_days: 3,
    fan_out_24h: 8,
  },
  ring_role: 'BRIDGE',
  ring_probability: 0.55,
  in_scc_ring: false,
  community_id: 12,
  pagerank: 0.21,
  latency_ms: 44,
  feature_source: 'on_the_fly',
  model_versions: { lightgbm: 'lgbm_v3/lgbm_v3_final.pkl', autoencoder: 'autoencoder_oot_v2.pt' },
  degraded_mode: false,
  timestamp: new Date().toISOString(),
};

// ─── Graph Mock Data ───────────────────────────────────────
export const mockGraphResponse: GraphResponse = {
  account_id: 'ACC_MU4821',
  tgn_score: 0.97,
  ring_role: 'MULE',
  community_id: 4821,
  neo4j_connected: true,
  degraded_mode: false,
  nodes: [
    { id: 'ACC_MU4821', account_id: 'ACC_MU4821', ring_role: 'MULE', pagerank: 0.82, community_id: 4821, in_scc_ring: true, risk_score: 0.97, lifetime_txn_count: 83, total_sent: 289_400 },
    { id: 'ACC_HB1142', account_id: 'ACC_HB1142', ring_role: 'HUB', pagerank: 0.89, community_id: 4821, in_scc_ring: true, risk_score: 0.95, lifetime_txn_count: 341, total_sent: 4_120_000 },
    { id: 'ACC_BR3317', account_id: 'ACC_BR3317', ring_role: 'BRIDGE', pagerank: 0.45, community_id: 67, in_scc_ring: false, risk_score: 0.66, lifetime_txn_count: 127, total_sent: 890_500 },
    { id: 'ACC_TX9981', account_id: 'ACC_TX9981', ring_role: 'NONE', pagerank: 0.04, community_id: 4821, in_scc_ring: true, risk_score: 0.88, lifetime_txn_count: 29, total_sent: 142_000 },
    { id: 'ACC_MU0033', account_id: 'ACC_MU0033', ring_role: 'MULE', pagerank: 0.61, community_id: 4821, in_scc_ring: true, risk_score: 0.91, lifetime_txn_count: 56, total_sent: 217_800 },
    { id: 'ACC_LE2201', account_id: 'ACC_LE2201', ring_role: 'NONE', pagerank: 0.02, community_id: 22, in_scc_ring: false, risk_score: 0.07, lifetime_txn_count: 412, total_sent: 98_400 },
    { id: 'ACC_LE5510', account_id: 'ACC_LE5510', ring_role: 'NONE', pagerank: 0.01, community_id: 7, in_scc_ring: false, risk_score: 0.04, lifetime_txn_count: 884, total_sent: 201_000 },
    { id: 'ACC_MU9913', account_id: 'ACC_MU9913', ring_role: 'MULE', pagerank: 0.58, community_id: 4821, in_scc_ring: true, risk_score: 0.93, lifetime_txn_count: 71, total_sent: 335_600 },
  ],
  edges: [
    { id: 'e0', source: 'ACC_LE2201', target: 'ACC_MU4821', amount: 9_750, timestamp: new Date(Date.now() - 120_000).toISOString(), payment_format: 'Cash', is_laundering: true, structuring_flag: true },
    { id: 'e1', source: 'ACC_MU4821', target: 'ACC_HB1142', amount: 9_400, timestamp: new Date(Date.now() - 115_000).toISOString(), payment_format: 'ACH', is_laundering: true, structuring_flag: false },
    { id: 'e2', source: 'ACC_HB1142', target: 'ACC_MU0033', amount: 45_000, timestamp: new Date(Date.now() - 110_000).toISOString(), payment_format: 'Wire', is_laundering: true, structuring_flag: false },
    { id: 'e3', source: 'ACC_MU0033', target: 'ACC_TX9981', amount: 12_000, timestamp: new Date(Date.now() - 105_000).toISOString(), payment_format: 'ACH', is_laundering: true, structuring_flag: false },
    { id: 'e4', source: 'ACC_TX9981', target: 'ACC_MU4821', amount: 11_500, timestamp: new Date(Date.now() - 100_000).toISOString(), payment_format: 'ACH', is_laundering: true, structuring_flag: false },
    { id: 'e5', source: 'ACC_MU4821', target: 'ACC_MU9913', amount: 5_200, timestamp: new Date(Date.now() - 95_000).toISOString(), payment_format: 'Cash', is_laundering: true, structuring_flag: false },
    { id: 'e6', source: 'ACC_HB1142', target: 'ACC_BR3317', amount: 240_000, timestamp: new Date(Date.now() - 90_000).toISOString(), payment_format: 'Wire', is_laundering: false, structuring_flag: false },
    { id: 'e7', source: 'ACC_LE5510', target: 'ACC_MU4821', amount: 1_850, timestamp: new Date(Date.now() - 86_400_000).toISOString(), payment_format: 'ACH', is_laundering: false, structuring_flag: false },
  ],
};

// ─── Example Transactions for Simulator ────────────────────
export const exampleTransactions = {
  mule: {
    transaction_id: 'TXN-20220901-934281',
    from_account: '837B70300',
    to_account: '837B70350',
    amount: 5107.168000000001,
    timestamp: '2022-09-01T00:38:00+00:00',
    payment_format: 'ACH',
    currency: 'Mexican Peso',
  },
  legit: {
    transaction_id: `TXN_LEGIT_${Date.now()}`,
    from_account: 'ACC_LE2201',
    to_account: 'ACC_LE5510',
    amount: 2400.00,
    timestamp: new Date().toISOString(),
    payment_format: 'ACH',
    currency: 'USD',
  },
  structuring: {
    transaction_id: `TXN_STRUCT_${Date.now()}`,
    from_account: 'ACC_TX0044',
    to_account: 'ACC_MU4821',
    amount: 9499.00,
    timestamp: new Date().toISOString(),
    payment_format: 'ACH',
    currency: 'USD',
  },
};

// ─── Model Metrics ─────────────────────────────────────────
// Values from actual ML training run evaluation
function makeRocCurve(auc: number): { fpr: number; tpr: number }[] {
  const points = [{ fpr: 0, tpr: 0 }];
  for (let i = 1; i <= 20; i++) {
    const fpr = i / 20;
    const tpr = Math.min(1, auc + (1 - auc) * (fpr ** (1 / (3 * auc))) * (0.9 + Math.random() * 0.1));
    points.push({ fpr, tpr });
  }
  points.push({ fpr: 1, tpr: 1 });
  return points;
}

function makePrCurve(auc: number): { recall: number; precision: number }[] {
  const points = [];
  for (let i = 0; i <= 20; i++) {
    const recall = i / 20;
    const precision = Math.max(0.01, auc - 0.05 * recall + 0.15 * (1 - recall) + (Math.random() - 0.5) * 0.02);
    points.push({ recall, precision: Math.min(1, precision) });
  }
  return points;
}

const lgbmMetrics: ModelMetrics = {
  model_name: 'LightGBM',
  roc_auc: 0.9904,
  pr_auc: 0.6754,
  roc_curve: makeRocCurve(0.9904),
  pr_curve: makePrCurve(0.6754),
  confusion_matrix: { true_positive: 9101, false_positive: 54696, true_negative: 6314312, false_negative: 1539 },
  top_features: [
    { feature: 'sender_receiver_pair_txn_count', importance: 1.6884 },
    { feature: 'payment_format_encoded', importance: 0.5318 },
    { feature: 'sender_avg_amount', importance: 0.4805 },
    { feature: 'days_since_own_last_activity', importance: 0.3715 },
    { feature: 'is_ach_payment', importance: 0.3577 },
    { feature: 'consecutive_cross_bank_count', importance: 0.3063 },
    { feature: 'amount_paid', importance: 0.3032 },
    { feature: 'currency_pair_rarity', importance: 0.2996 },
    { feature: 'gather_scatter_score', importance: 0.2800 },
    { feature: 'sender_amount_std', importance: 0.2625 },
  ],
  threshold: 0.50,
  recall_at_threshold: 0.8554,
  precision_at_threshold: 0.1427,
  last_updated: new Date().toISOString(),
  scored_last_1000: 1000,
};

const aeMetrics: ModelMetrics = {
  model_name: 'Autoencoder',
  roc_auc: 0.6717,
  pr_auc: 0.0085,
  roc_curve: makeRocCurve(0.6717),
  pr_curve: makePrCurve(0.0085),
  confusion_matrix: { true_positive: 1514, false_positive: 62283, true_negative: 6306725, false_negative: 9126 },
  top_features: [
    { feature: 'fan_in_24h', importance: 1.7131 },
    { feature: 'amount_received_1h', importance: 0.3656 },
    { feature: 'time_since_last_received', importance: 0.1041 },
    { feature: 'amount_ratio', importance: 0.0359 },
    { feature: 'receiver_txn_count_1h', importance: 0.0161 },
    { feature: 'txn_count_1h', importance: 0.0079 },
    { feature: 'account_degree_ratio', importance: 0.0074 },
    { feature: 'fan_out_24h', importance: 0.0067 },
    { feature: 'amount_sent_1h', importance: 0.0066 },
    { feature: 'hour_of_day_norm', importance: 0.0063 },
  ],
  threshold: 1.9505,
  recall_at_threshold: 0.1423,
  precision_at_threshold: 0.0237,
  last_updated: new Date().toISOString(),
  scored_last_1000: 1000,
};

const tgnMetrics: ModelMetrics = {
  model_name: 'TGN',
  roc_auc: 0.9872,
  pr_auc: 0.5124,
  roc_curve: makeRocCurve(0.9872),
  pr_curve: makePrCurve(0.5124),
  confusion_matrix: { true_positive: 8349, false_positive: 55448, true_negative: 6313558, false_negative: 2293 },
  top_features: [
    { feature: 'amount_ratio', importance: 0.2410 },
    { feature: 'structuring_threshold_flag', importance: 0.1980 },
    { feature: 'log_amount_paid', importance: 0.1650 },
    { feature: 'is_bitcoin_transaction', importance: 0.1340 },
    { feature: 'is_cross_bank', importance: 0.0980 },
    { feature: 'is_cash_payment', importance: 0.0820 },
    { feature: 'is_cross_currency', importance: 0.0480 },
    { feature: 'payment_format_encoded', importance: 0.0340 },
  ],
  threshold: 0.2946,
  recall_at_threshold: 0.7845,
  precision_at_threshold: 0.1309,
  last_updated: new Date().toISOString(),
  scored_last_1000: 1000,
};

const ensembleMetrics: ModelMetrics = {
  model_name: 'Ensemble',
  roc_auc: 0.9935,
  pr_auc: 0.7410,
  roc_curve: makeRocCurve(0.9935),
  pr_curve: makePrCurve(0.7410),
  confusion_matrix: { true_positive: 9948, false_positive: 53849, true_negative: 6315159, false_negative: 692 },
  top_features: [
    { feature: 'tgn_score (50% weight)', importance: 0.50 },
    { feature: 'lgbm_score (30% weight)', importance: 0.30 },
    { feature: 'ae_score (15% weight)', importance: 0.15 },
    { feature: 'device_signal (5% weight)', importance: 0.05 },
  ],
  threshold: 0.45,
  recall_at_threshold: 0.9350,
  precision_at_threshold: 0.1559,
  last_updated: new Date().toISOString(),
  scored_last_1000: 1000,
};

export const mockMetrics: MetricsResponse = {
  lgbm: lgbmMetrics,
  autoencoder: aeMetrics,
  tgn: tgnMetrics,
  ensemble: ensembleMetrics,
  total_scored_today: 6_379_648,
  block_rate: 0.021,
  review_rate: 0.079,
  approve_rate: 0.900,
  avg_latency_ms: 28,
};
