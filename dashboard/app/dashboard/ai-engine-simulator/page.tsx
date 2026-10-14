'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import DashboardLayout from '@/components/layout/DashboardLayout';
import type { AIEngineHealth, AIEngineScore, BatchSimulationSummary, Decision, TransactionInput } from '@/lib/types';
import styles from './page.module.css';

const DEFAULT_TRANSACTION: TransactionInput = {
  transaction_id: 'TXN-20220901-934281',
  from_account: '837B70300',
  to_account: '837B70350',
  amount: 5107.168000000001,
  timestamp: '2022-09-01T00:38:00+00:00',
  payment_format: 'ACH',
  currency: 'Mexican Peso',
};

const PAYMENT_FORMATS = ['Wire', 'ACH', 'Bitcoin', 'Cash', 'Credit Card', 'Debit Card', 'Cheque', 'Reinvestment', 'Other'];

export default function AIEngineSimulatorPage() {
  // Active view tab: 'single' | 'batch'
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  // Service Health Telemetry
  const [health, setHealth] = useState<AIEngineHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  // Single transaction playground state
  const [form, setForm] = useState<TransactionInput>(DEFAULT_TRANSACTION);
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState(JSON.stringify(DEFAULT_TRANSACTION, null, 2));
  const [scoring, setScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState<AIEngineScore | null>(null);
  const [copiedJson, setCopiedJson] = useState(false);

  // Batch load & stress testing state
  const [batchCount, setBatchCount] = useState<number>(25);
  const [batchProfile, setBatchProfile] = useState<'mixed' | 'attack' | 'legitimate'>('mixed');
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchSimulationSummary | null>(null);
  const [filterDecision, setFilterDecision] = useState<Decision | 'ALL'>('ALL');

  // Load Health probe on mount and periodically
  async function checkHealth() {
    setHealthLoading(true);
    try {
      const res = await fetch('/api/ai-engine/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch {
      // offline
    } finally {
      setHealthLoading(false);
    }
  }

  useEffect(() => {
    checkHealth();
    setForm(f => ({ ...f, timestamp: new Date().toISOString() }));
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  // Run Single Scoring
  async function handleScore() {
    setScoring(true);
    let payload = form;
    if (jsonMode) {
      try {
        payload = JSON.parse(jsonText);
      } catch {
        alert('Invalid JSON syntax');
        setScoring(false);
        return;
      }
    }

    try {
      const res = await fetch('/api/ai-engine/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const data: AIEngineScore = await res.json();
        setScoreResult(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setScoring(false);
    }
  }

  // Run Batch Load / Stress Test
  async function handleRunBatch() {
    setBatchRunning(true);
    try {
      const res = await fetch('/api/ai-engine/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: batchCount, profile: batchProfile }),
      });
      if (res.ok) {
        const summary: BatchSimulationSummary = await res.json();
        setBatchResult(summary);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setBatchRunning(false);
    }
  }

  // Copy JSON Response
  function copyJsonResponse() {
    if (!scoreResult) return;
    navigator.clipboard.writeText(JSON.stringify(scoreResult, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  }

  // Export Audit Report JSON
  function exportAuditReport() {
    if (!scoreResult) return;
    const report = {
      timestamp: new Date().toISOString(),
      service: 'MuleGuard.AI AI Engine (:8002)',
      test_payload: form,
      scoring_result: scoreResult,
      service_health_at_test: health,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-engine-audit-${scoreResult.transaction_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Calculate preliminary composite score
  const preliminaryScore = scoreResult
    ? Math.round((0.7 * scoreResult.lgbm_score + 0.3 * scoreResult.ae_score) * 1000) / 10
    : 0;

  const filteredBatchItems = batchResult?.items.filter(item => {
    if (filterDecision === 'ALL') return true;
    return item.decision === filterDecision;
  }) ?? [];

  return (
    <DashboardLayout
      title="AI Engine Studio & Simulator"
      subtitle="Submit any transaction → see real-time AI decision with SHAP explanation"
    >
      <div className={styles.container}>
        {/* ─── Live Telemetry & Health Probe Bar ────────────── */}
        <div className={styles.healthBar}>
          <div className={styles.healthLeft}>
            <div
              className={`${styles.healthPill} ${health?.status === 'ok'
                ? styles.healthPillOk
                : health?.status === 'degraded'
                  ? styles.healthPillDegraded
                  : styles.healthPillOffline
                }`}
            >
              <span className={styles.statusDot} />
              <span>
                {health?.status === 'ok'
                  ? 'SERVICE ONLINE'
                  : health?.status === 'degraded'
                    ? 'SERVICE DEGRADED'
                    : 'SERVICE OFFLINE'}
              </span>
            </div>

            <div className={styles.healthMeta}>
              <span className={styles.metaItem}>
                <span className={health?.lgbm_model_loaded ? styles.metaCheck : styles.metaCross}>
                  {health?.lgbm_model_loaded ? '✓' : '✗'}
                </span>{' '}
                LightGBM Booster
              </span>
              <span className={styles.metaItem}>
                <span className={health?.autoencoder_model_loaded ? styles.metaCheck : styles.metaCross}>
                  {health?.autoencoder_model_loaded ? '✓' : '✗'}
                </span>{' '}
                Autoencoder V6
              </span>
              <span className={styles.metaItem}>
                <span className={health?.shap_explainer_ready ? styles.metaCheck : styles.metaCross}>
                  {health?.shap_explainer_ready ? '✓' : '✗'}
                </span>{' '}
                SHAP TreeExplainer
              </span>
              <span className={styles.metaItem}>
                <span className={health?.redis_connected ? styles.metaCheck : styles.metaCross}>
                  {health?.redis_connected ? '✓' : '✗'}
                </span>{' '}
                Redis Feature Store
              </span>
            </div>
          </div>

          <div className={styles.healthRight}>
            <span className={styles.endpointBadge}>
              {health?.endpoint_url ?? 'http://localhost:8002'}
            </span>
            <button
              className={styles.refreshBtn}
              onClick={checkHealth}
              disabled={healthLoading}
              title="Ping AI Engine health probe"
            >
              {healthLoading ? '↻ Checking...' : '↻ Probe'}
            </button>
          </div>
        </div>

        {/* ─── Navigation Tabs ─────────────────────────────── */}
        <div className={styles.tabNav}>
          <button
            className={`${styles.tabBtn} ${activeTab === 'single' ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab('single')}
            id="tab-single-scoring"
          >
            <span>⚡</span> Transaction Playground
          </button>
          {/* Batch Load & Stress Testing Benchmark tab is temporarily hidden */}
        </div>

        {/* ═══════════════════════════════════════════════════
            TAB 1: SINGLE TRANSACTION PLAYGROUND
           ═══════════════════════════════════════════════════ */}
        {activeTab === 'single' && (
          <div className={styles.playgroundGrid}>
            {/* Left Column: Transaction Form */}
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h3 className={styles.cardTitle}>
                  <span>⚙</span> Transaction Configuration
                </h3>
                <div className={styles.actionLeft}>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={() => setJsonMode(m => !m)}
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                  >
                    {jsonMode ? '⊞ Form View' : '{ } Raw JSON'}
                  </button>
                </div>
              </div>

              {/* Form Input Mode */}
              {!jsonMode ? (
                <div className={styles.formGrid}>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Transaction ID</label>
                    <input
                      className={styles.inputField}
                      value={form.transaction_id}
                      onChange={e => setForm(f => ({ ...f, transaction_id: e.target.value }))}
                    />
                  </div>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Amount</label>
                    <input
                      type="number"
                      step="any"
                      className={styles.inputField}
                      value={form.amount}
                      onChange={e => setForm(f => ({ ...f, amount: parseFloat(e.target.value) || 0 }))}
                    />
                  </div>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Originating Account (From)</label>
                    <input
                      className={styles.inputField}
                      value={form.from_account}
                      onChange={e => setForm(f => ({ ...f, from_account: e.target.value }))}
                      placeholder="e.g. 842E23DC0 (Redis) or ACC-USER-10"
                    />
                  </div>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Beneficiary Account (To)</label>
                    <input
                      className={styles.inputField}
                      value={form.to_account}
                      onChange={e => setForm(f => ({ ...f, to_account: e.target.value }))}
                      placeholder="e.g. ACC-BENEFICIARY-01"
                    />
                  </div>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Payment Format</label>
                    <select
                      className={styles.inputField}
                      value={form.payment_format}
                      onChange={e => setForm(f => ({ ...f, payment_format: e.target.value }))}
                    >
                      {PAYMENT_FORMATS.map(fmt => (
                        <option key={fmt} value={fmt}>
                          {fmt}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className={styles.inputGroup}>
                    <label className={styles.inputLabel}>Currency</label>
                    <input
                      className={styles.inputField}
                      value={form.currency}
                      onChange={e => setForm(f => ({ ...f, currency: e.target.value }))}
                    />
                  </div>
                  <div className={`${styles.inputGroup} ${styles.fullWidth}`}>
                    <label className={styles.inputLabel}>Timestamp (ISO 8601)</label>
                    <input
                      className={styles.inputField}
                      value={form.timestamp}
                      onChange={e => setForm(f => ({ ...f, timestamp: e.target.value }))}
                    />
                  </div>
                </div>
              ) : (
                <textarea
                  className={styles.jsonTextarea}
                  value={jsonText}
                  onChange={e => setJsonText(e.target.value)}
                  spellCheck={false}
                />
              )}

              {/* Form Action Controls */}
              <div className={styles.formActions}>
                <div className={styles.actionLeft}>
                  <button
                    type="button"
                    className={`${styles.actionBtn} ${styles.primaryBtn}`}
                    onClick={handleScore}
                    disabled={scoring}
                    id="btn-run-score"
                  >
                    {scoring ? '⚙ Scoring...' : '⚡ Score via AI Engine'}
                  </button>
                </div>
                {scoreResult && (
                  <button
                    type="button"
                    className={`${styles.actionBtn} ${styles.secondaryBtn}`}
                    onClick={exportAuditReport}
                  >
                    📥 Export Audit Report
                  </button>
                )}
              </div>
            </div>

            {/* Right Column: Deep Inference & SHAP Explainability View */}
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h3 className={styles.cardTitle}>
                  <span>📊</span> AI Engine Inference & Explainability
                </h3>
              </div>

              {!scoreResult && !scoring && (
                <div style={{ textAlign: 'center', padding: '3.5rem 1rem', color: '#64748b' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>▷</div>
                  <h4 style={{ color: '#94a3b8', marginBottom: '0.5rem' }}>Awaiting Transaction Score</h4>
                  <p style={{ fontSize: '0.8125rem' }}>
                    Enter transaction parameters on the left and click <strong>Score via AI Engine</strong> to trigger
                    dual-model inference (LightGBM + Autoencoder V6 + SHAP).
                  </p>
                </div>
              )}

              {scoring && (
                <div style={{ textAlign: 'center', padding: '4rem 1rem', color: '#94a3b8' }}>
                  <div style={{ fontSize: '2rem', marginBottom: '1rem', animation: 'spin 1s infinite linear' }}>↻</div>
                  <h4>Executing Inference on AI Engine (:8002)...</h4>
                  <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.5rem' }}>
                    Feature retrieval • LightGBM Booster • Autoencoder V6 • TreeExplainer
                  </p>
                </div>
              )}

              {scoreResult && !scoring && (
                <div>
                  {/* Decision Banner */}
                  <div
                    className={`${styles.decisionCard} ${scoreResult.decision_preliminary === 'BLOCK'
                      ? styles.decisionBlock
                      : scoreResult.decision_preliminary === 'REVIEW'
                        ? styles.decisionReview
                        : styles.decisionApprove
                      }`}
                  >
                    <div>
                      <span
                        className={`${styles.decisionBadge} ${scoreResult.decision_preliminary === 'BLOCK'
                          ? styles.badgeBlock
                          : scoreResult.decision_preliminary === 'REVIEW'
                            ? styles.badgeReview
                            : styles.badgeApprove
                          }`}
                      >
                        {scoreResult.decision_preliminary}
                      </span>
                      <div className={styles.decisionSubtext}>
                        {scoreResult.decision_preliminary === 'BLOCK'
                          ? 'Immediate Intervention Required (Risk ≥ 0.75)'
                          : scoreResult.decision_preliminary === 'REVIEW'
                            ? 'Manual AML Investigation Triggered (0.45 ≤ Risk < 0.75)'
                            : 'Transaction Cleared Low Risk (Risk < 0.45)'}
                      </div>
                    </div>

                    <div className={styles.decisionMetric}>
                      <div
                        className={`${styles.decisionScoreVal} ${scoreResult.decision_preliminary === 'BLOCK'
                          ? styles.scoreValBlock
                          : scoreResult.decision_preliminary === 'REVIEW'
                            ? styles.scoreValReview
                            : styles.scoreValApprove
                          }`}
                      >
                        {preliminaryScore}%
                      </div>
                      <div className={styles.decisionFormula}>0.70 × LGBM + 0.30 × AE</div>
                    </div>
                  </div>

                  {/* Dual Model Breakdown */}
                  <div className={styles.dualModelGrid}>
                    {/* Model 1: LightGBM */}
                    <div className={styles.modelBox}>
                      <div className={styles.modelBoxTitle}>
                        <span>LightGBM Classifier</span>
                        <span className={styles.modelTag}>Supervised</span>
                      </div>
                      <div className={styles.modelScoreRow}>
                        <span className={styles.modelScoreNumber}>
                          {(scoreResult.lgbm_score * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className={styles.meterTrack}>
                        <div
                          className={styles.meterFill}
                          style={{
                            width: `${Math.min(100, scoreResult.lgbm_score * 100)}%`,
                            background:
                              scoreResult.lgbm_score >= 0.75
                                ? '#ef4444'
                                : scoreResult.lgbm_score >= 0.45
                                  ? '#f59e0b'
                                  : '#10b981',
                          }}
                        />
                      </div>
                    </div>

                    {/* Model 2: Autoencoder V6 */}
                    <div className={styles.modelBox}>
                      <div className={styles.modelBoxTitle}>
                        <span>Autoencoder V6</span>
                        <span className={styles.modelTag}>Unsupervised</span>
                      </div>
                      <div className={styles.modelScoreRow}>
                        <span className={styles.modelScoreNumber}>
                          {(scoreResult.ae_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className={styles.meterTrack}>
                        <div
                          className={styles.meterFill}
                          style={{
                            width: `${Math.min(100, scoreResult.ae_score * 100)}%`,
                            background:
                              scoreResult.ae_score >= 0.75
                                ? '#ef4444'
                                : scoreResult.ae_score >= 0.45
                                  ? '#f59e0b'
                                  : '#38bdf8',
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* SHAP Waterfall / Feature Attribution */}
                  <div className={styles.shapSection}>
                    <div className={styles.shapHeader}>
                      <span className={styles.shapTitle}>SHAP Top 5 Feature Attribution</span>
                      <span className={styles.shapSubtitle}>TreeExplainer Local Attribution</span>
                    </div>

                    {(() => {
                      const riskFeatures = (scoreResult.shap_top_5 || [])
                        .filter(feat => feat.direction === 'RAISES_RISK')
                        .sort((a, b) => b.shap_contribution - a.shap_contribution)
                        .slice(0, 5);

                      if (riskFeatures.length === 0) {
                        return (
                          <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
                            No SHAP feature contributions calculated for this event.
                          </p>
                        );
                      }

                      return riskFeatures.map((feat, idx) => {
                        const absContrib = Math.min(1, Math.abs(feat.shap_contribution));
                        return (
                          <div key={idx} className={styles.shapRow}>
                            <span className={styles.shapFeatureName} title={feat.feature}>
                              {feat.feature}
                            </span>
                            <div className={styles.shapBarContainer}>
                              <div
                                className={`${styles.shapBar} ${styles.shapBarRisk}`}
                                style={{ width: `${Math.max(8, absContrib * 100)}%` }}
                              />
                            </div>
                            <span className={`${styles.shapValueBadge} ${styles.shapValRisk}`}>
                              +{feat.shap_contribution.toFixed(3)}
                            </span>
                          </div>
                        );
                      });
                    })()}
                  </div>

                  {/* Autoencoder Top Anomalies */}
                  <div className={styles.shapSection}>
                    <div className={styles.shapHeader}>
                      <span className={styles.shapTitle}>Autoencoder Top Anomalies</span>
                      <span className={styles.shapSubtitle}>Reconstruction Z-Score Breakdown</span>
                    </div>

                    {scoreResult.ae_top_5 && scoreResult.ae_top_5.length > 0 ? (
                      (() => {
                        const maxZ = Math.max(2.0, ...scoreResult.ae_top_5.map(f => f.reconstruction_z_score));

                        return scoreResult.ae_top_5.slice(0, 5).map((feat, idx) => {
                          const humanName: string = ('displayName' in feat && feat.displayName)
                            ? String(feat.displayName)
                            : feat.feature.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());

                          let barClass = styles.aeBarLow;
                          let valClass = styles.aeValLow;
                          if (feat.direction === 'RAISES_ANOMALY') {
                            barClass = styles.aeBarHigh;
                            valClass = styles.aeValHigh;
                          } else if (feat.direction === 'SMALL_ANOMALY_CONTRIBUTION') {
                            barClass = styles.aeBarMedium;
                            valClass = styles.aeValMedium;
                          }

                          const widthPct = Math.max(2, (feat.reconstruction_z_score / maxZ) * 100);

                          return (
                            <div key={idx} className={styles.shapRow}>
                              <span className={styles.shapFeatureName} title={feat.feature}>
                                {humanName}
                              </span>
                              <div className={styles.shapBarContainer}>
                                <div
                                  className={`${styles.shapBar} ${barClass}`}
                                  style={{ width: `${widthPct}%` }}
                                />
                              </div>
                              <span className={`${styles.shapValueBadge} ${valClass}`}>
                                {feat.reconstruction_z_score.toFixed(2)}
                              </span>
                            </div>
                          );
                        });
                      })()
                    ) : (
                      <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
                        No anomalous features detected.
                      </p>
                    )}
                  </div>

                  {/* Telemetry Stats */}
                  <div className={styles.telemetryGrid}>
                    <div className={styles.telemetryBox}>
                      <div className={styles.telemetryLabel}>Feature Source</div>
                      <div
                        className={styles.telemetryVal}
                        style={{
                          color: scoreResult.feature_source === 'redis' ? '#10b981' : '#f59e0b',
                        }}
                      >
                        {scoreResult.feature_source === 'redis' ? 'REDIS STORE' : 'ON-THE-FLY'}
                      </div>
                    </div>
                    <div className={styles.telemetryBox}>
                      <div className={styles.telemetryLabel}>Model Versions</div>
                      <div className={styles.telemetryVal} style={{ fontSize: '0.75rem' }}>
                        LGBM 2.0 / AE V6
                      </div>
                    </div>
                  </div>

                  {/* Pivot to Graph Explorer */}
                  {/* 
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '10px 14px',
                      background: 'rgba(56, 189, 248, 0.08)',
                      border: '1px solid rgba(56, 189, 248, 0.25)',
                      borderRadius: '8px',
                      marginTop: '12px',
                      marginBottom: '12px',
                      gap: '8px',
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>
                      Explore <strong>{form.from_account}</strong> in the TGN Graph Explorer
                    </div>
                    <Link
                      href={`/dashboard/explorer?account_id=${encodeURIComponent(form.from_account)}`}
                      className="btn btn-primary"
                      style={{ fontSize: '0.6875rem', padding: '4px 10px', textDecoration: 'none' }}
                    >
                      ◎ Open Subgraph →
                    </Link>
                  </div>
                  */}

                  {/* Developer Code Snippet & Copy */}
                  {/* 
                  <div className={styles.devStudio}>
                    <div className={styles.devHeader}>
                      <span className={styles.devTitle}>Direct Raw Output Inspector</span>
                      <button
                        type="button"
                        className={styles.secondaryBtn}
                        onClick={copyJsonResponse}
                        style={{ padding: '0.25rem 0.6rem', fontSize: '0.6875rem' }}
                      >
                        {copiedJson ? '✓ Copied' : 'Copy JSON'}
                      </button>
                    </div>
                    <pre className={styles.codeSnippet}>{JSON.stringify(scoreResult, null, 2)}</pre>
                  </div>
                  */}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════
            TAB 2: BATCH LOAD & STRESS TESTING BENCHMARK
           ═══════════════════════════════════════════════════ */}
        {activeTab === 'batch' && (
          <div>
            {/* Control Panel */}
            <div className={styles.batchConfigCard}>
              <div className={styles.batchControls}>
                <div className={styles.inputGroup} style={{ width: '160px' }}>
                  <label className={styles.inputLabel}>Batch Size (Txns)</label>
                  <select
                    className={styles.inputField}
                    value={batchCount}
                    onChange={e => setBatchCount(parseInt(e.target.value) || 25)}
                    disabled={batchRunning}
                  >
                    <option value={10}>10 Transactions</option>
                    <option value={25}>25 Transactions</option>
                    <option value={50}>50 Transactions</option>
                    <option value={100}>100 Transactions (Stress)</option>
                  </select>
                </div>

                <div className={styles.inputGroup} style={{ width: '220px' }}>
                  <label className={styles.inputLabel}>Traffic Profile</label>
                  <select
                    className={styles.inputField}
                    value={batchProfile}
                    onChange={e => setBatchProfile(e.target.value as any)}
                    disabled={batchRunning}
                  >
                    <option value="mixed">Realistic Mix (20% Mule)</option>
                    <option value="attack">Mule Attack Burst (65% Mule)</option>
                    <option value="legitimate">Legitimate Traffic (5% Mule)</option>
                  </select>
                </div>

                <button
                  type="button"
                  className={`${styles.actionBtn} ${styles.primaryBtn}`}
                  onClick={handleRunBatch}
                  disabled={batchRunning}
                  id="btn-launch-batch-stress"
                >
                  {batchRunning ? '⏳ Executing Benchmark...' : '🚀 Launch Stress Test'}
                </button>
              </div>
            </div>

            {/* KPI Cards */}
            {batchResult && (
              <div className={styles.batchKpiGrid}>
                <div className={styles.batchKpiCard}>
                  <div className={styles.kpiLabel}>Total Scored</div>
                  <div className={styles.kpiVal}>{batchResult.total_count}</div>
                  <div className={styles.kpiSub}>Conducted via :8002</div>
                </div>
                <div className={styles.batchKpiCard}>
                  <div className={styles.kpiLabel}>Average Latency</div>
                  <div className={styles.kpiVal} style={{ fontSize: '1.4rem' }}>
                    {batchResult.avg_latency_ms} ms
                  </div>
                  <div className={styles.kpiSub}>Target &lt; 200ms</div>
                </div>
                <div className={styles.batchKpiCard}>
                  <div className={styles.kpiLabel}>SLA Compliance</div>
                  <div
                    className={styles.kpiVal}
                    style={{
                      color: batchResult.sla_compliance_pct >= 95 ? '#10b981' : '#f59e0b',
                    }}
                  >
                    {batchResult.sla_compliance_pct}%
                  </div>
                  <div className={styles.kpiSub}>
                    {batchResult.sla_breach_count} breaches &gt; 200ms
                  </div>
                </div>
                <div className={styles.batchKpiCard}>
                  <div className={styles.kpiLabel}>Decisions Breakdown</div>
                  <div className={styles.kpiVal} style={{ fontSize: '1.15rem' }}>
                    <span style={{ color: '#ef4444' }}>{batchResult.blocked_count}B</span> /{' '}
                    <span style={{ color: '#f59e0b' }}>{batchResult.review_count}R</span> /{' '}
                    <span style={{ color: '#10b981' }}>{batchResult.approved_count}A</span>
                  </div>
                  <div className={styles.kpiSub}>
                    {((batchResult.blocked_count / batchResult.total_count) * 100).toFixed(0)}% Blocked
                  </div>
                </div>
                <div className={styles.batchKpiCard}>
                  <div className={styles.kpiLabel}>Redis Hit Ratio</div>
                  <div className={styles.kpiVal} style={{ color: '#10b981' }}>
                    {batchResult.redis_source_pct}%
                  </div>
                  <div className={styles.kpiSub}>Pre-cached Features</div>
                </div>
              </div>
            )}

            {/* Batch Execution Table */}
            {batchResult && (
              <div className={styles.tableContainer}>
                <div className={styles.tableHeader}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <h4 style={{ margin: 0, color: '#fff' }}>Execution Trace Log</h4>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Showing {filteredBatchItems.length} of {batchResult.items.length} events
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Filter:</span>
                    {(['ALL', 'BLOCK', 'REVIEW', 'APPROVE'] as const).map(dec => (
                      <button
                        key={dec}
                        type="button"
                        className={styles.secondaryBtn}
                        style={{
                          padding: '0.25rem 0.6rem',
                          fontSize: '0.6875rem',
                          background: filterDecision === dec ? '#1e2430' : undefined,
                          borderColor: filterDecision === dec ? '#3b82f6' : undefined,
                        }}
                        onClick={() => setFilterDecision(dec)}
                      >
                        {dec}
                      </button>
                    ))}
                  </div>
                </div>

                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Txn ID</th>
                        <th>From Account</th>
                        <th>Amount</th>
                        <th>Format</th>
                        <th>LGBM Score</th>
                        <th>AE Score (MSE)</th>
                        <th>Composite</th>
                        <th>Decision</th>
                        <th>Source</th>
                        <th>Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredBatchItems.map(item => (
                        <tr key={item.index} className={item.sla_breached ? styles.breachRow : undefined}>
                          <td style={{ fontFamily: 'monospace', color: '#64748b' }}>{item.index}</td>
                          <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{item.transaction_id}</td>
                          <td style={{ fontFamily: 'monospace' }}>{item.from_account}</td>
                          <td>${item.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                          <td>{item.payment_format}</td>
                          <td style={{ fontFamily: 'monospace' }}>{(item.lgbm_score * 100).toFixed(1)}%</td>
                          <td style={{ fontFamily: 'monospace' }}>
                            {(item.ae_score * 100).toFixed(1)}%{' '}
                            <span style={{ color: '#64748b', fontSize: '0.6875rem' }}>
                              ({item.ae_reconstruction_error})
                            </span>
                          </td>
                          <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                            {(item.combined_score * 100).toFixed(1)}%
                          </td>
                          <td>
                            <span
                              style={{
                                padding: '0.2rem 0.5rem',
                                borderRadius: '4px',
                                fontSize: '0.6875rem',
                                fontWeight: 700,
                                background:
                                  item.decision === 'BLOCK'
                                    ? 'rgba(239, 68, 68, 0.2)'
                                    : item.decision === 'REVIEW'
                                      ? 'rgba(245, 158, 11, 0.2)'
                                      : 'rgba(16, 185, 129, 0.2)',
                                color:
                                  item.decision === 'BLOCK'
                                    ? '#ef4444'
                                    : item.decision === 'REVIEW'
                                      ? '#f59e0b'
                                      : '#10b981',
                              }}
                            >
                              {item.decision}
                            </span>
                          </td>
                          <td style={{ fontSize: '0.75rem', fontFamily: 'monospace' }}>
                            {item.feature_source === 'redis' ? (
                              <span style={{ color: '#10b981' }}>redis</span>
                            ) : (
                              <span style={{ color: '#f59e0b' }}>on_the_fly</span>
                            )}
                          </td>
                          <td
                            className={item.sla_breached ? styles.slaBreach : styles.slaOk}
                            style={{ fontFamily: 'monospace' }}
                          >
                            {item.latency_ms}ms {item.sla_breached ? '⚠' : ''}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
