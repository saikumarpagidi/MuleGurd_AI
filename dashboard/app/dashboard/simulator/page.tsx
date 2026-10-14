'use client';

import Link from 'next/link';
import { useState } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import TransactionForm from '@/components/simulator/TransactionForm';
import RiskFusionGauge from '@/components/charts/RiskFusionGauge';

import { scoreTransaction } from '@/lib/api';
import { formatAmount, formatLatency, formatScore, getDecisionBgClass, getDecisionColor, getRingRoleColor } from '@/lib/utils';
import type { ScoreResponse, TransactionInput } from '@/lib/types';
import styles from './page.module.css';

export default function SimulatorPage() {
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [lastScoredAccount, setLastScoredAccount] = useState<string>('');

  async function handleScore(input: TransactionInput) {
    setLoading(true);
    setSubmitted(false);
    setLastScoredAccount(input.from_account);
    try {
      const res = await scoreTransaction(input);
      setResult(res);
      setSubmitted(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <DashboardLayout
      title="Live Scoring Simulator"
      subtitle="Submit any transaction → see real-time AI decision with SHAP explanation"
    >
      {/* Engine Switcher Banner */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem', alignItems: 'center' }}>
        <Link
          href="/dashboard/ai-engine-simulator"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            background: 'rgba(59, 130, 246, 0.12)',
            border: '1px solid rgba(59, 130, 246, 0.4)',
            color: '#38bdf8',
            fontSize: '0.8125rem',
            fontWeight: 600,
            textDecoration: 'none',
          }}
        >
          <span>⚡</span> Switch to AI Engine Standalone Studio (:8002) →
        </Link>
      </div>

      <div className={styles.layout}>
        {/* Left: Form */}
        <div className={styles.formCol}>
          <TransactionForm onSubmit={handleScore} isLoading={loading} />
        </div>

        {/* Right: Result */}
        <div className={styles.resultCol}>
          {!submitted && !loading && (
            <div className={styles.emptyState} id="simulator-empty">
              <div className={styles.emptyIcon}>▷</div>
              <p className={styles.emptyTitle}>Ready to Score</p>
              <p className={styles.emptySubtitle}>
                Fill in the transaction form and click Score Transaction.<br />
                The AI ensemble will return a decision.
              </p>
            </div>
          )}

          {loading && (
            <div className={styles.loadingState} id="simulator-loading">
              <div className={styles.loadingSpinner}>
                <div className={styles.spinRing} />
                <div className={styles.spinCore} />
              </div>
              <p className={styles.loadingText}>Scoring transaction…</p>
              <p className={styles.loadingSubtext}>LightGBM + Autoencoder + TGN running in parallel</p>
            </div>
          )}

          {submitted && result && (
            <div className={`${styles.resultPanel} animate-scale-in`} id="simulator-result">
              {/* Decision Banner */}
              <div
                className={styles.decisionBanner}
                style={{
                  background: `${getDecisionColor(result.decision)}14`,
                  borderColor: `${getDecisionColor(result.decision)}40`,
                }}
              >
                <div>
                  <span className={`badge ${getDecisionBgClass(result.decision)}`}
                    style={{ fontSize: '0.875rem', padding: '5px 16px' }}>
                    {result.decision}
                  </span>
                  <p className={styles.decisionScore}
                    style={{ color: getDecisionColor(result.decision) }}>
                    {(result.final_score * 100).toFixed(1)}% Risk Score
                  </p>
                </div>
                <div className={styles.decisionMeta}>
                  {result.is_mule_probability != null && (
                    <div className={styles.metaItem}>
                      <span className={styles.metaLabel}>Mule Risk</span>
                      <span className={styles.metaValue} style={{ color: result.is_mule_probability > 0.3 ? 'var(--block)' : 'inherit' }}>
                        {(result.is_mule_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Graph Neural Network Subgraph Deep Dive Card */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '12px 16px',
                  background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(56, 189, 248, 0.1))',
                  border: '1px solid rgba(124, 58, 237, 0.35)',
                  borderRadius: 'var(--radius-md)',
                  marginBottom: '16px',
                  gap: '12px',
                  flexWrap: 'wrap',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '1.25rem', color: '#a78bfa' }}>◎</span>
                  <div>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#f8fafc' }}>
                      Neo4j Subgraph &amp; TGN Inference Ready
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      Account: <code style={{ color: '#38bdf8' }}>{lastScoredAccount}</code> • TGN Risk: {formatScore(result.component_scores.tgn_score || 0)}
                      {result.is_mule_probability != null && (
                        <span> • Mule: <strong style={{ color: result.is_mule_probability > 0.3 ? '#ef4444' : '#10b981' }}>{(result.is_mule_probability * 100).toFixed(1)}%</strong></span>
                      )}
                    </div>
                  </div>
                </div>
                <Link
                  href={`/dashboard/explorer?account_id=${encodeURIComponent(lastScoredAccount)}&tgn_score=${result.component_scores.tgn_score || 0}`}
                  className="btn btn-primary"
                  style={{
                    fontSize: '0.75rem',
                    padding: '6px 14px',
                    textDecoration: 'none',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  Explore 2-Hop Network →
                </Link>
              </div>

              {/* Degraded Mode Warning */}
              {result.degraded_mode && (
                <div className={styles.degradedBanner}>
                  ⚠ Graph service running in heuristic mode — TGN score approximated.
                  Fusion weights renormalized to LightGBM (67%) + Autoencoder (33%).
                </div>
              )}

              {/* Risk Gauge */}
              <div className={styles.gaugeSection}>
                <RiskFusionGauge
                  finalScore={result.final_score}
                  componentScores={result.component_scores}
                  size={200}
                  animated
                />
              </div>

              {/* SHAP Feature Attribution */}
              <div className={styles.shapSection}>
                <div className={styles.shapHeader}>
                  <span className={styles.shapTitle}>SHAP Top 5 Feature Attribution</span>
                  <span className={styles.shapSubtitle}>TreeExplainer Local Attribution</span>
                </div>

                {(() => {
                  const riskFeatures = (result.shap_top_5 || [])
                    .filter(feat => feat.direction === 'RAISES_RISK')
                    .sort((a, b) => b.contribution - a.contribution)
                    .slice(0, 5);

                  if (riskFeatures.length === 0) {
                    return (
                      <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
                        No SHAP feature contributions calculated for this event.
                      </p>
                    );
                  }

                  return riskFeatures.map((feat, idx) => {
                    const absContrib = Math.min(1, Math.abs(feat.contribution));
                    const humanName = feat.displayName || feat.feature;
                    return (
                      <div key={idx} className={styles.shapRow}>
                        <span className={styles.shapFeatureName} title={feat.feature}>
                          {humanName}
                        </span>
                        <div className={styles.shapBarContainer}>
                          <div
                            className={`${styles.shapBar} ${styles.shapBarRisk}`}
                            style={{ width: `${Math.max(8, absContrib * 100)}%` }}
                          />
                        </div>
                        <span className={`${styles.shapValueBadge} ${styles.shapValRisk}`}>
                          +{feat.contribution.toFixed(3)}
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

                {result.ae_top_5 && result.ae_top_5.length > 0 ? (
                  (() => {
                    const maxZ = Math.max(2.0, ...result.ae_top_5.map(f => f.reconstruction_z_score));

                    return result.ae_top_5.slice(0, 5).map((feat, idx) => {
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

              {/* TGN Graph Structural Attributes */}
              {!result.degraded_mode && (
                <div className={styles.shapSection}>
                  <div className={styles.shapHeader}>
                    <span className={styles.shapTitle}>TGN Graph Structural Attributes</span>
                    <span className={styles.shapSubtitle}>Neo4j GDS Subgraph & TGN Node Metrics</span>
                  </div>

                  {result.is_mule_probability != null && (
                    <div className={styles.shapRow}>
                      <span className={styles.shapFeatureName}>Mule Account Probability</span>
                      <div className={styles.shapBarContainer}>
                        <div
                          className={`${styles.shapBar} ${result.is_mule_probability > 0.3 ? styles.shapBarRisk : styles.aeBarLow}`}
                          style={{ width: `${Math.max(8, result.is_mule_probability * 100)}%` }}
                        />
                      </div>
                      <span className={`${styles.shapValueBadge} ${result.is_mule_probability > 0.3 ? styles.shapValRisk : styles.aeValLow}`} style={{ width: '60px', textAlign: 'right' }}>
                        {(result.is_mule_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  <div className={styles.shapRow}>
                    <span className={styles.shapFeatureName}>Detected Ring Role</span>
                    <div className={styles.shapBarContainer} style={{ background: 'transparent' }}></div>
                    <span className={styles.shapValueBadge} style={{ color: getRingRoleColor(result.ring_role), width: 'auto', paddingLeft: '1rem' }}>
                      {result.ring_role} ({(result.ring_probability * 100).toFixed(1)}%)
                    </span>
                  </div>
                  
                  <div className={styles.shapRow}>
                    <span className={styles.shapFeatureName}>Strongly Connected Component</span>
                    <div className={styles.shapBarContainer} style={{ background: 'transparent' }}></div>
                    <span className={styles.shapValueBadge} style={{ color: result.in_scc_ring ? '#ef4444' : '#10b981', width: 'auto', paddingLeft: '1rem' }}>
                      {result.in_scc_ring ? '✓ TRUE (High Risk)' : 'FALSE'}
                    </span>
                  </div>
                  
                  <div className={styles.shapRow}>
                    <span className={styles.shapFeatureName}>PageRank Centrality</span>
                    <div className={styles.shapBarContainer} style={{ background: 'transparent' }}></div>
                    <span className={styles.shapValueBadge} style={{ color: '#cbd5e1', width: 'auto', paddingLeft: '1rem' }}>
                      {result.pagerank.toFixed(5)}
                    </span>
                  </div>

                  <div className={styles.shapRow}>
                    <span className={styles.shapFeatureName}>Louvain Community ID</span>
                    <div className={styles.shapBarContainer} style={{ background: 'transparent' }}></div>
                    <span className={styles.shapValueBadge} style={{ color: '#cbd5e1', width: 'auto', paddingLeft: '1rem' }}>
                      {result.community_id >= 0 ? `#${result.community_id}` : 'UNASSIGNED'}
                    </span>
                  </div>
                </div>
              )}


            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
