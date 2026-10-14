'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import AUCChart from '@/components/charts/AUCChart';
import { fetchMetrics } from '@/lib/api';
import type { MetricsResponse, ModelMetrics } from '@/lib/types';
import styles from './page.module.css';

type ModelKey = 'lgbm' | 'autoencoder' | 'tgn' | 'ensemble';

const MODEL_TABS: { key: ModelKey; label: string; color: string }[] = [
  { key: 'lgbm',        label: 'LightGBM',    color: '#00D4FF' },
  { key: 'autoencoder', label: 'Autoencoder', color: '#F59E0B' },
  { key: 'tgn',         label: 'TGN',          color: '#7C3AED' },
  { key: 'ensemble',    label: 'Ensemble',     color: '#10B981' },
];

function ConfusionMatrix({ data }: { data: ModelMetrics['confusion_matrix'] }) {
  const total = data.true_positive + data.false_positive + data.true_negative + data.false_negative;
  const cells = [
    { label: 'True Positive',  value: data.true_positive,  color: '#10B981', sub: 'Correctly blocked mules' },
    { label: 'False Positive', value: data.false_positive, color: '#F59E0B', sub: 'Legitimate flagged wrong' },
    { label: 'False Negative', value: data.false_negative, color: '#EF4444', sub: 'Missed mules' },
    { label: 'True Negative',  value: data.true_negative,  color: '#00D4FF', sub: 'Correctly approved' },
  ];
  return (
    <div className={styles.cmGrid} id="confusion-matrix">
      {cells.map(cell => (
        <div key={cell.label} className={styles.cmCell}
          style={{ '--cell-color': cell.color } as React.CSSProperties}>
          <span className={styles.cmLabel}>{cell.label}</span>
          <span className={styles.cmValue} style={{ color: cell.color }}>
            {cell.value.toLocaleString()}
          </span>
          <span className={styles.cmPct}>{((cell.value / total) * 100).toFixed(1)}%</span>
          <span className={styles.cmSub}>{cell.sub}</span>
        </div>
      ))}
    </div>
  );
}

function FeatureImportance({ features }: { features: ModelMetrics['top_features'] }) {
  const maxImp = Math.max(...features.map(f => f.importance));
  return (
    <div className={styles.featList} id="feature-importance-chart">
      {features.map((f, i) => (
        <div key={f.feature} className={styles.featRow}
          style={{ animationDelay: `${i * 50}ms` }}>
          <span className={styles.featName}>{f.feature.replace(/_/g, ' ')}</span>
          <div className={styles.featBar}>
            <div
              className={styles.featFill}
              style={{ '--fill': `${(f.importance / maxImp) * 100}%` } as React.CSSProperties}
            />
          </div>
          <span className={styles.featValue}>{(f.importance * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [activeTab, setActiveTab] = useState<ModelKey>('lgbm');

  useEffect(() => {
    fetchMetrics().then(setMetrics);
  }, []);

  const model: ModelMetrics | null = metrics ? metrics[activeTab] : null;

  return (
    <DashboardLayout
      title="Model Metrics"
      subtitle="Out-Of-Time (OOT) evaluation benchmark on 6,379,648 held-out transactions"
    >
      {/* Summary KPIs */}
      {metrics && (
        <div className={`${styles.summaryRow} animate-fade-in-up`}>
          <div className={styles.kpi} id="kpi-total-scored">
            <span className={styles.kpiLabel}>Evaluated Population</span>
            <span className={styles.kpiValue}>{(metrics.total_scored_today / 1_000_000).toFixed(2)}M</span>
          </div>
          <div className={styles.kpi} id="kpi-block-rate">
            <span className={styles.kpiLabel}>Block Rate</span>
            <span className={styles.kpiValue} style={{ color: 'var(--block)' }}>
              {(metrics.block_rate * 100).toFixed(1)}%
            </span>
          </div>
          <div className={styles.kpi} id="kpi-review-rate">
            <span className={styles.kpiLabel}>Review Rate</span>
            <span className={styles.kpiValue} style={{ color: 'var(--review)' }}>
              {(metrics.review_rate * 100).toFixed(1)}%
            </span>
          </div>
          <div className={styles.kpi} id="kpi-approve-rate">
            <span className={styles.kpiLabel}>Approve Rate</span>
            <span className={styles.kpiValue} style={{ color: 'var(--approve)' }}>
              {(metrics.approve_rate * 100).toFixed(1)}%
            </span>
          </div>
          <div className={styles.kpi} id="kpi-avg-latency">
            <span className={styles.kpiLabel}>Avg Latency</span>
            <span className={styles.kpiValue}>{metrics.avg_latency_ms}ms</span>
          </div>
        </div>
      )}

      {/* Model Tabs */}
      <div className="tabs" id="model-tabs">
        {MODEL_TABS.map(tab => (
          <button key={tab.key} id={`tab-${tab.key}`}
            className={`tab-btn ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
            style={activeTab === tab.key ? { color: tab.color, borderBottomColor: tab.color } : {}}>
            {tab.label}
            {metrics && (
              <span style={{
                marginLeft: 6, fontFamily: 'var(--font-mono)', fontSize: '0.625rem',
                color: activeTab === tab.key ? tab.color : 'var(--text-faint)',
              }}>
                {metrics[tab.key].roc_auc.toFixed(4)}
              </span>
            )}
          </button>
        ))}
      </div>

      {model ? (
        <div className={`${styles.metricsGrid} animate-fade-in`}>
          {/* AUC Chart */}
          <div className={styles.chartCard} id="auc-chart-card">
            <div className={styles.chartHeader}>
              <span className={styles.chartTitle}>ROC Curve</span>
              <span className={styles.aucBadge}
                style={{ color: MODEL_TABS.find(t => t.key === activeTab)?.color }}>
                AUC = {model.roc_auc.toFixed(4)}
              </span>
            </div>
            <AUCChart
              rocCurve={model.roc_curve}
              prCurve={model.pr_curve}
              rocAuc={model.roc_auc}
              prAuc={model.pr_auc}
              modelName={model.model_name}
              activeTab="roc"
            />
          </div>

          {/* Confusion Matrix */}
          <div className={styles.chartCard} id="confusion-matrix-card">
            <div className={styles.chartHeader}>
              <span className={styles.chartTitle}>Confusion Matrix</span>
              <span className={styles.chartSubtitle}>6.38M OOT test population</span>
            </div>
            <ConfusionMatrix data={model.confusion_matrix} />

            {/* Threshold info */}
            <div className={styles.thresholdRow}>
              <div className={styles.threshInfo}>
                <span className={styles.threshLabel}>Threshold</span>
                <span className={styles.threshValue}>{model.threshold}</span>
              </div>
              <div className={styles.threshInfo}>
                <span className={styles.threshLabel}>Recall @ Threshold</span>
                <span className={styles.threshValue} style={{ color: 'var(--approve)' }}>
                  {(model.recall_at_threshold * 100).toFixed(1)}%
                </span>
              </div>
              <div className={styles.threshInfo}>
                <span className={styles.threshLabel}>Precision @ Threshold</span>
                <span className={styles.threshValue} style={{ color: 'var(--cyan)' }}>
                  {(model.precision_at_threshold * 100).toFixed(1)}%
                </span>
              </div>
              <div className={styles.threshInfo}>
                <span className={styles.threshLabel}>PR-AUC</span>
                <span className={styles.threshValue}>{model.pr_auc.toFixed(4)}</span>
              </div>
            </div>
          </div>

          {/* Feature Importance */}
          <div className={`${styles.chartCard} ${styles.fullWidth}`} id="feature-importance-card">
            <div className={styles.chartHeader}>
              <span className={styles.chartTitle}>Feature Importance (SHAP Global)</span>
              <span className={styles.chartSubtitle}>Top 10 features</span>
            </div>
            <FeatureImportance features={model.top_features} />
          </div>
        </div>
      ) : (
        <div className={styles.loading}>
          <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
          <span>Loading metrics…</span>
        </div>
      )}
    </DashboardLayout>
  );
}
