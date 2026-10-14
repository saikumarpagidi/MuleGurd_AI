'use client';

import React, { useState, useEffect, useCallback } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { fetchAlerts } from '@/lib/api';
import { formatAmount, formatRelativeTime, formatScore,
         getDecisionBgClass, getDecisionColor, getRingRoleColor } from '@/lib/utils';
import type { Alert, Decision } from '@/lib/types';
import styles from './page.module.css';

const DECISION_FILTERS: { label: string; value: Decision | 'ALL' }[] = [
  { label: 'All',    value: 'ALL'    },
  { label: 'Block',  value: 'BLOCK'  },
  { label: 'Review', value: 'REVIEW' },
  { label: 'Approve',value: 'APPROVE'},
];

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [total,  setTotal]  = useState(0);
  const [filter, setFilter] = useState<Decision | 'ALL'>('ALL');
  const [page,   setPage]   = useState(1);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    const res = await fetchAlerts({ decision: filter, page, limit: 15 });
    setAlerts(res.alerts);
    setTotal(res.total);
    setLoading(false);
  }, [filter, page]);

  useEffect(() => { load(); }, [load]);

  function handleFilterChange(f: Decision | 'ALL') {
    setFilter(f);
    setPage(1);
  }

  function exportCSV() {
    const header = 'id,transaction_id,account_id,decision,score,amount,payment_format,ring_role,timestamp\n';
    const rows = alerts.map(a =>
      `${a.id},${a.transaction_id},${a.account_id},${a.decision},${a.final_score},${a.amount},${a.payment_format},${a.ring_role},${a.timestamp}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `muleguard_alerts_${Date.now()}.csv`; a.click();
  }

  return (
    <DashboardLayout title="Alert Log" subtitle="REVIEW and BLOCK decisions · Live feed">
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.filters} id="alert-filters">
          {DECISION_FILTERS.map(f => (
            <button key={f.value} id={`filter-${f.value.toLowerCase()}`}
              className={`${styles.filterBtn} ${filter === f.value ? styles.active : ''}`}
              onClick={() => handleFilterChange(f.value)}>
              {f.label}
            </button>
          ))}
        </div>
        <div className={styles.toolbarRight}>
          <span className={styles.totalCount}>{total.toLocaleString()} total</span>
          <button className="btn btn-ghost" onClick={exportCSV} id="btn-export-csv"
            style={{ fontSize: '0.8125rem', height: 34 }}>
            ↓ Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className={`${styles.tableWrap} animate-fade-in-up`}>
        <table className="table" id="alerts-table">
          <thead>
            <tr>
              <th>Decision</th>
              <th>Transaction ID</th>
              <th>Account</th>
              <th>Ring Role</th>
              <th>Score</th>
              <th>Amount</th>
              <th>Format</th>
              <th>Time</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 10 }).map((_, i) => (
                <tr key={i}>
                  {Array.from({ length: 9 }).map((_, j) => (
                    <td key={j}><div className={`skeleton ${styles.cellSkel}`} /></td>
                  ))}
                </tr>
              ))
            ) : alerts.map(alert => (
              <React.Fragment key={alert.id}>
                <tr id={`alert-row-${alert.id}`}
                  className={`${styles.alertRow} ${expanded === alert.id ? styles.expanded : ''}`}
                  onClick={() => setExpanded(e => e === alert.id ? null : alert.id)}>
                  <td>
                    <span className={`badge ${getDecisionBgClass(alert.decision)}`}>
                      {alert.decision}
                    </span>
                  </td>
                  <td>
                    <span className={styles.txnId}>{alert.transaction_id}</span>
                  </td>
                  <td>
                    <span className={styles.accountId}>{alert.account_id}</span>
                  </td>
                  <td>
                    <span className={styles.roleLabel}
                      style={{ color: getRingRoleColor(alert.ring_role) }}>
                      {alert.ring_role}
                    </span>
                  </td>
                  <td>
                    <div className={styles.scoreCell}>
                      <div className={styles.scoreMini}
                        style={{
                          '--fill': `${alert.final_score * 100}%`,
                          '--color': getDecisionColor(alert.decision),
                        } as React.CSSProperties}
                      />
                      <span style={{ color: getDecisionColor(alert.decision), fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.8125rem' }}>
                        {(alert.final_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className={styles.mono}>{formatAmount(alert.amount)}</td>
                  <td>
                    <span className={styles.format}>{alert.payment_format}</span>
                  </td>
                  <td className={styles.timeCell}>{formatRelativeTime(alert.timestamp)}</td>
                  <td>
                    <span className={`${styles.ackBadge} ${alert.acknowledged ? styles.acked : ''}`}>
                      {alert.acknowledged ? 'Acknowledged' : 'Open'}
                    </span>
                  </td>
                </tr>
                {expanded === alert.id && (
                  <tr key={`${alert.id}-exp`} className={styles.expandedRow}>
                    <td colSpan={9}>
                      <div className={styles.expandedContent}>
                        <div className={styles.expSection}>
                          <span className={styles.expLabel}>Explanation</span>
                          <p className={styles.expText}>{alert.explanation_summary}</p>
                        </div>
                        <div className={styles.expScores}>
                          {[
                            { name: 'TGN',        val: alert.component_scores.tgn_score,  color: '#7C3AED' },
                            { name: 'LightGBM',   val: alert.component_scores.lgbm_score, color: '#00D4FF' },
                            { name: 'Autoencoder',val: alert.component_scores.ae_score,   color: '#F59E0B' },
                          ].map(({ name, val, color }) => (
                            <div key={name} className={styles.expScore}>
                              <span className={styles.expScoreName}>{name}</span>
                              <span className={styles.expScoreVal} style={{ color }}>
                                {(val * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className={styles.pagination} id="alerts-pagination">
        <button className="btn btn-ghost" onClick={() => setPage(p => Math.max(1, p - 1))}
          disabled={page === 1} id="btn-prev-page">
          ← Prev
        </button>
        <span className={styles.pageInfo}>Page {page}</span>
        <button className="btn btn-ghost" onClick={() => setPage(p => p + 1)}
          disabled={alerts.length < 15} id="btn-next-page">
          Next →
        </button>
      </div>
    </DashboardLayout>
  );
}
