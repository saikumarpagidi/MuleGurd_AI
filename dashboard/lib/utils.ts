/* ============================================================
   Utility Functions — MuleGuard.AI Dashboard
   ============================================================ */

import type { Decision, RingRole } from './types';

// ─── Number Formatting ─────────────────────────────────────
export function formatScore(score: number): string {
  return (score * 100).toFixed(1) + '%';
}

export function formatAmount(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(2)}M`;
  if (amount >= 1_000)     return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(2)}`;
}

export function formatLatency(ms: number): string {
  if (ms < 1)    return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatNumber(n: number | undefined | null): string {
  if (n == null || isNaN(n)) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

// ─── Date / Time Formatting ────────────────────────────────
export function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export function formatDate(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
  });
}

export function formatRelativeTime(ts: string): string {
  const diffMs  = Date.now() - new Date(ts).getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60)  return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return formatDate(ts);
}

// ─── Decision Helpers ──────────────────────────────────────
export function getDecisionColor(decision: Decision): string {
  switch (decision) {
    case 'APPROVE': return '#10B981';
    case 'REVIEW':  return '#F59E0B';
    case 'BLOCK':   return '#EF4444';
  }
}

export function getDecisionBgClass(decision: Decision): string {
  switch (decision) {
    case 'APPROVE': return 'badge-approve';
    case 'REVIEW':  return 'badge-review';
    case 'BLOCK':   return 'badge-block';
  }
}

export function getDecisionFromScore(score: number): Decision {
  if (score < 0.45) return 'APPROVE';
  if (score < 0.75) return 'REVIEW';
  return 'BLOCK';
}

// ─── Risk Score Color Scale ────────────────────────────────
export function getRiskColor(score: number): string {
  if (score < 0.33) return '#10B981'; // green
  if (score < 0.55) return '#22C55E'; // light green
  if (score < 0.65) return '#F59E0B'; // amber
  if (score < 0.75) return '#F97316'; // orange
  return '#EF4444';                   // red
}

export function getRiskGradient(score: number): string {
  return `linear-gradient(90deg, ${getRiskColor(0)} 0%, ${getRiskColor(score)} ${score * 100}%)`;
}

// ─── Ring Role Helpers ─────────────────────────────────────
export function getRingRoleColor(role: RingRole): string {
  switch (role) {
    case 'HUB':    return '#EF4444';
    case 'BRIDGE': return '#F59E0B';
    case 'MULE':   return '#EC4899';
    case 'NONE':   return '#00D4FF';
  }
}

export function getRingRoleIcon(role: RingRole): string {
  switch (role) {
    case 'HUB':    return '⬟';
    case 'BRIDGE': return '⬡';
    case 'MULE':   return '⬢';
    case 'NONE':   return '○';
  }
}

// ─── Graph Node Sizing ────────────────────────────────────
export function getNodeRadius(role: RingRole, pagerank: number): number {
  const base = { HUB: 16, BRIDGE: 12, MULE: 10, NONE: 7 }[role] ?? 8;
  // PageRank in Neo4j GDS scales with degree/damping and can reach 200–300+ on major hubs.
  // Linear scaling (pagerank * 8) blows up canvas (e.g. 284 PR -> 2,295px radius).
  // Using logarithmic scaling ensures hubs are visually prominent without overflowing.
  const pr = Math.max(0, pagerank || 0);
  const prBoost = Math.log1p(pr) * 3.5;
  const radius = base + prBoost;
  return Math.min(Math.max(radius, 6), 34);
}

// ─── SHAP Direction Color ─────────────────────────────────
export function getSHAPColor(direction: 'RAISES_RISK' | 'LOWERS_RISK'): string {
  return direction === 'RAISES_RISK' ? '#EF4444' : '#00D4FF';
}

// ─── Clamp ────────────────────────────────────────────────
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Random ID ────────────────────────────────────────────
export function generateId(prefix = 'txn'): string {
  return `${prefix}_${Math.random().toString(36).substring(2, 11).toUpperCase()}`;
}

// ─── Payment Format Label ─────────────────────────────────
export function formatPaymentMethod(format: string): string {
  const map: Record<string, string> = {
    WIRE: 'Wire Transfer',
    ACH: 'ACH',
    CASH: 'Cash',
    BITCOIN: 'Bitcoin',
    CRYPTO: 'Crypto',
    CHEQUE: 'Cheque',
    CREDIT_CARD: 'Credit Card',
    REINVESTMENT: 'Reinvestment',
  };
  return map[format.toUpperCase()] ?? format;
}

// ─── Service Name Helpers ─────────────────────────────────
export function getServiceDisplayName(service: string): string {
  const map: Record<string, string> = {
    'ai-engine':            'AI Engine',
    'graph-service':        'Graph Service',
    'scoring-orchestrator': 'Orchestrator',
    'explainability-service': 'Explainability',
    'feature-service':      'Feature Store',
    'transaction-ingest':   'Ingest Service',
  };
  return map[service] ?? service;
}
