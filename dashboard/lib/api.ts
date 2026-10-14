/* ============================================================
   API Client — MuleGuard.AI Dashboard
   Handles mock ↔ live switching via env var
   ============================================================ */

import type {
  AlertFilters,
  AlertsResponse,
  GraphResponse,
  LiveStats,
  MetricsResponse,
  ScoreResponse,
  ServiceHealth,
  TransactionInput,
} from './types';

import {
  mockAlertsResponse,
  mockGraphResponse,
  mockLiveStats,
  mockMetrics,
  mockScoreApprove,
  mockScoreBlock,
  mockServiceHealth,
} from './mock-data';

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? '/api';

async function fetchWithFallback<T>(url: string, mock: T, options?: RequestInit, timeoutMs = 8000): Promise<T> {
  if (USE_MOCK) {
    await new Promise(r => setTimeout(r, 300 + Math.random() * 400));
    return mock;
  }
  try {
    const res = await fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json() as Promise<T>;
  } catch (e) {
    console.warn(`[MuleGuard API] Falling back to mock for ${url}`, e);
    return mock;
  }
}

// ─── Score a Transaction ───────────────────────────────────
export async function scoreTransaction(input: TransactionInput): Promise<ScoreResponse> {
  const mock = input.amount > 5000 ? mockScoreBlock : mockScoreApprove;
  return fetchWithFallback<ScoreResponse>(
    `${API_BASE}/score`,
    mock,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }
  );
}

// ─── Fetch Alerts ──────────────────────────────────────────
export async function fetchAlerts(filters: AlertFilters = {}): Promise<AlertsResponse> {
  const params = new URLSearchParams(
    Object.entries(filters)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => [k, String(v)])
  );
  return fetchWithFallback<AlertsResponse>(
    `${API_BASE}/alerts?${params}`,
    mockAlertsResponse
  );
}

// ─── Fetch Graph Data ──────────────────────────────────────
export async function fetchGraphData(accountId: string, limit = 60, tgnScoreOverride?: string | null): Promise<GraphResponse> {
  const url = `${API_BASE}/graph?account_id=${encodeURIComponent(accountId)}&limit=${limit}${tgnScoreOverride ? `&tgn_score=${encodeURIComponent(tgnScoreOverride)}` : ''}`;
  return fetchWithFallback<GraphResponse>(
    url,
    { ...mockGraphResponse, account_id: accountId },
    undefined,
    15000
  );
}

// ─── Fetch Model Metrics ───────────────────────────────────
export async function fetchMetrics(): Promise<MetricsResponse> {
  return fetchWithFallback<MetricsResponse>(`${API_BASE}/metrics`, mockMetrics);
}

// ─── Fetch Live Stats ──────────────────────────────────────
export async function fetchLiveStats(): Promise<LiveStats> {
  const data = await fetchWithFallback<{ stats: LiveStats }>(`${API_BASE}/live-stats`, {
    stats: {
      ...mockLiveStats,
      transactions_per_second: 800 + Math.floor(Math.random() * 200),
      avg_latency_ms: 30 + Math.floor(Math.random() * 20),
    }
  });
  return data.stats;
}

// ─── Fetch Service Health ──────────────────────────────────
export async function fetchServiceHealth(): Promise<ServiceHealth[]> {
  const data = await fetchWithFallback<{ services: ServiceHealth[] }>(`${API_BASE}/live-stats`, {
    services: mockServiceHealth
  });
  return data.services;
}
