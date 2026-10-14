/**
 * /api/metrics — Model performance metrics proxy
 *
 * Sources (priority order):
 *   1. AI Engine GET /metrics (has lgbm + ae model versions, latency)
 *   2. Mock data enriched with real latency from health endpoints
 *
 * The AI Engine /metrics endpoint currently returns a Prometheus placeholder.
 * Until a real metrics API exists, we serve training-time metrics from the
 * mock data (which are the actual evaluated metrics from the ML training runs)
 * enriched with live latency from service health checks.
 */

import { NextResponse } from 'next/server';
import type { MetricsResponse } from '@/lib/types';
import { mockMetrics } from '@/lib/mock-data';

const AI_ENGINE_URL = process.env.AI_ENGINE_URL ?? 'http://127.0.0.1:8002';
const USE_MOCK      = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';

async function getLiveLatency(): Promise<number | null> {
  try {
    const t0 = Date.now();
    const res = await fetch(`${AI_ENGINE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return null;
    return Date.now() - t0;
  } catch {
    return null;
  }
}

export async function GET() {
  if (USE_MOCK) {
    return NextResponse.json({
      ...mockMetrics,
      total_scored_today: mockMetrics.total_scored_today + Math.floor(Math.random() * 1000),
      avg_latency_ms:     30 + Math.floor(Math.random() * 20),
      data_source: 'mock',
    });
  }

  // Try to get live latency from AI Engine health
  const liveLatency = await getLiveLatency();

  // Return training-time metrics (accurate from ML evaluation runs)
  // enriched with live latency if available
  const response: MetricsResponse & { data_source: string; ai_engine_reachable: boolean } = {
    ...mockMetrics,
    avg_latency_ms:     liveLatency ?? mockMetrics.avg_latency_ms,
    data_source:        liveLatency !== null ? 'live_latency' : 'training_metrics',
    ai_engine_reachable: liveLatency !== null,
  };

  return NextResponse.json(response);
}
