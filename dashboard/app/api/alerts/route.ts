/**
 * /api/alerts — Dashboard Alert Log proxy
 *
 * Strategy (with graceful degradation):
 *   1. Try Postgres scoring_decisions table directly (fastest, richest data)
 *   2. Fallback: try scoring-orchestrator REST /api/v1/decisions
 *   3. Final fallback: return mock data with a degraded_mode flag
 *
 * Postgres schema (from infrastructure/postgres/schema.sql):
 *   scoring_decisions(id, transaction_id, from_account, to_account, amount,
 *     payment_format, final_score, lgbm_score, ae_score, tgn_score,
 *     decision, ring_role, ring_flag, community_id, explanation_text,
 *     acknowledged, created_at)
 */

import { NextResponse } from 'next/server';
import type { Alert, AlertsResponse, ComponentScores, Decision, RingRole } from '@/lib/types';
import { mockAlertsResponse } from '@/lib/mock-data';

const USE_MOCK          = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';
const ORCHESTRATOR_URL  = process.env.SCORING_ORCHESTRATOR_URL ?? 'http://127.0.0.1:8080';

// Postgres connection via pg (server-side only)
const POSTGRES_CONFIG = {
  host:     process.env.POSTGRES_HOST     ?? 'localhost',
  port:     parseInt(process.env.POSTGRES_PORT ?? '5433'),
  database: process.env.POSTGRES_DB       ?? 'muleguard',
  user:     process.env.POSTGRES_USER     ?? 'muleguard',
  password: process.env.POSTGRES_PASSWORD ?? 'muleguard_local_dev',
};

async function queryPostgres(
  decision: string,
  page: number,
  limit: number
): Promise<AlertsResponse | null> {
  try {
    // Dynamic import to keep pg server-side only
    const { Pool } = await import('pg');
    const pool = new Pool({ ...POSTGRES_CONFIG, connectionTimeoutMillis: 3000, idleTimeoutMillis: 5000 });
    const client = await pool.connect();

    try {
      const decisionClause = decision !== 'ALL' ? `WHERE decision = $3` : '';
      const params: (string | number)[] = [limit, (page - 1) * limit];
      if (decision !== 'ALL') params.push(decision);

      const [rowsResult, countResult] = await Promise.all([
        client.query(
          `SELECT id, transaction_id, source_token as from_account, destination_token as to_account, amount, channel as payment_format,
                  final_risk_score as final_score, lgbm_score, anomaly_score as ae_score, tgn_score,
                  decision, ring_role, ring_flag, 
                  explanation as explanation_text, false as acknowledged, scored_at as created_at
           FROM scoring_decisions
           ${decisionClause}
           ORDER BY scored_at DESC
           LIMIT $1 OFFSET $2`,
          params
        ),
        client.query(
          `SELECT COUNT(*) as total FROM scoring_decisions ${decisionClause}`,
          decision !== 'ALL' ? [decision] : []
        ),
      ]);

      const total = parseInt(countResult.rows[0]?.total ?? 0);
      const alerts: Alert[] = rowsResult.rows.map(row => ({
        id:              String(row.id),
        transaction_id:  row.transaction_id,
        account_id:      row.from_account,
        decision:        row.decision as Decision,
        final_score:     parseFloat(row.final_score),
        component_scores: {
          lgbm_score: parseFloat(row.lgbm_score ?? 0),
          ae_score:   parseFloat(row.ae_score   ?? 0),
          tgn_score:  parseFloat(row.tgn_score  ?? 0),
        } as ComponentScores,
        ring_role:           (row.ring_role ?? 'NONE') as RingRole,
        amount:              parseFloat(row.amount),
        payment_format:      row.payment_format ?? 'OTHER',
        timestamp:           new Date(row.created_at).toISOString(),
        explanation_summary: row.explanation_text ?? '',
        acknowledged:        Boolean(row.acknowledged),
      }));

      await client.release();
      await pool.end();

      return {
        alerts,
        total,
        page,
        limit,
        total_pages: Math.ceil(total / limit),
      };
    } finally {
      client.release();
      await pool.end().catch(() => {});
    }
  } catch (err) {
    console.warn('[/api/alerts] Postgres unavailable:', (err as Error).message);
    return null;
  }
}

async function queryOrchestrator(
  decision: string,
  page: number,
  limit: number
): Promise<AlertsResponse | null> {
  try {
    const params = new URLSearchParams({
      page:  String(page),
      limit: String(limit),
      ...(decision !== 'ALL' ? { decision } : {}),
    });
    const res = await fetch(`${ORCHESTRATOR_URL}/api/v1/decisions?${params}`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    // Map orchestrator format if needed
    return data as AlertsResponse;
  } catch (err) {
    console.warn('[/api/alerts] Orchestrator unavailable:', (err as Error).message);
    return null;
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const page     = Math.max(1, Number(searchParams.get('page')  ?? 1));
  const limit    = Math.min(100, Math.max(1, Number(searchParams.get('limit') ?? 15)));
  const decision = (searchParams.get('decision') ?? 'ALL').toUpperCase();

  if (USE_MOCK) {
    const all    = mockAlertsResponse.alerts.filter(a => decision === 'ALL' || a.decision === decision);
    const start  = (page - 1) * limit;
    const sliced = all.slice(start, start + limit);
    return NextResponse.json({
      alerts: sliced,
      total: all.length,
      page,
      limit,
      total_pages: Math.ceil(all.length / limit),
      data_source: 'mock',
    });
  }

  // 1. Try Postgres
  const pgResult = await queryPostgres(decision, page, limit);
  if (pgResult) {
    return NextResponse.json({ ...pgResult, data_source: 'postgres' });
  }

  // 2. Try orchestrator REST API
  const orchResult = await queryOrchestrator(decision, page, limit);
  if (orchResult) {
    return NextResponse.json({ ...orchResult, data_source: 'orchestrator' });
  }

  // 3. Fallback to mock with degraded warning
  console.error('[/api/alerts] All data sources unavailable. Returning mock data with degraded flag.');
  const all    = mockAlertsResponse.alerts.filter(a => decision === 'ALL' || a.decision === decision);
  const start  = (page - 1) * limit;
  const sliced = all.slice(start, start + limit);
  return NextResponse.json({
    alerts: sliced,
    total: all.length,
    page,
    limit,
    total_pages: Math.ceil(all.length / limit),
    data_source: 'mock_fallback',
    degraded: true,
    message: 'Neither Postgres nor orchestrator available. Showing demo data.',
  }, { status: 206 }); // 206 Partial Content signals degraded state
}
