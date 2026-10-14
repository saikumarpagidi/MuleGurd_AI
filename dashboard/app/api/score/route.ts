/**
 * /api/score — Dashboard scoring proxy
 *
 * Calls AI Engine (POST /score) and Graph Service (POST /graph-score) in
 * parallel, then fuses results with:
 *   final_score = 0.50 × tgn_score + 0.30 × lgbm_score + 0.15 × ae_score
 * (renormalized if graph service is in degraded mode)
 *
 * Weights match docker-compose WEIGHT_* env vars in scoring-orchestrator.
 * Device score (0.05) is excluded — no device fingerprinting implemented.
 */

import { NextResponse } from 'next/server';
import type { AIEngineScore, GraphScoreResult, ScoreResponse, TransactionInput } from '@/lib/types';
import { mockScoreBlock, mockScoreApprove } from '@/lib/mock-data';

const AI_ENGINE_URL        = process.env.AI_ENGINE_URL        ?? 'http://127.0.0.1:8002';
const GRAPH_SERVICE_URL    = process.env.GRAPH_SERVICE_URL    ?? 'http://127.0.0.1:8003';
const USE_MOCK             = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';

// Weight constants (match docker-compose scoring-orchestrator env)
const W_TGN  = 0.50;
const W_LGBM = 0.30;
const W_AE   = 0.15;
// Device weight (0.05) redistributed when absent

function fuseScores(lgbm: number, ae: number, tgn: number | null, degraded: boolean): number {
  if (tgn === null || degraded) {
    // Renormalize without TGN: redistribute 0.50 to LGBM/AE proportionally
    const totalWithout = W_LGBM + W_AE;
    return (lgbm * W_LGBM + ae * W_AE) / totalWithout;
  }
  const totalWeight = W_TGN + W_LGBM + W_AE;
  return (tgn * W_TGN + lgbm * W_LGBM + ae * W_AE) / totalWeight;
}

function decidFromScore(score: number): 'APPROVE' | 'REVIEW' | 'BLOCK' {
  if (score >= 0.75) return 'BLOCK';
  if (score >= 0.45) return 'REVIEW';
  return 'APPROVE';
}

function humanFeatureName(f: string): string {
  return f
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace(/\bTxn\b/, 'Txn')
    .replace(/\bFan Out\b/, 'Fan-out')
    .replace(/\bLgbm\b/, 'LightGBM');
}

function buildExplanation(
  aiScore: AIEngineScore,
  graphResult: GraphScoreResult | null,
  finalScore: number,
): ScoreResponse['explanation'] {
  const decision = decidFromScore(finalScore);
  const role = graphResult?.ring_role ?? 'NONE';
  const mode = 'fallback' as const;

  const parts: string[] = [];

  if (decision === 'BLOCK') {
    parts.push(`Transaction flagged as HIGH RISK (score ${(finalScore * 100).toFixed(1)}%).`);
  } else if (decision === 'REVIEW') {
    parts.push(`Transaction flagged for REVIEW (score ${(finalScore * 100).toFixed(1)}%).`);
  } else {
    parts.push(`Transaction appears LOW RISK (score ${(finalScore * 100).toFixed(1)}%).`);
  }

  // Top SHAP driver
  if (aiScore.shap_top_5.length > 0) {
    const top = aiScore.shap_top_5[0];
    const dir = top.direction === 'RAISES_RISK' ? 'raises' : 'lowers';
    parts.push(
      `Primary driver: "${humanFeatureName(top.feature)}" ${dir} risk (SHAP: ${top.shap_contribution > 0 ? '+' : ''}${top.shap_contribution.toFixed(3)}).`
    );
  }

  if (graphResult) {
    if (graphResult.in_scc_ring) {
      parts.push(`Source account is in a Strongly Connected Component ring (community #${graphResult.community_id}).`);
    }
    if (role !== 'NONE') {
      parts.push(`Graph role detected: ${role} (ring probability: ${(graphResult.ring_probability * 100).toFixed(0)}%).`);
    }
    if (graphResult.degraded_mode) {
      parts.push('Graph service ran in heuristic mode — TGN score approximated from GDS signals.');
    }
  }

  parts.push(
    `Model scores — LightGBM: ${(aiScore.lgbm_score * 100).toFixed(1)}%, ` +
    `Autoencoder: ${(aiScore.ae_score * 100).toFixed(1)}%` +
    (graphResult?.tgn_score != null ? `, TGN: ${((graphResult.tgn_score) * 100).toFixed(1)}%` : '') +
    `. Feature source: ${aiScore.feature_source}.`
  );

  return {
    text: parts.join(' '),
    mode,
    ring_membership: graphResult?.in_scc_ring ? `Ring #${graphResult.community_id}` : undefined,
  };
}

export async function POST(request: Request) {
  const t0 = Date.now();
  try {
    const body: TransactionInput = await request.json();

    if (USE_MOCK) {
      const isHighRisk = body.amount > 5000 || body.payment_format === 'Cash' || body.payment_format === 'Bitcoin';
      const mock = isHighRisk ? mockScoreBlock : mockScoreApprove;
      return NextResponse.json({ ...mock, transaction_id: body.transaction_id, timestamp: new Date().toISOString() });
    }

    // ── Parallel calls to AI Engine + Graph Service ────────────────────────
    const aiEnginePayload = {
      transaction_id: body.transaction_id,
      from_account:   body.from_account,
      to_account:     body.to_account,
      amount:         body.amount,
      timestamp:      body.timestamp,
      payment_format: body.payment_format,
      currency:       body.currency,
    };

    const graphPayload = {
      from_account:   body.from_account,
      to_account:     body.to_account,
      amount:         body.amount,
      timestamp:      body.timestamp,
      payment_format: body.payment_format,
      currency:       body.currency,
    };

    const [aiRes, graphRes] = await Promise.allSettled([
      fetch(`${AI_ENGINE_URL}/score`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(aiEnginePayload),
        signal:  AbortSignal.timeout(8000),
      }),
      fetch(`${GRAPH_SERVICE_URL}/graph-score`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(graphPayload),
        signal:  AbortSignal.timeout(10000),
      }),
    ]);

    // ── Parse AI Engine response (required) ─────────────────────────────
    if (aiRes.status === 'rejected' || !aiRes.value.ok) {
      const reason = aiRes.status === 'rejected'
        ? aiRes.reason?.message
        : `HTTP ${aiRes.value.status}`;
      console.error('[/api/score] AI Engine failed:', reason);
      return NextResponse.json(
        { error: `AI Engine unavailable: ${reason}. Is the Docker stack running?` },
        { status: 503 }
      );
    }

    const aiScore: AIEngineScore = await aiRes.value.json();

    // ── Parse Graph Service response (optional — graceful degradation) ──
    let graphResult: GraphScoreResult | null = null;
    if (graphRes.status === 'fulfilled' && graphRes.value.ok) {
      graphResult = await graphRes.value.json();
    } else {
      const reason = graphRes.status === 'rejected'
        ? graphRes.reason?.message
        : `HTTP ${graphRes.value.status}`;
      console.warn('[/api/score] Graph Service unavailable — running without TGN:', reason);
    }

    // ── Fuse scores ────────────────────────────────────────────────────
    const tgnScore    = graphResult?.tgn_score ?? null;
    const degraded    = graphResult?.degraded_mode ?? true;
    const finalScore  = fuseScores(aiScore.lgbm_score, aiScore.ae_score, tgnScore, degraded);
    const decision    = decidFromScore(finalScore);

    const response: ScoreResponse = {
      transaction_id:  aiScore.transaction_id,
      final_score:     finalScore,
      decision,
      component_scores: {
        lgbm_score: aiScore.lgbm_score,
        ae_score:   aiScore.ae_score,
        tgn_score:  tgnScore ?? 0,
      },
      shap_top_5: aiScore.shap_top_5.map(f => ({
        feature:     f.feature,
        value:       f.value,
        contribution: f.shap_contribution,
        direction:   f.direction,
        displayName: humanFeatureName(f.feature),
      })),
      ae_top_5: (aiScore.ae_top_5 ?? []).map(f => ({
        feature:               f.feature,
        value:                 f.value,
        reconstruction_z_score: f.reconstruction_z_score,
        direction:             f.direction,
        displayName:           humanFeatureName(f.feature),
      })),
      explanation:     buildExplanation(aiScore, graphResult, finalScore),
      is_mule_probability: graphResult?.is_mule_probability ?? null,
      ring_role:       graphResult?.ring_role ?? 'NONE',
      ring_probability: graphResult?.ring_probability ?? 0,
      in_scc_ring:     graphResult?.in_scc_ring ?? false,
      community_id:    graphResult?.community_id ?? -1,
      pagerank:        graphResult?.pagerank ?? 0,
      latency_ms:      Date.now() - t0,
      feature_source:  aiScore.feature_source,
      model_versions:  aiScore.model_versions,
      degraded_mode:   degraded,
      timestamp:       new Date().toISOString(),
    };


    return NextResponse.json(response);

  } catch (err) {
    console.error('[/api/score] Unhandled error:', err);
    return NextResponse.json({ error: 'Internal error in scoring proxy' }, { status: 500 });
  }
}
