/**
 * /api/graph — Dashboard Graph Explorer proxy
 *
 * Calls Graph Service GET /subgraph/{account_id} (port 8003) and also runs
 * POST /graph-score to get TGN + GDS metadata for the centre node.
 *
 * Transforms the SubgraphResponse into the GraphResponse expected by
 * the GraphExplorer D3 component.
 */

import { NextResponse } from 'next/server';
import type { GraphResponse, GraphNode, GraphEdge, SubgraphEdge } from '@/lib/types';
import { mockGraphResponse } from '@/lib/mock-data';

const GRAPH_SERVICE_URL = process.env.GRAPH_SERVICE_URL ?? 'http://127.0.0.1:8003';
const USE_MOCK          = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';

function ringRoleFromStr(role?: string): 'HUB' | 'BRIDGE' | 'MULE' | 'NONE' {
  const r = (role ?? 'NONE').toUpperCase();
  if (r === 'HUB' || r === 'BRIDGE' || r === 'MULE') return r;
  return 'NONE';
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const accountId = searchParams.get('account_id') ?? '';
  const limit     = Number(searchParams.get('limit') ?? 60);
  const tgnScoreOverride = searchParams.get('tgn_score');

  if (USE_MOCK) {
    return NextResponse.json({ ...mockGraphResponse, account_id: accountId || mockGraphResponse.account_id });
  }

  if (!accountId) {
    return NextResponse.json({ error: 'account_id is required' }, { status: 400 });
  }

  try {
    // ── Fetch subgraph + graph-score in parallel ──────────────────────
    const [subRes, scoreRes] = await Promise.allSettled([
      fetch(`${GRAPH_SERVICE_URL}/subgraph/${encodeURIComponent(accountId)}?limit=${limit}`, {
        signal: AbortSignal.timeout(12000),
      }),
      fetch(`${GRAPH_SERVICE_URL}/graph-score`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_account:   accountId,
          to_account:     accountId,
          amount:         1,
          timestamp:      new Date().toISOString(),
          payment_format: 'OTHER',
          currency:       'USD',
        }),
        signal: AbortSignal.timeout(8000),
      }),
    ]);

    // ── Parse subgraph ───────────────────────────────────────────────
    if (subRes.status === 'rejected' || !subRes.value.ok) {
      const reason = subRes.status === 'rejected'
        ? subRes.reason?.message
        : `HTTP ${subRes.value.status}`;
      console.error('[/api/graph] Subgraph fetch failed:', reason);
      return NextResponse.json(
        { error: `Graph Service unavailable: ${reason}. Is the Docker stack running?` },
        { status: 503 }
      );
    }
    const subgraph = await subRes.value.json();

    // ── Parse graph-score (optional enrichment) ──────────────────────
    let graphScore = null;
    if (scoreRes.status === 'fulfilled' && scoreRes.value.ok) {
      graphScore = await scoreRes.value.json();
    } else {
      console.warn('[/api/graph] graph-score unavailable, using GDS-only properties');
    }

    const acctProps = subgraph.account_properties ?? {};
    const centreRingRole = ringRoleFromStr(graphScore?.ring_role ?? acctProps.ring_role);
    const centreTgn      = tgnScoreOverride ? parseFloat(tgnScoreOverride) : (graphScore?.tgn_score ?? 0);
    const centreMuleProb = graphScore?.is_mule_probability ?? null;
    const centrePR       = parseFloat(acctProps.pagerank ?? 0);
    const centreCom      = parseInt(acctProps.community_id ?? -1);
    const centreSCC      = Boolean(acctProps.in_scc_ring ?? graphScore?.in_scc_ring ?? false);

    // ── Build node list from edges ────────────────────────────────────
    const nodeMap = new Map<string, GraphNode>();
    const edges: GraphEdge[] = [];

    // Add center node
    nodeMap.set(accountId, {
      id:                 accountId,
      account_id:         accountId,
      ring_role:          centreRingRole,
      pagerank:           centrePR,
      community_id:       centreCom,
      in_scc_ring:        centreSCC,
      risk_score:         centreTgn,
      is_mule_probability: centreMuleProb,
      lifetime_txn_count: subgraph.edge_count,
      total_sent:         subgraph.edges.reduce((s: number, e: SubgraphEdge) => e.src === accountId ? s + e.amount : s, 0),
    });

    // Derive neighbour nodes from edges
    let edgeIndex = 0;
    for (const e of subgraph.edges as SubgraphEdge[]) {
      const src = e.src;
      const dst = e.dst;

      if (!nodeMap.has(src)) {
        nodeMap.set(src, {
          id: src, account_id: src,
          ring_role: 'NONE', pagerank: 0, community_id: -1,
          in_scc_ring: false, risk_score: e.is_laundering ? 0.8 : 0.1,
          is_mule_probability: null,
          lifetime_txn_count: 1, total_sent: e.amount,
        });
      }
      if (!nodeMap.has(dst)) {
        nodeMap.set(dst, {
          id: dst, account_id: dst,
          ring_role: 'NONE', pagerank: 0, community_id: -1,
          in_scc_ring: false, risk_score: e.is_laundering ? 0.8 : 0.1,
          is_mule_probability: null,
          lifetime_txn_count: 1, total_sent: 0,
        });
      }

      edges.push({
        id:              `e_${edgeIndex++}`,
        source:          src,
        target:          dst,
        amount:          e.amount,
        timestamp:       new Date(e.timestamp_s * 1000).toISOString(),
        payment_format:  e.payment_format,
        is_laundering:   e.is_laundering,
        structuring_flag: e.amount > 9000 && e.amount < 10000,
      });
    }

    const response: GraphResponse = {
      account_id:      accountId,
      nodes:           Array.from(nodeMap.values()),
      edges,
      tgn_score:       centreTgn,
      is_mule_probability: centreMuleProb,
      ring_role:       centreRingRole,
      community_id:    centreCom,
      neo4j_connected: subgraph.neo4j_connected,
      degraded_mode:   graphScore?.degraded_mode ?? false,
    };

    return NextResponse.json(response);

  } catch (err) {
    console.error('[/api/graph] Unhandled error:', err);
    return NextResponse.json({ error: 'Graph proxy error' }, { status: 500 });
  }
}
