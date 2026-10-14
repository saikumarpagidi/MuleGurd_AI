'use client';

import { useState, useEffect, useTransition, useMemo, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import DashboardLayout from '@/components/layout/DashboardLayout';
import GraphExplorer from '@/components/charts/GraphExplorer';
import { fetchGraphData } from '@/lib/api';
import { formatAmount, formatScore, getRingRoleColor } from '@/lib/utils';
import type { GraphResponse, GraphNode, GraphEdge, RingRole } from '@/lib/types';
import styles from './page.module.css';



function ExplorerContent() {
  const searchParams = useSearchParams();
  const initialParam = searchParams.get('account_id');
  const passedTgnScore = searchParams.get('tgn_score');

  const [query, setQuery] = useState(initialParam || '837B70300');
  const [edgeLimit, setEdgeLimit] = useState(40);
  const [graph, setGraph] = useState<GraphResponse | null>(null);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [isPending, startTransition] = useTransition();
  const [loadError, setLoadError] = useState<string | null>(null);

  // Fetch graph data for an account
  const loadAccount = (accountId: string, limit = edgeLimit, tgnScoreOverride: string | null = null) => {
    setLoadError(null);
    startTransition(async () => {
      try {
        const data = await fetchGraphData(accountId.trim(), limit, tgnScoreOverride);
        setGraph(data);
        const center = data.nodes.find(n => n.account_id === data.account_id);
        setSelectedNode(center || null);
      } catch (err: any) {
        setLoadError(err?.message || 'Failed to load graph data');
      }
    });
  };

  // Initial load on mount or parameter change
  useEffect(() => {
    const target = initialParam || '837B70300';
    setQuery(target);
    loadAccount(target, edgeLimit, passedTgnScore);
  }, [initialParam, passedTgnScore]);

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    loadAccount(query.trim(), edgeLimit);
  }

  function handlePivot(accountId: string) {
    setQuery(accountId);
    loadAccount(accountId, edgeLimit);
  }

  const filteredData = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] };
    return { nodes: graph.nodes, edges: graph.edges };
  }, [graph]);

  return (
    <DashboardLayout
      title="TGN Subgraph Explorer"
      subtitle="Temporal Graph Network (TGN) inference & Neo4j 2-hop transaction neighbourhood"
    >
      {/* ── Toolbar: Search & Quick Target Chips ── */}
      <div className={styles.toolbar}>
        <form onSubmit={handleSearch} className={styles.searchBar} id="graph-search-form">
          <input
            id="graph-account-input"
            className={`input ${styles.searchInput}`}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter Neo4j Account ID (e.g. 100428660, 81F201190)"
          />
          <select
            className={styles.limitSelect}
            value={edgeLimit}
            onChange={e => {
              const lim = Number(e.target.value);
              setEdgeLimit(lim);
              if (query.trim()) loadAccount(query.trim(), lim);
            }}
            title="Max 2-hop traversal edges"
          >
            <option value={20}>20 Edges</option>
            <option value={40}>40 Edges</option>
            <option value={60}>60 Edges</option>
            <option value={100}>100 Edges</option>
          </select>
          <button
            type="submit"
            className="btn btn-primary"
            id="btn-graph-search"
            disabled={isPending}
          >
            {isPending ? (
              <>
                <span className="spinner" style={{ width: 14, height: 14 }} /> Traversing Graph…
              </>
            ) : (
              '◎ Explore Subgraph'
            )}
          </button>
        </form>


      </div>

      {loadError && (
        <div style={{ padding: '12px 16px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#fca5a5', marginBottom: '20px', fontSize: '0.8125rem' }}>
          ⚠ {loadError} — defaulting to local cache. Ensure Graph Service is running on port 8003.
        </div>
      )}

      {/* ── Focal Account Summary Cards ── */}
      {graph && (
        <div className={`${styles.summaryRow} animate-fade-in-up`}>
          <div className={styles.summaryCard} id="graph-account-summary">
            <span className={styles.summaryLabel}>Investigated Account</span>
            <span className={styles.summaryValue} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              {graph.account_id}
            </span>
            <span className={styles.summarySub}>Focal Node Center</span>
          </div>

          <div className={styles.summaryCard} id="graph-ring-role">
            <span className={styles.summaryLabel}>Graph Ring Role</span>
            <span className={styles.summaryValue} style={{ color: getRingRoleColor(graph.ring_role) }}>
              {graph.ring_role}
            </span>
            <span className={styles.summarySub}>
              {graph.ring_role === 'HUB' ? 'Central High-Degree' : graph.ring_role === 'BRIDGE' ? 'Network Conduit' : 'Participant'}
            </span>
          </div>

          <div className={styles.summaryCard} id="graph-tgn-score">
            <span className={styles.summaryLabel}>TGN Risk Score</span>
            <span className={styles.summaryValue} style={{ color: graph.tgn_score > 0.6 ? '#ef4444' : graph.tgn_score > 0.3 ? '#f59e0b' : '#10b981' }}>
              {formatScore(graph.tgn_score)}
            </span>
            <span className={styles.summarySub}>Temporal Attention</span>
          </div>

          {graph.is_mule_probability != null && (
            <div className={styles.summaryCard} id="graph-mule-prob">
              <span className={styles.summaryLabel}>Mule Probability</span>
              <span className={styles.summaryValue} style={{ color: graph.is_mule_probability > 0.3 ? '#ef4444' : '#10b981' }}>
                {(graph.is_mule_probability * 100).toFixed(1)}%
              </span>
              <span className={styles.summarySub}>TGN v4 Node Detection</span>
            </div>
          )}
        </div>
      )}

      {/* ── Interactive D3 Graph Canvas ── */}
      <div className="animate-fade-in-up delay-2">
        {graph && (
          <GraphExplorer
            nodes={filteredData.nodes}
            edges={filteredData.edges}
            height={560}
            centeredAccountId={graph.account_id}
            selectedNodeId={selectedNode?.account_id}
            onSelectNode={node => setSelectedNode(node)}
            onPivotAccount={handlePivot}
          />
        )}
      </div>


    </DashboardLayout>
  );
}

export default function ExplorerPage() {
  return (
    <Suspense fallback={<div style={{ padding: 40, color: '#94a3b8' }}>Loading Graph Explorer…</div>}>
      <ExplorerContent />
    </Suspense>
  );
}
