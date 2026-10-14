'use client';

import { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphEdge } from '@/lib/types';
import { getRingRoleColor, getNodeRadius, formatAmount } from '@/lib/utils';
import styles from './GraphExplorer.module.css';

interface GraphExplorerProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  height?: number;
  centeredAccountId?: string;
  selectedNodeId?: string | null;
  onSelectNode?: (node: GraphNode | null) => void;
  onPivotAccount?: (accountId: string) => void;
}

interface SimNode extends GraphNode, d3.SimulationNodeDatum {
  id: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  edgeData: GraphEdge;
  isSelfLoop?: boolean;
  curvature: number;
}

/** Quadratic bezier curved edge path */
function curvePath(sx: number, sy: number, tx: number, ty: number, k: number): string {
  const mx = (sx + tx) / 2;
  const my = (sy + ty) / 2;
  const len = Math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2) || 1;
  const nx = -(ty - sy) / len;
  const ny =  (tx - sx) / len;
  const offset = len * k;
  return `M ${sx} ${sy} Q ${mx + nx * offset} ${my + ny * offset} ${tx} ${ty}`;
}

const ROLE_ICONS: Record<string, string> = {
  HUB: '⬟', BRIDGE: '⬡', MULE: '⬢', NONE: '·',
};

export default function GraphExplorer({
  nodes,
  edges,
  height = 580,
  centeredAccountId,
  onSelectNode,
  onPivotAccount,
}: GraphExplorerProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const svgRef    = useRef<SVGSVGElement>(null);
  const tipRef    = useRef<HTMLDivElement>(null);
  const zoomRef   = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const zoomBy = useCallback((factor: number) => {
    if (svgRef.current && zoomRef.current)
      d3.select(svgRef.current).transition().duration(240).call(zoomRef.current.scaleBy, factor);
  }, []);

  const resetZoom = useCallback(() => {
    if (svgRef.current && zoomRef.current)
      d3.select(svgRef.current).transition().duration(380)
        .call(zoomRef.current.transform, d3.zoomIdentity);
  }, []);

  const fitView = useCallback(() => {
    if (!svgRef.current || !wrapperRef.current || !zoomRef.current) return;
    const g = d3.select(svgRef.current).select<SVGGElement>('g.zr');
    if (g.empty()) return;
    const bb = (g.node() as SVGGElement).getBBox();
    const ww = wrapperRef.current.clientWidth;
    const wh = height;
    const s  = 0.82 / Math.max(bb.width / ww, bb.height / wh);
    const tx = ww / 2 - s * (bb.x + bb.width  / 2);
    const ty = wh / 2 - s * (bb.y + bb.height / 2);
    d3.select(svgRef.current).transition().duration(480)
      .call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(s));
  }, [height]);

  useEffect(() => {
    if (!svgRef.current || !wrapperRef.current || !nodes.length) return;

    const width = wrapperRef.current.clientWidth || 900;
    const svg   = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    /* ── Defs ─────────────────────────────────────────── */
    const defs = svg.append('defs');

    // Dot-grid background pattern
    defs.append('pattern')
      .attr('id', 'dot-grid').attr('x', 0).attr('y', 0)
      .attr('width', 28).attr('height', 28)
      .attr('patternUnits', 'userSpaceOnUse')
      .append('circle').attr('cx', 14).attr('cy', 14).attr('r', 0.9)
      .attr('fill', 'rgba(255,255,255,0.055)');

    // Per-role glow filters
    const roles: [string, string][] = [
      ['HUB',    '#EF4444'],
      ['BRIDGE', '#F59E0B'],
      ['MULE',   '#EC4899'],
      ['NONE',   '#00D4FF'],
    ];

    roles.forEach(([role, color]) => {
      // Node glow
      const f = defs.append('filter').attr('id', `gf-${role}`)
        .attr('x', '-60%').attr('y', '-60%').attr('width', '220%').attr('height', '220%');
      f.append('feGaussianBlur').attr('in', 'SourceGraphic').attr('stdDeviation', '3.5').attr('result', 'blur');
      const m = f.append('feMerge');
      m.append('feMergeNode').attr('in', 'blur');
      m.append('feMergeNode').attr('in', 'blur');
      m.append('feMergeNode').attr('in', 'SourceGraphic');

      // Radial gradient fill
      const g2 = defs.append('radialGradient').attr('id', `rg-${role}`)
        .attr('cx', '35%').attr('cy', '35%').attr('r', '65%');
      g2.append('stop').attr('offset', '0%').attr('stop-color', color).attr('stop-opacity', 0.55);
      g2.append('stop').attr('offset', '100%').attr('stop-color', color).attr('stop-opacity', 0.07);
    });

    // Stronger center glow
    const cf = defs.append('filter').attr('id', 'gf-center')
      .attr('x', '-80%').attr('y', '-80%').attr('width', '260%').attr('height', '260%');
    cf.append('feGaussianBlur').attr('in', 'SourceGraphic').attr('stdDeviation', '8').attr('result', 'blur');
    const cm = cf.append('feMerge');
    cm.append('feMergeNode').attr('in', 'blur');
    cm.append('feMergeNode').attr('in', 'blur');
    cm.append('feMergeNode').attr('in', 'SourceGraphic');

    // Center pulse gradient
    const cp = defs.append('radialGradient').attr('id', 'rg-center-pulse')
      .attr('cx', '50%').attr('cy', '50%').attr('r', '50%');
    cp.append('stop').attr('offset', '0%').attr('stop-color', '#00D4FF').attr('stop-opacity', 0.28);
    cp.append('stop').attr('offset', '100%').attr('stop-color', '#00D4FF').attr('stop-opacity', 0);

    // Arrow markers
    [
      { id: 'mk-launder', c: '#EF4444' },
      { id: 'mk-high',    c: '#F59E0B' },
      { id: 'mk-normal',  c: 'rgba(100,116,139,0.65)' },
    ].forEach(({ id, c }) => {
      defs.append('marker')
        .attr('id', id).attr('viewBox', '0 -5 10 10')
        .attr('refX', 30).attr('refY', 0)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L10,0L0,4').attr('fill', c);
    });

    /* ── Background ───────────────────────────────────── */
    svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', '#06080F');
    svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', 'url(#dot-grid)');

    /* ── Data prep ────────────────────────────────────── */
    const simNodes: SimNode[] = nodes.map(n => ({ ...n, id: n.id || n.account_id }));
    const nodeById = new Map(simNodes.map(n => [n.id, n]));
    const simLinks: SimLink[] = [];
    const selfLoops: { node: SimNode; edgeData: GraphEdge }[] = [];

    const pairCount = new Map<string, number>();
    edges.forEach(e => {
      const src = nodeById.get(e.source);
      const dst = nodeById.get(e.target);
      if (!src || !dst) return;
      if (e.source === e.target) { selfLoops.push({ node: src, edgeData: e }); return; }

      const fwd = `${e.source}|${e.target}`;
      const rev = `${e.target}|${e.source}`;
      const tot = (pairCount.get(fwd) ?? 0) + (pairCount.get(rev) ?? 0);

      // Curvature: alternating direction for parallel / bidirectional edges
      let k: number;
      if (tot === 0) k = e.is_laundering ? 0.28 : 0.16;
      else k = tot % 2 === 1 ? 0.35 : -0.35;
      pairCount.set(fwd, (pairCount.get(fwd) ?? 0) + 1);

      simLinks.push({ source: src, target: dst, edgeData: e, curvature: k });
    });

    /* ── Zoom container ───────────────────────────────── */
    const g = svg.append('g').attr('class', 'zr');
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 6])
      .on('zoom', ev => g.attr('transform', ev.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    /* ── Force simulation ─────────────────────────────── */
    const sim = d3.forceSimulation<SimNode>(simNodes)
      .force('link',
        d3.forceLink<SimNode, SimLink>(simLinks)
          .id(d => d.id)
          .distance(d => {
            const sr = (d.source as SimNode).ring_role;
            const tr = (d.target as SimNode).ring_role;
            return (sr === 'HUB' || tr === 'HUB') ? 150 : 110;
          })
          .strength(0.42)
      )
      .force('charge', d3.forceManyBody<SimNode>()
        .strength(d => d.ring_role === 'HUB' ? -550 : d.ring_role === 'BRIDGE' ? -320 : -200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide<SimNode>(d => getNodeRadius(d.ring_role, d.pagerank) + 22));

    /* ── Links ────────────────────────────────────────── */
    const linkG = g.append('g').attr('class', 'links');

    const link = linkG.selectAll<SVGPathElement, SimLink>('path')
      .data(simLinks).join('path')
      .attr('fill', 'none')
      .attr('stroke', d => {
        if (d.edgeData.is_laundering) return '#EF4444';
        if (d.edgeData.amount >= 5000) return '#F59E0B';
        return 'rgba(100,116,139,0.28)';
      })
      .attr('stroke-width', d =>
        d.edgeData.is_laundering ? 2.2 :
        Math.max(1.2, Math.log10((d.edgeData.amount || 100) / 100 + 1) * 1.7)
      )
      .attr('stroke-dasharray', d =>
        d.edgeData.is_laundering ? '7 3' :
        d.edgeData.structuring_flag ? '4 3' : '0'
      )
      .attr('marker-end', d =>
        d.edgeData.is_laundering ? 'url(#mk-launder)' :
        d.edgeData.amount >= 5000 ? 'url(#mk-high)' : 'url(#mk-normal)'
      )
      .attr('opacity', 0.75)
      .style('cursor', 'pointer');

    // Self-loops (cubic bezier arc)
    const loopLink = linkG.selectAll<SVGPathElement, { node: SimNode; edgeData: GraphEdge }>('path.sl')
      .data(selfLoops).join('path').attr('class', 'sl')
      .attr('fill', 'none')
      .attr('stroke', d => d.edgeData.is_laundering ? '#EF4444' : 'rgba(100,116,139,0.4)')
      .attr('stroke-width', 1.8)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.65);

    /* ── Nodes ────────────────────────────────────────── */
    const nodeG = g.append('g')
      .selectAll<SVGGElement, SimNode>('g.node')
      .data(simNodes).join('g').attr('class', 'node')
      .attr('id', d => `graph-node-${d.id}`)
      .style('cursor', 'grab')
      .call(
        d3.drag<SVGGElement, SimNode>()
          .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag',  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on('end',   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Center node: large pulse halo ring (dashed)
    nodeG.filter(d => d.account_id === centeredAccountId)
      .append('circle').attr('class', 'pulse-halo')
      .attr('r', d => getNodeRadius(d.ring_role, d.pagerank) + 24)
      .attr('fill', 'url(#rg-center-pulse)')
      .attr('opacity', 1);

    nodeG.filter(d => d.account_id === centeredAccountId)
      .append('circle')
      .attr('r', d => getNodeRadius(d.ring_role, d.pagerank) + 20)
      .attr('fill', 'none')
      .attr('stroke', '#00D4FF').attr('stroke-width', 1)
      .attr('stroke-dasharray', '6 4').attr('opacity', 0.45);

    // SCC orbit ring
    nodeG.filter(d => d.in_scc_ring && d.account_id !== centeredAccountId)
      .append('circle')
      .attr('r', d => getNodeRadius(d.ring_role, d.pagerank) + 7)
      .attr('fill', 'none')
      .attr('stroke', d => getRingRoleColor(d.ring_role))
      .attr('stroke-width', 1.4).attr('stroke-dasharray', '3 3').attr('opacity', 0.55);

    // Main node body — gradient fill + glow filter
    nodeG.append('circle').attr('class', 'node-body')
      .attr('r', d => getNodeRadius(d.ring_role, d.pagerank))
      .attr('fill', d => `url(#rg-${d.ring_role})`)
      .attr('stroke', d => getRingRoleColor(d.ring_role))
      .attr('stroke-width', d => d.account_id === centeredAccountId ? 2.8 : 1.8)
      .style('filter', d =>
        d.account_id === centeredAccountId ? 'url(#gf-center)' : `url(#gf-${d.ring_role})`
      );

    // Inner highlight core
    nodeG.append('circle')
      .attr('r', d => getNodeRadius(d.ring_role, d.pagerank) * 0.42)
      .attr('fill', d => getRingRoleColor(d.ring_role))
      .attr('opacity', d => d.account_id === centeredAccountId ? 0.38 : 0.16)
      .attr('pointer-events', 'none');

    // Role icon
    nodeG.append('text')
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', d => getRingRoleColor(d.ring_role))
      .attr('font-size', d => Math.max(10, getNodeRadius(d.ring_role, d.pagerank) * 0.68))
      .attr('font-weight', '700').attr('pointer-events', 'none')
      .text(d => ROLE_ICONS[d.ring_role] ?? '·');

    // Label pill background
    const labelDy = (d: SimNode) => getNodeRadius(d.ring_role, d.pagerank) + 15;
    const isCenter = (d: SimNode) => d.account_id === centeredAccountId;

    nodeG.append('rect')
      .attr('class', 'label-bg')
      .attr('rx', 4).attr('ry', 4)
      .attr('height', 14)
      .attr('width', d => isCenter(d) ? 76 : 62)
      .attr('x', d => isCenter(d) ? -38 : -31)
      .attr('y', d => labelDy(d) - 8)
      .attr('fill', 'rgba(6,8,15,0.75)')
      .attr('pointer-events', 'none');

    // Label text
    nodeG.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', d => labelDy(d) + 1)
      .attr('fill', d => isCenter(d) ? '#38BDF8' : '#7B92A8')
      .attr('font-size', d => isCenter(d) ? 10.5 : 8.5)
      .attr('font-weight', d => isCenter(d) ? '700' : '500')
      .attr('font-family', '"JetBrains Mono","Fira Code",monospace')
      .attr('pointer-events', 'none')
      .text(d => d.account_id.length > 11 ? `${d.account_id.slice(0, 10)}…` : d.account_id);

    /* ── Tooltip ──────────────────────────────────────── */
    const tip = d3.select(tipRef.current);

    function showTip(html: string) {
      tip.style('display', 'block').html(html);
    }
    function moveTip(ev: MouseEvent) {
      const [mx, my] = d3.pointer(ev, wrapperRef.current);
      tip.style('left', `${mx + 18}px`).style('top', `${my - 14}px`);
    }
    function hideTip() { tip.style('display', 'none'); }

    // Node hover
    nodeG
      .on('mouseover', (ev, d) => {
        link.attr('opacity', (l: SimLink) => {
          const s = (l.source as SimNode).account_id;
          const t = (l.target as SimNode).account_id;
          return (s === d.account_id || t === d.account_id) ? 1 : 0.06;
        });
        const col = getRingRoleColor(d.ring_role);
        const riskCol = d.risk_score > 0.6 ? '#ef4444' : d.risk_score > 0.3 ? '#f59e0b' : '#10b981';
        showTip(`
          <div class="${styles.tipHeader}" style="border-left-color:${col}">
            <span class="${styles.tipRole}" style="color:${col}">${d.ring_role}</span>
            ${d.in_scc_ring ? `<span class="${styles.tipBadge}" style="color:#ef4444;border-color:rgba(239,68,68,0.4);background:rgba(239,68,68,0.1)">SCC</span>` : ''}
          </div>
          <div class="${styles.tipId}">${d.account_id}</div>
          <div class="${styles.tipDivider}"></div>
          <div class="${styles.tipRow}"><span>Risk Score</span><span style="color:${riskCol}">${(d.risk_score * 100).toFixed(1)}%</span></div>
          <div class="${styles.tipRow}"><span>PageRank</span><span>${d.pagerank.toFixed(4)}</span></div>
          <div class="${styles.tipRow}"><span>Community</span><span>#${d.community_id}</span></div>
          <div class="${styles.tipHint}">Double-click to pivot center</div>
        `);
      })
      .on('mousemove', moveTip)
      .on('mouseleave', () => { link.attr('opacity', 0.75); hideTip(); })
      .on('click', (_, d) => { if (onSelectNode) onSelectNode(d); })
      .on('dblclick', (ev, d) => { ev.stopPropagation(); if (onPivotAccount) onPivotAccount(d.account_id); });

    // Link hover
    link
      .on('mouseover', (ev, d) => {
        const col = d.edgeData.is_laundering ? '#EF4444' : d.edgeData.amount >= 5000 ? '#F59E0B' : '#38BDF8';
        showTip(`
          <div class="${styles.tipHeader}" style="border-left-color:${col}">
            <span class="${styles.tipRole}" style="color:${col}">
              ${d.edgeData.is_laundering ? '⚠ LAUNDERING FLOW' : d.edgeData.amount >= 5000 ? '★ HIGH VALUE' : 'TRANSACTION'}
            </span>
          </div>
          <div class="${styles.tipDivider}"></div>
          <div class="${styles.tipRow}"><span>Amount</span><span style="color:${col};font-weight:700">${formatAmount(d.edgeData.amount)}</span></div>
          <div class="${styles.tipRow}"><span>From</span><span style="font-size:0.7rem">${d.edgeData.source}</span></div>
          <div class="${styles.tipRow}"><span>To</span><span style="font-size:0.7rem">${d.edgeData.target}</span></div>
          <div class="${styles.tipRow}"><span>Channel</span><span>${d.edgeData.payment_format || 'WIRE'}</span></div>
          ${d.edgeData.structuring_flag ? `<div class="${styles.tipRow}"><span>Flag</span><span style="color:#f59e0b">⚠ Structuring</span></div>` : ''}
        `);
      })
      .on('mousemove', moveTip)
      .on('mouseleave', hideTip);

    /* ── Tick ─────────────────────────────────────────── */
    sim.on('tick', () => {
      link.attr('d', d => curvePath(
        (d.source as SimNode).x ?? 0, (d.source as SimNode).y ?? 0,
        (d.target as SimNode).x ?? 0, (d.target as SimNode).y ?? 0,
        d.curvature
      ));
      loopLink.attr('d', d => {
        const x = d.node.x ?? 0, y = d.node.y ?? 0;
        const r = getNodeRadius(d.node.ring_role, d.node.pagerank);
        return `M ${x - r * 0.55} ${y - r} C ${x - r * 2.1} ${y - r * 3.2}, ${x + r * 2.1} ${y - r * 3.2}, ${x + r * 0.55} ${y - r}`;
      });
      nodeG.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    sim.on('end', () => { fitView(); });

    return () => { sim.stop(); };
  }, [nodes, edges, height, centeredAccountId, onSelectNode, onPivotAccount, fitView]);

  const launderCount = edges.filter(e => e.is_laundering).length;
  const highValCount = edges.filter(e => e.amount >= 5000).length;

  return (
    <div className={styles.wrapper} ref={wrapperRef} id="graph-explorer">

      {/* Floating Controls */}
      <div className={styles.controls}>
        <button className={styles.ctrlBtn} onClick={() => zoomBy(1.35)} title="Zoom In" aria-label="Zoom In">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M6.5 1.5v10M1.5 6.5h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
        <button className={styles.ctrlBtn} onClick={() => zoomBy(0.74)} title="Zoom Out" aria-label="Zoom Out">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M1.5 6.5h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
        <button className={styles.ctrlBtn} onClick={fitView} title="Fit to Screen" aria-label="Fit to Screen">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M1.5 5V1.5H5M8 1.5h3.5V5M11.5 8v3.5H8M5 11.5H1.5V8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <button className={styles.ctrlBtn} onClick={resetZoom} title="Reset View" aria-label="Reset View">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M2 6.5a4.5 4.5 0 1 1 1.4 3.2M2 6.5V3.5M2 6.5H5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      {/* Canvas Stats */}
      <div className={styles.canvasStats}>
        <span className={styles.liveDot} />
        <span className={styles.statVal}>{nodes.length}</span>
        <span className={styles.statLbl}>nodes</span>
        <span className={styles.statSep}>·</span>
        <span className={styles.statVal}>{edges.length}</span>
        <span className={styles.statLbl}>edges</span>
        {launderCount > 0 && <>
          <span className={styles.statSep}>·</span>
          <span className={styles.statVal} style={{ color: '#EF4444' }}>{launderCount}</span>
          <span className={styles.statLbl} style={{ color: 'rgba(239,68,68,0.7)' }}>laundering</span>
        </>}
        {highValCount > 0 && <>
          <span className={styles.statSep}>·</span>
          <span className={styles.statVal} style={{ color: '#F59E0B' }}>{highValCount}</span>
          <span className={styles.statLbl} style={{ color: 'rgba(245,158,11,0.7)' }}>≥$5k</span>
        </>}
      </div>

      {/* Interaction Hint */}
      <div className={styles.hintBar}>
        Hover for telemetry · Double-click to pivot · Drag to reposition · Scroll to zoom
      </div>

      {/* SVG Canvas */}
      <svg ref={svgRef} className={styles.svg} width="100%" height={height} />
      <div ref={tipRef} className={styles.tooltip} style={{ display: 'none' }} />

      {/* Legend */}
      <div className={styles.legend}>
        <div className={styles.legendTitle}>Legend</div>
        {[
          { color: '#EF4444', label: 'Mega Hub',        scc: true  },
          { color: '#F59E0B', label: 'Bridge / Conduit',scc: false },
          { color: '#EC4899', label: 'Money Mule',      scc: false },
          { color: '#00D4FF', label: 'Legitimate',      scc: false },
        ].map(({ color, label, scc }) => (
          <div key={label} className={styles.legendRow}>
            <span className={styles.legendNode} style={{ borderColor: color, background: `${color}1A` }}>
              {scc && <span className={styles.legendSCCRing} style={{ borderColor: color }} />}
            </span>
            <span className={styles.legendLbl}>{label}</span>
          </div>
        ))}
        <div className={styles.legendSep} />
        <div className={styles.legendRow}>
          <span className={styles.legendEdge} style={{ background: '#EF4444' }} />
          <span className={styles.legendLbl} style={{ color: 'rgba(239,68,68,0.85)' }}>Laundering</span>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendEdge} style={{ background: '#F59E0B' }} />
          <span className={styles.legendLbl} style={{ color: 'rgba(245,158,11,0.85)' }}>High Value</span>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendEdge} style={{ background: 'rgba(100,116,139,0.5)' }} />
          <span className={styles.legendLbl}>Standard</span>
        </div>
      </div>
    </div>
  );
}
