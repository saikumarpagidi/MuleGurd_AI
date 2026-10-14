'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { SHAPFeature } from '@/lib/types';
import styles from './SHAPWaterfall.module.css';

interface SHAPWaterfallProps {
  features: SHAPFeature[];
  baseScore?: number;
  finalScore?: number;
  animated?: boolean;
}

export default function SHAPWaterfall({
  features,
  baseScore = 0.5,
  finalScore,
  animated = true,
}: SHAPWaterfallProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !features.length) return;

    const sorted = [...features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    const svgEl = svgRef.current;
    const totalW = svgEl.clientWidth || 480;
    const barH   = 32;
    const gap    = 10;
    const labelW = 260;
    const valueW = 56;
    const paddingT = 16;
    const paddingB = 32;
    const totalH   = paddingT + sorted.length * (barH + gap) + paddingB;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('height', totalH);

    const chartW = totalW - labelW - valueW - 16;

    // X scale: symmetric around 0
    const maxAbs = Math.max(...sorted.map(f => Math.abs(f.contribution))) * 1.15;
    const xScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, chartW]);

    // Zero line position
    const zeroX = xScale(0);

    // Zero line
    svg.append('line')
      .attr('x1', labelW + zeroX)
      .attr('y1', paddingT)
      .attr('x2', labelW + zeroX)
      .attr('y2', totalH - paddingB)
      .attr('stroke', 'rgba(255,255,255,0.08)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 4');

    sorted.forEach((feat, i) => {
      const y     = paddingT + i * (barH + gap);
      const isRisk = feat.direction === 'RAISES_RISK';
      const color  = isRisk ? '#EF4444' : '#00D4FF';
      const barX   = isRisk ? zeroX : xScale(feat.contribution);
      const barW   = Math.abs(xScale(feat.contribution) - zeroX);

      // Feature name label
      svg.append('text')
        .attr('x', labelW - 8)
        .attr('y', y + barH / 2 + 5)
        .attr('text-anchor', 'end')
        .attr('fill', '#94A3B8')
        .attr('font-size', 12)
        .attr('font-family', 'Inter, sans-serif')
        .attr('font-weight', 500)
        .text(feat.displayName);

      // Bar background
      svg.append('rect')
        .attr('x', labelW)
        .attr('y', y + 4)
        .attr('width', chartW)
        .attr('height', barH - 8)
        .attr('rx', 4)
        .attr('fill', '#161B26');

      // Contribution bar
      const bar = svg.append('rect')
        .attr('x', labelW + barX)
        .attr('y', y + 4)
        .attr('height', barH - 8)
        .attr('rx', 4)
        .attr('fill', color)
        .style('filter', `drop-shadow(0 0 4px ${color}66)`);

      if (animated) {
        bar.attr('width', 0)
          .transition()
          .delay(i * 80)
          .duration(500)
          .ease(d3.easeQuadOut)
          .attr('width', barW);
      } else {
        bar.attr('width', barW);
      }

      // Direction icon + contribution value
      const valueX = labelW + chartW + 8;
      svg.append('text')
        .attr('x', valueX)
        .attr('y', y + barH / 2 + 5)
        .attr('fill', color)
        .attr('font-size', 11)
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-weight', 600)
        .text(`${isRisk ? '+' : ''}${feat.contribution.toFixed(2)}`);
    });

    // Axis labels
    svg.append('text')
      .attr('x', labelW + zeroX - 4)
      .attr('y', totalH - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#334155')
      .attr('font-size', 10)
      .attr('font-family', 'Inter, sans-serif')
      .text('0');

    svg.append('text')
      .attr('x', labelW + 4)
      .attr('y', totalH - 8)
      .attr('fill', '#00D4FF')
      .attr('font-size', 10)
      .attr('font-family', 'Inter, sans-serif')
      .text('LOWERS RISK ←');

    svg.append('text')
      .attr('x', labelW + chartW - 4)
      .attr('y', totalH - 8)
      .attr('text-anchor', 'end')
      .attr('fill', '#EF4444')
      .attr('font-size', 10)
      .attr('font-family', 'Inter, sans-serif')
      .text('→ RAISES RISK');
  }, [features, baseScore, finalScore, animated]);

  return (
    <div className={styles.container} id="shap-waterfall">
      <div className={styles.header}>
        <span className={styles.title}>SHAP Feature Contributions</span>
        {finalScore !== undefined && (
          <span className={styles.finalScore}>
            Final: {(finalScore * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <svg ref={svgRef} width="100%" className={styles.svg} />
    </div>
  );
}
