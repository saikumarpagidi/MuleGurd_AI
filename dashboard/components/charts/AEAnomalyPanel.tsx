'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { AEFeature, AIEngineAeFeature } from '@/lib/types';
import styles from './AEAnomalyPanel.module.css';

interface AEAnomalyPanelProps {
  features: (AEFeature | AIEngineAeFeature)[];
  finalScore?: number;
  animated?: boolean;
}

export default function AEAnomalyPanel({
  features,
  finalScore,
  animated = true,
}: AEAnomalyPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !features.length) return;

    // Sort by Z-score descending
    const sorted = [...features].sort((a, b) => b.reconstruction_z_score - a.reconstruction_z_score);
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

    // X scale: from 0 to max Z-score (with a minimum max of 2.0 to give perspective)
    const maxZ = Math.max(2.0, ...sorted.map(f => f.reconstruction_z_score)) * 1.15;
    const xScale = d3.scaleLinear().domain([0, maxZ]).range([0, chartW]);

    // Zero line
    svg.append('line')
      .attr('x1', labelW)
      .attr('y1', paddingT)
      .attr('x2', labelW)
      .attr('y2', totalH - paddingB)
      .attr('stroke', 'rgba(255,255,255,0.08)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 4');

    sorted.forEach((feat, i) => {
      const y = paddingT + i * (barH + gap);
      
      let color = '#64748B'; // Default (No Anomaly)
      if (feat.direction === 'RAISES_ANOMALY') color = '#EF4444'; // Red for high anomaly
      else if (feat.direction === 'SMALL_ANOMALY_CONTRIBUTION') color = '#F59E0B'; // Orange for small anomaly
      
      const barW = xScale(feat.reconstruction_z_score);

      const humanName = ('displayName' in feat && feat.displayName) 
        ? feat.displayName 
        : feat.feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

      // Feature name label
      svg.append('text')
        .attr('x', labelW - 8)
        .attr('y', y + barH / 2 + 5)
        .attr('text-anchor', 'end')
        .attr('fill', '#94A3B8')
        .attr('font-size', 12)
        .attr('font-family', 'Inter, sans-serif')
        .attr('font-weight', 500)
        .text(humanName);

      // Bar background
      svg.append('rect')
        .attr('x', labelW)
        .attr('y', y + 4)
        .attr('width', chartW)
        .attr('height', barH - 8)
        .attr('rx', 4)
        .attr('fill', '#161B26');

      // Anomaly bar
      const bar = svg.append('rect')
        .attr('x', labelW)
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

      // Z-score value
      const valueX = labelW + chartW + 8;
      svg.append('text')
        .attr('x', valueX)
        .attr('y', y + barH / 2 + 5)
        .attr('fill', color)
        .attr('font-size', 11)
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-weight', 600)
        .text(feat.reconstruction_z_score.toFixed(2));
    });

    // Axis labels
    svg.append('text')
      .attr('x', labelW - 4)
      .attr('y', totalH - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#334155')
      .attr('font-size', 10)
      .attr('font-family', 'Inter, sans-serif')
      .text('0 (Normal)');

    svg.append('text')
      .attr('x', labelW + chartW - 4)
      .attr('y', totalH - 8)
      .attr('text-anchor', 'end')
      .attr('fill', '#EF4444')
      .attr('font-size', 10)
      .attr('font-family', 'Inter, sans-serif')
      .text('HIGHLY ANOMALOUS →');
  }, [features, finalScore, animated]);

  return (
    <div className={styles.container} id="ae-anomaly-panel">
      <div className={styles.header}>
        <span className={styles.title}>Autoencoder Top Anomalies</span>
        {finalScore !== undefined && (
          <span className={styles.finalScore}>
            AE Score: {(finalScore * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <svg ref={svgRef} width="100%" className={styles.svg} />
    </div>
  );
}
