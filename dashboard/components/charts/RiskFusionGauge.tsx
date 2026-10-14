'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { ComponentScores } from '@/lib/types';
import { getDecisionColor, getDecisionFromScore, formatScore } from '@/lib/utils';
import styles from './RiskFusionGauge.module.css';

interface RiskFusionGaugeProps {
  finalScore: number;
  componentScores: ComponentScores;
  size?: number;
  animated?: boolean;
}

const WEIGHTS = {
  tgn_score:    { label: 'TGN',        weight: 0.50, color: '#7C3AED' },
  lgbm_score:   { label: 'LightGBM',   weight: 0.30, color: '#00D4FF' },
  ae_score:     { label: 'Autoencoder', weight: 0.15, color: '#F59E0B' },
  device_score: { label: 'Device',      weight: 0.05, color: '#10B981' },
} as const;

export default function RiskFusionGauge({
  finalScore,
  componentScores,
  size = 240,
  animated = true,
}: RiskFusionGaugeProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const decision = getDecisionFromScore(finalScore);
  const decisionColor = getDecisionColor(decision);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const cx = size / 2;
    const cy = size / 2 + 20;
    const R  = size * 0.38;
    const startAngle = -Math.PI * 0.85;
    const endAngle   =  Math.PI * 0.85;
    const arcRange   = endAngle - startAngle;

    // Background track
    const bgArc = d3.arc<number>()
      .innerRadius(R - 18)
      .outerRadius(R)
      .startAngle(startAngle)
      .endAngle(endAngle)
      .cornerRadius(9);

    svg.append('path')
      .attr('d', bgArc(0) ?? '')
      .attr('transform', `translate(${cx},${cy})`)
      .attr('fill', '#161B26')
      .attr('stroke', 'rgba(255,255,255,0.05)')
      .attr('stroke-width', 1);

    // Score arc
    const targetEnd = startAngle + arcRange * finalScore;
    const scoreArc = d3.arc<number>()
      .innerRadius(R - 18)
      .outerRadius(R)
      .startAngle(startAngle)
      .cornerRadius(9);

    const scorePath = svg.append('path')
      .attr('transform', `translate(${cx},${cy})`)
      .attr('fill', decisionColor)
      .style('filter', `drop-shadow(0 0 8px ${decisionColor}88)`);

    if (animated) {
      scorePath.datum(startAngle)
        .attr('d', d => scoreArc.endAngle(d)(0) ?? '')
        .transition()
        .duration(900)
        .ease(d3.easeQuadOut)
        .attrTween('d', function () {
          const interp = d3.interpolate(startAngle, targetEnd);
          return (t: number) => scoreArc.endAngle(interp(t))(0) ?? '';
        });
    } else {
      scorePath.attr('d', scoreArc.endAngle(targetEnd)(0) ?? '');
    }

    // Tick marks for thresholds
    const thresholds = [
      { pct: 0.45, label: '45%', color: '#F59E0B' },
      { pct: 0.75, label: '75%', color: '#EF4444' },
    ];

    thresholds.forEach(({ pct, color }) => {
      const angle = startAngle + arcRange * pct - Math.PI / 2;
      const x1 = cx + (R - 22) * Math.cos(angle);
      const y1 = cy + (R - 22) * Math.sin(angle);
      const x2 = cx + (R + 4)  * Math.cos(angle);
      const y2 = cy + (R + 4)  * Math.sin(angle);
      svg.append('line')
        .attr('x1', x1).attr('y1', y1)
        .attr('x2', x2).attr('y2', y2)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '2 2');
    });

    // Center text — score
    const scoreGroup = svg.append('g').attr('transform', `translate(${cx},${cy})`);

    const pct = Math.round(finalScore * 100);
    const displayNum = animated ? 0 : pct;

    const numText = scoreGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.1em')
      .attr('fill', decisionColor)
      .attr('font-size', size * 0.175)
      .attr('font-weight', 800)
      .attr('font-family', 'Inter, sans-serif')
      .attr('letter-spacing', '-2')
      .text(displayNum);

    if (animated) {
      numText.transition().duration(900).ease(d3.easeQuadOut)
        .tween('text', function () {
          const interp = d3.interpolateNumber(0, pct);
          return function (t: number) {
            (this as SVGTextElement).textContent = String(Math.round(interp(t)));
          };
        });
    }

    scoreGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.4em')
      .attr('fill', '#64748B')
      .attr('font-size', size * 0.065)
      .attr('font-weight', 600)
      .attr('font-family', 'Inter, sans-serif')
      .attr('letter-spacing', '1')
      .text('RISK SCORE');

    // Decision label below
    scoreGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '2.8em')
      .attr('fill', decisionColor)
      .attr('font-size', size * 0.07)
      .attr('font-weight', 700)
      .attr('font-family', 'Inter, sans-serif')
      .attr('letter-spacing', '2')
      .text(decision)
      .style('filter', `drop-shadow(0 0 4px ${decisionColor})`);

  }, [finalScore, componentScores, size, animated, decisionColor, decision]);

  return (
    <div className={styles.container}>
      <svg ref={svgRef} width={size} height={size * 0.82} id="risk-fusion-gauge" />

      {/* Component Score Bars */}
      <div className={styles.components}>
        {(Object.entries(WEIGHTS) as [keyof ComponentScores, typeof WEIGHTS[keyof typeof WEIGHTS]][]).map(
          ([key, { label, weight, color }]) => {
            const score = componentScores[key] ?? 0;
            return (
              <div key={key} className={styles.componentRow} id={`gauge-${key}`}>
                <span className={styles.componentLabel}>{label}</span>
                <div className={styles.componentBar}>
                  <div
                    className={styles.componentFill}
                    style={{ '--fill': `${score * 100}%`, '--color': color } as React.CSSProperties}
                  />
                </div>
                <span className={styles.componentScore} style={{ color }}>
                  {formatScore(score)}
                </span>
                <span className={styles.componentWeight}>×{weight}</span>
              </div>
            );
          }
        )}
      </div>
    </div>
  );
}
