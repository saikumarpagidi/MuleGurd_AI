'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { ROCPoint, PRPoint, ModelMetrics } from '@/lib/types';
import styles from './AUCChart.module.css';

interface AUCChartProps {
  rocCurve: ROCPoint[];
  prCurve: PRPoint[];
  rocAuc: number;
  prAuc: number;
  modelName: string;
  activeTab?: 'roc' | 'pr';
}

function drawROC(
  container: SVGSVGElement,
  data: ROCPoint[],
  auc: number,
  w: number,
  h: number
) {
  const margin = { top: 20, right: 20, bottom: 40, left: 44 };
  const innerW = w - margin.left - margin.right;
  const innerH = h - margin.top - margin.bottom;

  const svg = d3.select(container);
  svg.selectAll('*').remove();

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  // Grid
  const gridLines = [0.2, 0.4, 0.6, 0.8];
  gridLines.forEach(v => {
    g.append('line')
      .attr('x1', x(v)).attr('y1', 0)
      .attr('x2', x(v)).attr('y2', innerH)
      .attr('stroke', 'rgba(255,255,255,0.04)').attr('stroke-width', 1);
    g.append('line')
      .attr('x1', 0).attr('y1', y(v))
      .attr('x2', innerW).attr('y2', y(v))
      .attr('stroke', 'rgba(255,255,255,0.04)').attr('stroke-width', 1);
  });

  // Random diagonal
  g.append('line')
    .attr('x1', x(0)).attr('y1', y(0))
    .attr('x2', x(1)).attr('y2', y(1))
    .attr('stroke', 'rgba(255,255,255,0.12)')
    .attr('stroke-dasharray', '4 4')
    .attr('stroke-width', 1);

  // AUC fill
  const area = d3.area<ROCPoint>()
    .x(d => x(d.fpr)).y0(innerH).y1(d => y(d.tpr))
    .curve(d3.curveMonotoneX);

  g.append('path')
    .datum(data)
    .attr('d', area)
    .attr('fill', 'rgba(0,212,255,0.06)');

  // ROC curve line
  const line = d3.line<ROCPoint>()
    .x(d => x(d.fpr)).y(d => y(d.tpr))
    .curve(d3.curveMonotoneX);

  const path = g.append('path')
    .datum(data)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', '#00D4FF')
    .attr('stroke-width', 2.5)
    .style('filter', 'drop-shadow(0 0 4px rgba(0,212,255,0.5))');

  // Animate
  const totalLen = (path.node() as SVGPathElement)?.getTotalLength() ?? 0;
  path
    .attr('stroke-dasharray', totalLen)
    .attr('stroke-dashoffset', totalLen)
    .transition().duration(900).ease(d3.easeLinear)
    .attr('stroke-dashoffset', 0);

  // AUC label
  g.append('text')
    .attr('x', innerW * 0.55)
    .attr('y', innerH * 0.25)
    .attr('fill', '#00D4FF')
    .attr('font-size', 13)
    .attr('font-weight', 700)
    .attr('font-family', 'Inter, sans-serif')
    .text(`AUC = ${auc.toFixed(4)}`);

  // Axes
  const axisBottom = d3.axisBottom(x).ticks(5).tickFormat(d => `${+d * 100}%`);
  const axisLeft   = d3.axisLeft(y).ticks(5).tickFormat(d => `${+d * 100}%`);

  g.append('g').attr('transform', `translate(0,${innerH})`).call(axisBottom)
    .call(a => a.select('.domain').remove())
    .call(a => a.selectAll('text').attr('fill', '#64748B').attr('font-size', 10))
    .call(a => a.selectAll('line').attr('stroke', 'rgba(255,255,255,0.06)'));

  g.append('g').call(axisLeft)
    .call(a => a.select('.domain').remove())
    .call(a => a.selectAll('text').attr('fill', '#64748B').attr('font-size', 10))
    .call(a => a.selectAll('line').attr('stroke', 'rgba(255,255,255,0.06)'));

  // Axis labels
  g.append('text').attr('x', innerW / 2).attr('y', innerH + 36)
    .attr('text-anchor', 'middle').attr('fill', '#64748B').attr('font-size', 11)
    .text('False Positive Rate');
  g.append('text')
    .attr('transform', `rotate(-90)`)
    .attr('x', -innerH / 2).attr('y', -36)
    .attr('text-anchor', 'middle').attr('fill', '#64748B').attr('font-size', 11)
    .text('True Positive Rate');
}

export default function AUCChart({ rocCurve, prCurve, rocAuc, prAuc, modelName, activeTab = 'roc' }: AUCChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const w = svgRef.current.clientWidth || 440;
    const h = 260;
    svgRef.current.setAttribute('height', String(h));
    if (activeTab === 'roc') {
      drawROC(svgRef.current, rocCurve, rocAuc, w, h);
    }
  }, [rocCurve, prCurve, rocAuc, prAuc, activeTab]);

  return (
    <div className={styles.container} id={`auc-chart-${modelName.toLowerCase().replace(/\s/g, '-')}`}>
      <svg ref={svgRef} width="100%" />
    </div>
  );
}
