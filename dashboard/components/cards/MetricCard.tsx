'use client';

import { useEffect, useRef } from 'react';
import styles from './MetricCard.module.css';

interface MetricCardProps {
  id: string;
  label: string;
  value: string;
  subValue?: string;
  icon?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendLabel?: string;
  color?: string;
  highlight?: boolean;
  className?: string;
  delay?: number;
}

export default function MetricCard({
  id, label, value, subValue, icon,
  trend, trendLabel,
  color = 'var(--cyan)', highlight = false, className = '', delay = 0,
}: MetricCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    el.style.opacity = '0';
    el.style.transform = 'translateY(14px)';
    const t = setTimeout(() => {
      el.style.transition = 'opacity 420ms ease, transform 420ms ease';
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, delay);
    return () => clearTimeout(t);
  }, [delay]);

  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→';
  const trendClass =
    trend === 'up' ? styles.trendUp :
    trend === 'down' ? styles.trendDown :
    styles.trendNeutral;

  return (
    <div
      ref={cardRef}
      id={id}
      className={`${styles.card} ${highlight ? styles.highlight : ''} ${className}`}
      style={{ '--card-accent': color } as React.CSSProperties}
    >
      {/* Top accent line */}
      <div className={styles.accentLine} />

      {/* Glow blob */}
      <div className={styles.glowBlob} />

      <div className={styles.header}>
        <div className={styles.labelRow}>
          {icon && <span className={styles.icon}>{icon}</span>}
          <span className={styles.label}>{label}</span>
        </div>
        {trend && trendLabel && (
          <span className={`${styles.trend} ${trendClass}`}>
            {trendIcon} {trendLabel}
          </span>
        )}
      </div>

      <div className={styles.value}>{value}</div>
      {subValue && <div className={styles.subValue}>{subValue}</div>}
    </div>
  );
}
