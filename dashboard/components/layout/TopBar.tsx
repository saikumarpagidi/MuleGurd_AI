'use client';

import { useState, useEffect } from 'react';
import styles from './TopBar.module.css';

interface TopBarProps {
  sidebarCollapsed: boolean;
  title: string;
  subtitle?: string;
}

export default function TopBar({ sidebarCollapsed, title, subtitle }: TopBarProps) {
  const [time, setTime] = useState('');

  useEffect(() => {
    const tick = () =>
      setTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header
      className={`${styles.topbar} ${sidebarCollapsed ? styles.sidebarCollapsed : ''}`}
      id="topbar"
    >
      {/* Page Title */}
      <div className={styles.titleArea}>
        <h1 className={styles.title}>{title}</h1>
        {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
      </div>

      {/* Right strip — UTC clock + live dot only */}
      <div className={styles.metricsStrip}>
        <div className={styles.metric} id="topbar-clock">
          <span className={styles.metricLabel}>UTC</span>
          <span className={`${styles.metricValue} ${styles.mono}`}>{time}</span>
        </div>
        <div className={styles.liveDot} title="Live data" aria-label="Live data" />
      </div>
    </header>
  );
}

