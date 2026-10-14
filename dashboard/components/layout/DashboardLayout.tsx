'use client';

import { useState } from 'react';
import Sidebar from '@/components/layout/Sidebar';
import TopBar from '@/components/layout/TopBar';
import styles from './DashboardLayout.module.css';

interface DashboardLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export default function DashboardLayout({ children, title, subtitle }: DashboardLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={styles.shell}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
      <div className={`${styles.main} ${collapsed ? styles.collapsed : ''}`}>
        <TopBar sidebarCollapsed={collapsed} title={title} subtitle={subtitle} />
        <main className={styles.content}>
          {children}
        </main>
      </div>
    </div>
  );
}
