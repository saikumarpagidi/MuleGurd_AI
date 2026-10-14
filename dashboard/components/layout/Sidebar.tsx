'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import styles from './Sidebar.module.css';

const NAV_SECTIONS = [
  {
    heading: 'Investigation',
    items: [
      { href: '/dashboard/simulator',           label: 'Full Simulator',   icon: '▶', id: 'nav-simulator'    },
      { href: '/dashboard/ai-engine-simulator', label: 'AI Engine Studio', icon: '⚡', id: 'nav-ai-simulator' },
      { href: '/dashboard/explorer',            label: 'Graph Explorer',   icon: '◎', id: 'nav-explorer'     },
    ],
  },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''}`} id="sidebar">
      {/* Brand */}
      <div className={styles.brand}>
        <div className={styles.logo}>
          <div className={styles.logoIconWrap}>
            <span className={styles.logoIcon}>⬡</span>
          </div>
          {!collapsed && (
            <div className={styles.logoText}>
              <span className={styles.logoName}>MuleGuard</span>
              <span className={styles.logoSuffix}>.AI Platform</span>
            </div>
          )}
        </div>
        <button
          className={styles.collapseBtn}
          onClick={onToggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          id="sidebar-toggle"
        >
          <span
            style={{
              transform: collapsed ? 'rotate(180deg)' : 'none',
              display: 'inline-block',
              transition: '250ms',
            }}
          >
            ‹
          </span>
        </button>
      </div>

      {/* Live Status Pill */}
      {!collapsed && (
        <div className={styles.statusPill}>
          <span className={styles.statusDot} />
          <span className={styles.statusText}>Graph &amp; AI Connected</span>
          <span className={styles.statusBadge}>5/5</span>
        </div>
      )}

      {/* Nav Items */}
      <nav className={styles.nav} aria-label="Main navigation">
        {NAV_SECTIONS.map(section => (
          <div key={section.heading} className={styles.navSection}>
            {!collapsed && (
              <span className={styles.navSectionHeading}>{section.heading}</span>
            )}
            {section.items.map(item => {
              const isActive =
                pathname === item.href ||
                (item.href !== '/dashboard' && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  id={item.id}
                  className={`${styles.navItem} ${isActive ? styles.active : ''}`}
                  title={collapsed ? item.label : undefined}
                >
                  <span className={styles.navIcon}>{item.icon}</span>
                  {!collapsed && (
                    <span className={styles.navLabel}>{item.label}</span>
                  )}
                  {isActive && <span className={styles.activeBar} />}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Bottom Section */}
      <div className={styles.bottom}>
        {!collapsed ? (
          <div className={styles.versionBadge}>
            <div className={styles.versionRow}>
              <span className={styles.versionDot} />
              <span className={styles.versionLabel}>v2.0.0-alpha</span>
            </div>
            <span className={styles.buildLabel}>Sprint 8 · IBM AML Dataset</span>
          </div>
        ) : (
          <div className={styles.versionBadge}>
            <span className={styles.versionLabel}>v2</span>
          </div>
        )}
      </div>
    </aside>
  );
}
