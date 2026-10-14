import type { Metadata } from 'next';
import '../styles/globals.css';
import '../styles/animations.css';

export const metadata: Metadata = {
  title: {
    default: 'MuleGuard.AI — Investigator Dashboard',
    template: '%s | MuleGuard.AI',
  },
  description:
    'Enterprise AI-powered Anti-Money Laundering detection platform. Real-time transaction scoring with LightGBM, Autoencoder, and Temporal Graph Networks.',
  keywords: ['AML', 'Anti-Money Laundering', 'AI', 'Fraud Detection', 'Financial Crime'],
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
