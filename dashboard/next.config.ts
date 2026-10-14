import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  allowedDevOrigins: ['172.16.0.2', '10.226.43.215'],
  async rewrites() {
    return [
      {
        source: '/proxy/score',
        destination: process.env.AI_ENGINE_URL
          ? `${process.env.AI_ENGINE_URL}/score`
          : 'http://127.0.0.1:8002/score',
      },
      {
        source: '/proxy/graph-score',
        destination: process.env.GRAPH_SERVICE_URL
          ? `${process.env.GRAPH_SERVICE_URL}/graph-score`
          : 'http://127.0.0.1:8003/graph-score',
      },
      {
        source: '/proxy/subgraph/:path*',
        destination: process.env.GRAPH_SERVICE_URL
          ? `${process.env.GRAPH_SERVICE_URL}/subgraph/:path*`
          : 'http://127.0.0.1:8003/subgraph/:path*',
      },
    ];
  },
};

export default nextConfig;
