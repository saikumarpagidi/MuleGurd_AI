import { NextResponse } from 'next/server';
import type { AIEngineHealth } from '@/lib/types';

const AI_ENGINE_URL = process.env.AI_ENGINE_URL ?? 'http://127.0.0.1:8002';

export async function GET() {
  const t0 = performance.now();
  try {
    const res = await fetch(`${AI_ENGINE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(3000),
      cache: 'no-store',
    });

    const ping = Math.round(performance.now() - t0);

    if (res.ok) {
      const data = await res.json();
      const health: AIEngineHealth = {
        status: data.status === 'ok' ? 'ok' : 'degraded',
        lgbm_model_loaded: Boolean(data.lgbm_model_loaded),
        autoencoder_model_loaded: Boolean(data.autoencoder_model_loaded),
        shap_explainer_ready: Boolean(data.shap_explainer_ready),
        redis_connected: Boolean(data.redis_connected),
        uptime_seconds: Number(data.uptime_seconds || 0),
        endpoint_url: AI_ENGINE_URL,
        ping_ms: ping,
      };
      return NextResponse.json(health);
    } else {
      return NextResponse.json<AIEngineHealth>({
        status: 'degraded',
        lgbm_model_loaded: false,
        autoencoder_model_loaded: false,
        shap_explainer_ready: false,
        redis_connected: false,
        uptime_seconds: 0,
        endpoint_url: AI_ENGINE_URL,
        ping_ms: ping,
      });
    }
  } catch (error) {
    console.error('[AI Engine Health Route] Fetch failed:', error);
    return NextResponse.json<AIEngineHealth>({
      status: 'offline',
      lgbm_model_loaded: false,
      autoencoder_model_loaded: false,
      shap_explainer_ready: false,
      redis_connected: false,
      uptime_seconds: 0,
      endpoint_url: AI_ENGINE_URL,
      ping_ms: Math.round(performance.now() - t0),
    });
  }
}
