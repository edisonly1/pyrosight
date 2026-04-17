import type { SamplesResponse, Assessment, ModelInfo, BatchResponse, LiveAssessment } from "./types";

const BASE = typeof window === "undefined"
  ? (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
  : "";

export async function getModelInfo(): Promise<ModelInfo> {
  const r = await fetch(`${BASE}/api/model/info`);
  return r.json();
}

export async function getSamples(): Promise<SamplesResponse> {
  const r = await fetch(`${BASE}/api/samples`);
  return r.json();
}

export async function assess(id: number): Promise<Assessment> {
  const r = await fetch(`${BASE}/api/samples/${id}/assess`);
  if (!r.ok) throw new Error(`Assess failed: ${r.status}`);
  return r.json();
}

export async function compare(ids: number[]): Promise<{ results: Assessment[] }> {
  const r = await fetch(`${BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_ids: ids }),
  });
  if (!r.ok) throw new Error(`Compare failed: ${r.status}`);
  return r.json();
}

export async function getActiveFires(): Promise<GeoJSON.FeatureCollection> {
  const r = await fetch(`${BASE}/api/fires/active`);
  return r.json();
}

export async function geocode(query: string): Promise<{ lat: number; lng: number; display_name: string }[]> {
  const r = await fetch(`${BASE}/api/geocode?q=${encodeURIComponent(query)}`);
  if (!r.ok) return [];
  return r.json();
}

export async function assessLive(lat: number, lng: number, date?: string): Promise<LiveAssessment> {
  const r = await fetch(`${BASE}/api/assess/live`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lng, date }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `Assess failed: ${r.status}`);
  }
  return r.json();
}

export async function batchAssess(
  ids: number[],
  onProgress?: (processed: number, total: number) => void,
): Promise<BatchResponse> {
  const CHUNK = 20;
  let allResults: BatchResponse["results"] = [];

  for (let i = 0; i < ids.length; i += CHUNK) {
    const chunk = ids.slice(i, i + CHUNK);
    const r = await fetch(`${BASE}/api/batch/assess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_ids: chunk }),
    });
    if (!r.ok) throw new Error(`Batch failed: ${r.status}`);
    const data = await r.json();
    allResults = allResults.concat(data.results);
    onProgress?.(Math.min(i + CHUNK, ids.length), ids.length);
  }

  const dist = { CRITICAL: 0, HIGH: 0, MODERATE: 0, LOW: 0 } as Record<string, number>;
  let probSum = 0;
  for (const r of allResults) {
    dist[r.risk_level] = (dist[r.risk_level] || 0) + 1;
    probSum += r.stats?.mean_fire_prob || 0;
  }
  const top10 = [...allResults]
    .sort((a, b) => (b.stats?.max_fire_prob || 0) - (a.stats?.max_fire_prob || 0))
    .slice(0, 10);

  return {
    results: allResults,
    summary: {
      assessed: allResults.length,
      risk_distribution: dist as BatchResponse["summary"]["risk_distribution"],
      avg_fire_prob: allResults.length ? +(probSum / allResults.length).toFixed(4) : 0,
      top_risk: top10.map((r) => ({
        sample_id: r.sample_id,
        risk_level: r.risk_level,
        max_fire_prob: r.stats?.max_fire_prob || 0,
        confidence: r.confidence,
      })),
    },
  };
}
