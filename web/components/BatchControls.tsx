"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { getSamples, batchAssess } from "@/lib/api";
import { RISK_COLORS } from "@/lib/utils";
import type { BatchResponse, RiskLevel } from "@/lib/types";
import RiskBadge from "@/components/RiskBadge";

const LEVELS: RiskLevel[] = ["CRITICAL", "HIGH", "MODERATE", "LOW"];

function drawChart(
  canvas: HTMLCanvasElement,
  distribution: Record<RiskLevel, number>,
  total: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const barHeight = 28;
  const gap = 12;
  const labelWidth = 90;
  const countWidth = 50;
  const barAreaWidth = w - labelWidth - countWidth - 16;

  LEVELS.forEach((level, i) => {
    const y = i * (barHeight + gap);
    const count = distribution[level] || 0;
    const pct = total > 0 ? count / total : 0;

    // Label
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#8A8882";
    ctx.textBaseline = "middle";
    ctx.fillText(level, 0, y + barHeight / 2);

    // Track
    ctx.fillStyle = "#F3F1EC";
    ctx.beginPath();
    ctx.roundRect(labelWidth, y, barAreaWidth, barHeight, 4);
    ctx.fill();

    // Bar
    if (pct > 0) {
      ctx.fillStyle = RISK_COLORS[level];
      ctx.beginPath();
      ctx.roundRect(labelWidth, y, Math.max(barAreaWidth * pct, 8), barHeight, 4);
      ctx.fill();
    }

    // Count
    ctx.fillStyle = "#1A1917";
    ctx.font = "13px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(String(count), w, y + barHeight / 2);
    ctx.textAlign = "left";
  });
}

export default function BatchControls() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [results, setResults] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const runBatch = useCallback(async (mode: "assess" | "random50" | "all") => {
    setLoading(true);
    setError(null);
    setResults(null);
    setProgress(0);
    setTotal(0);

    try {
      const { samples } = await getSamples();
      let ids: number[];

      if (mode === "random50") {
        const shuffled = [...samples].sort(() => Math.random() - 0.5);
        ids = shuffled.slice(0, 50).map((s) => s.id);
      } else if (mode === "all") {
        ids = samples.map((s) => s.id);
      } else {
        // "assess" — regions with fire
        ids = samples.filter((s) => s.has_fire).map((s) => s.id);
        if (ids.length === 0) ids = samples.slice(0, 50).map((s) => s.id);
      }

      setTotal(ids.length);

      const data = await batchAssess(ids, (processed, t) => {
        setProgress(processed);
        setTotal(t);
      });

      setResults(data);
      sessionStorage.setItem("pyrosight_batch", JSON.stringify(data));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch assessment failed");
    } finally {
      setLoading(false);
    }
  }, []);

  // Draw chart when results change
  useEffect(() => {
    if (!results || !canvasRef.current) return;
    drawChart(
      canvasRef.current,
      results.summary.risk_distribution,
      results.summary.assessed,
    );

    const handleResize = () => {
      if (canvasRef.current && results) {
        drawChart(
          canvasRef.current,
          results.summary.risk_distribution,
          results.summary.assessed,
        );
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [results]);

  const progressPct = total > 0 ? (progress / total) * 100 : 0;

  return (
    <section className="max-w-5xl mx-auto px-6 md:px-10 py-20">
      {/* Section header */}
      <div className="mb-12">
        <div className="w-12 h-0.5 bg-text mb-6" />
        <p className="font-mono text-xs uppercase tracking-widest text-text-3 mb-3">
          Batch Assessment
        </p>
        <h2 className="font-serif text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Assess Fire Regions
        </h2>
        <p className="text-text-2 max-w-2xl leading-relaxed">
          Run the PyroSight model across multiple regions simultaneously.
          Results include risk classification, confidence scores, and peak
          fire probability for each region.
        </p>
      </div>

      {/* Buttons */}
      <div className="flex flex-wrap gap-3 mb-8">
        <button
          onClick={() => runBatch("assess")}
          disabled={loading}
          className="px-5 py-2.5 bg-fire text-white text-sm font-medium rounded-lg hover:bg-fire-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Assess Fire Regions
        </button>
        <button
          onClick={() => runBatch("random50")}
          disabled={loading}
          className="px-5 py-2.5 border border-border text-text text-sm font-medium rounded-lg hover:bg-bg-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Random 50
        </button>
        <button
          onClick={() => runBatch("all")}
          disabled={loading}
          className="px-5 py-2.5 border border-border text-text text-sm font-medium rounded-lg hover:bg-bg-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          All 1,689
        </button>
      </div>

      {/* Progress */}
      {loading && (
        <div className="mb-10">
          <div className="h-1.5 rounded-full bg-bg-warm overflow-hidden mb-2">
            <div
              className="h-full bg-fire rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p className="font-mono text-xs text-text-3">
            Processing {progress} of {total} regions...
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-bg-red border border-red/20 rounded-lg p-4 mb-10">
          <p className="text-red text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-16">
          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-8 py-6 border-y border-border">
            <div className="text-center">
              <div className="font-mono text-2xl font-semibold text-text">
                {results.summary.assessed}
              </div>
              <div className="text-xs text-text-3 mt-1">Assessed</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-2xl font-semibold text-fire">
                {(results.summary.avg_fire_prob * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text-3 mt-1">Avg Fire Prob</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-2xl font-semibold text-red">
                {(results.summary.risk_distribution.CRITICAL || 0) +
                  (results.summary.risk_distribution.HIGH || 0)}
              </div>
              <div className="text-xs text-text-3 mt-1">High+ Risk</div>
            </div>
          </div>

          {/* Risk Distribution Chart */}
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-text-3 mb-6">
              Risk Distribution
            </p>
            <canvas
              ref={canvasRef}
              className="w-full"
              style={{ height: "176px" }}
            />
          </div>

          {/* Top Risk List */}
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-text-3 mb-6">
              Top Risk Regions
            </p>
            <div className="space-y-0 border-t border-border">
              {results.summary.top_risk.map((item, i) => (
                <Link
                  key={item.sample_id}
                  href={`/assess/${item.sample_id}`}
                  className="flex items-center gap-4 py-3.5 px-2 border-b border-border hover:bg-bg-warm transition-colors group"
                >
                  <span className="font-mono text-xs text-text-4 w-6 text-right">
                    {i + 1}
                  </span>
                  <span className="font-mono text-sm text-text flex-1">
                    Region #{item.sample_id}
                  </span>
                  <RiskBadge level={item.risk_level} />
                  <span className="font-mono text-sm text-text-2 w-20 text-right">
                    {(item.max_fire_prob * 100).toFixed(1)}%
                  </span>
                  <span className="text-text-4 group-hover:text-fire transition-colors">
                    &rarr;
                  </span>
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
