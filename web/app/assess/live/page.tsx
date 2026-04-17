"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { assessLive } from "@/lib/api";
import type { LiveAssessment } from "@/lib/types";
import RiskBanner from "@/components/RiskBanner";
import ThermalMap from "@/components/ThermalMap";
import StatCard from "@/components/StatCard";
import EnvGrid from "@/components/EnvGrid";
import ChannelStrip from "@/components/ChannelStrip";
import UncertaintyHist from "@/components/UncertaintyHist";
import { envNarrative } from "@/lib/utils";

export default function LiveAssessPage() {
  return (
    <Suspense fallback={<div className="max-w-4xl mx-auto px-10 py-20 text-center"><div className="w-8 h-8 border-2 border-border border-t-fire rounded-full animate-spin mx-auto" /></div>}>
      <LiveAssessContent />
    </Suspense>
  );
}

function LiveAssessContent() {
  const params = useSearchParams();
  const lat = parseFloat(params.get("lat") || "0");
  const lng = parseFloat(params.get("lng") || "0");

  const [data, setData] = useState<LiveAssessment | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!lat || !lng) { setError("No coordinates provided"); setLoading(false); return; }

    setLoading(true);
    setError(null);
    assessLive(lat, lng)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [lat, lng]);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-10 py-20 text-center">
        <div className="w-8 h-8 border-2 border-border border-t-fire rounded-full animate-spin mx-auto mb-4" />
        <p className="text-text-2">
          Fetching real-time environmental data for {lat.toFixed(4)}°N, {Math.abs(lng).toFixed(4)}°W...
        </p>
        <p className="text-xs text-text-3 mt-2">
          Downloading GRIDMET weather, elevation, vegetation, and fire detection data
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-10 py-20 text-center">
        <p className="text-red text-lg font-semibold mb-2">Assessment Failed</p>
        <p className="text-text-2 mb-6">{error}</p>
        <Link href="/map" className="text-fire no-underline hover:underline">← Back to Map</Link>
      </div>
    );
  }

  if (!data) return null;

  const s = data.stats;
  const uf = data.uncertainty.flat();
  const uncMean = uf.reduce((a, b) => a + b, 0) / uf.length;
  const uncMax = Math.max(...uf);
  const uncHigh = uf.filter((v) => v > 0.5).length;

  return (
    <main className="max-w-4xl mx-auto px-10 py-12">
      {/* Header */}
      <div className="mb-8">
        <div className="w-12 h-[3px] bg-text mb-6" />
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-3">Live Risk Assessment</p>
        <h1 className="font-serif text-4xl font-semibold tracking-tight mb-2">
          {lat.toFixed(4)}°N, {Math.abs(lng).toFixed(4)}°W
        </h1>
        <div className="flex gap-4 text-xs text-text-3 font-mono">
          <span>Assessment: {data.assessment_date}</span>
          <span>Weather: {data.data_freshness.weather}</span>
          <span>Fire data: {data.data_freshness.fire_detections}</span>
        </div>
        <div className="text-xs text-text-3 mt-1 font-mono">
          Coverage: {(data.bbox.north - data.bbox.south).toFixed(1)}° lat × {(data.bbox.east - data.bbox.west).toFixed(1)}° lng (~64 × 64 km)
        </div>
      </div>

      {/* Risk Banner */}
      <div className="mb-10">
        <RiskBanner level={data.risk_level} confidence={data.confidence} />
      </div>

      {/* Fire Probability Map */}
      <section className="mb-12">
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-4">Fire Spread Prediction</p>
        <ThermalMap fireProb={data.fire_prob} uncertainty={data.uncertainty} size={384} />
      </section>

      {/* Key Metrics */}
      <section className="mb-12">
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-6">Key Metrics</p>
        <div className="grid grid-cols-3 gap-8">
          <StatCard value={`${(s.mean_fire_prob * 100).toFixed(1)}%`} label="Mean Fire Probability" color="fire" />
          <StatCard value={`${(s.max_fire_prob * 100).toFixed(1)}%`} label="Peak Fire Probability" color="fire" />
          <StatCard value={`${s.high_risk_percent.toFixed(1)}%`} label="High Risk Area" color="amber" />
        </div>
      </section>

      {/* Environmental Conditions */}
      <section className="mb-12">
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-4">Conditions on the Ground</p>
        <p className="text-text-2 text-lg leading-relaxed mb-6">{envNarrative(data.environment)}</p>
        <EnvGrid env={data.environment} />
      </section>

      {/* Uncertainty */}
      <section className="mb-12">
        <div className="bg-bg-white border border-border rounded-xl p-6 shadow-sm">
          <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-4">Uncertainty Analysis</p>
          <div className="grid grid-cols-3 gap-4 mb-2">
            <div>
              <span className="text-xs text-text-3">Mean</span>
              <p className="font-mono text-lg font-semibold text-blue">{(uncMean * 100).toFixed(2)}%</p>
            </div>
            <div>
              <span className="text-xs text-text-3">Max</span>
              <p className="font-mono text-lg font-semibold text-blue">{(uncMax * 100).toFixed(2)}%</p>
            </div>
            <div>
              <span className="text-xs text-text-3">&gt;50% Pixels</span>
              <p className="font-mono text-lg font-semibold text-blue">{uncHigh}</p>
            </div>
          </div>
          <UncertaintyHist data={data.uncertainty} />
        </div>
      </section>

      {/* Input Channels */}
      {data.input_channels && (
        <section className="mb-12">
          <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-2">Model Inputs</p>
          <p className="text-sm text-text-3 mb-4">12 environmental channels from real satellite and weather data at 1 km resolution.</p>
          <ChannelStrip channels={data.input_channels} />
        </section>
      )}

      {/* About */}
      <section className="mb-12">
        <div className="bg-bg-warm rounded-xl p-6 text-sm text-text-2 leading-relaxed">
          <strong className="text-text">About this assessment.</strong>{" "}
          This prediction uses <strong>real environmental data</strong> fetched from GRIDMET (weather),
          GMTED2010 (elevation), MODIS NDVI (vegetation), GPWv4 (population), and NASA FIRMS (fire detections).
          The model (DualBranchUNetEDL) processes these 12 channels through dual fuel/weather branches with
          cross-attention, fuses with Rothermel fire-spread physics, and outputs calibrated fire probability
          with evidential uncertainty. Weather data has a ~2-day latency from GRIDMET.
        </div>
      </section>

      {/* Actions */}
      <div className="flex gap-3">
        <Link href="/map" className="px-5 py-2.5 rounded-lg text-sm font-semibold border border-border text-text hover:bg-bg-warm transition-colors no-underline">
          ← Back to Map
        </Link>
        <Link href={`/report/live?lat=${lat}&lng=${lng}`} className="px-5 py-2.5 rounded-lg text-sm font-semibold border border-border text-text hover:bg-bg-warm transition-colors no-underline">
          Generate Report →
        </Link>
      </div>
    </main>
  );
}
