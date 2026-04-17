"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getPins, removePin, clearPins } from "@/lib/pins";
import { compare } from "@/lib/api";
import type { Assessment } from "@/lib/types";
import RiskBadge from "@/components/RiskBadge";
import ThermalMap from "@/components/ThermalMap";
import EnvGrid from "@/components/EnvGrid";

export default function ComparePage() {
  const [pins, setPins] = useState<number[]>([]);
  const [results, setResults] = useState<Assessment[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const p = getPins();
    setPins(p);
    if (p.length >= 2) {
      setLoading(true);
      compare(p)
        .then((data) => setResults(data.results))
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  }, []);

  const handleUnpin = (id: number) => {
    removePin(id);
    window.dispatchEvent(new Event("pinsUpdated"));
    const newPins = getPins();
    setPins(newPins);
    if (newPins.length < 2) {
      setResults(null);
    } else {
      setLoading(true);
      compare(newPins)
        .then((data) => setResults(data.results))
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  };

  const handleClear = () => {
    clearPins();
    window.dispatchEvent(new Event("pinsUpdated"));
    setPins([]);
    setResults(null);
  };

  return (
    <main className="max-w-6xl mx-auto px-10 py-16">
      <div className="mb-10">
        <div className="w-12 h-[3px] bg-text mb-6" />
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-3">Comparative Analysis</p>
        <h1 className="font-serif text-4xl font-semibold tracking-tight mb-2">Side-by-Side Assessment</h1>
        <p className="text-lg text-text-2 max-w-xl">
          Compare environmental conditions, fire probabilities, and risk classifications across multiple regions.
        </p>
      </div>

      {pins.length < 2 && !loading && (
        <div className="text-center py-20 max-w-md mx-auto">
          <div className="text-5xl opacity-20 mb-4">&#9878;</div>
          <h3 className="font-serif text-xl font-semibold mb-2">No regions pinned</h3>
          <p className="text-text-2 text-sm mb-5">
            Pin 2–3 regions from the{" "}
            <Link href="/assess" className="text-fire no-underline hover:underline">Assessment</Link>{" "}
            or{" "}
            <Link href="/map" className="text-fire no-underline hover:underline">Map</Link>{" "}
            pages to compare them side by side.
          </p>
          <Link href="/assess" className="inline-block px-6 py-2.5 bg-fire text-white rounded-lg font-semibold text-sm no-underline hover:bg-fire-dark transition-colors">
            Go to Assessment
          </Link>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center gap-3 py-16 text-text-2">
          <div className="w-5 h-5 border-2 border-border border-t-fire rounded-full animate-spin" />
          <span>Running comparative assessment…</span>
        </div>
      )}

      {error && (
        <p className="text-red text-center py-8">Failed: {error}</p>
      )}

      {results && results.length >= 2 && (
        <>
          <div className="flex justify-end mb-5">
            <button onClick={handleClear} className="text-sm text-text-2 hover:text-text transition-colors">
              Clear All Pins
            </button>
          </div>

          <div className={`grid gap-6 mb-10 ${results.length === 3 ? "grid-cols-3" : "grid-cols-2"}`}>
            {results.map((r) => (
              <div key={r.sample_id} className="bg-bg-white border border-border rounded-xl shadow-sm overflow-hidden">
                {/* Header */}
                <div className="p-5 border-b border-border flex items-center justify-between">
                  <div>
                    <h3 className="font-serif text-lg font-semibold">Region #{r.sample_id}</h3>
                    <div className="mt-1.5"><RiskBadge level={r.risk_level} /></div>
                  </div>
                  <div className="text-center">
                    <div className="font-mono text-xl font-semibold">{r.confidence}%</div>
                    <div className="text-[10px] text-text-3 uppercase tracking-wider">confidence</div>
                  </div>
                </div>

                {/* Map */}
                <div className="p-5 border-b border-border flex justify-center">
                  <ThermalMap fireProb={r.fire_prob} uncertainty={r.uncertainty} size={200} />
                </div>

                {/* Stats */}
                <div className="p-5 border-b border-border">
                  <div className="space-y-3">
                    <div className="flex justify-between"><span className="text-text-2 text-sm">Mean Prob</span><span className="font-mono text-sm font-semibold text-fire">{(r.stats.mean_fire_prob * 100).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-text-2 text-sm">Peak Prob</span><span className="font-mono text-sm font-semibold text-fire">{(r.stats.max_fire_prob * 100).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-text-2 text-sm">High Risk</span><span className="font-mono text-sm font-semibold text-amber">{r.stats.high_risk_percent.toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-text-2 text-sm">Uncertainty</span><span className="font-mono text-sm font-semibold text-blue">{(r.stats.mean_uncertainty * 100).toFixed(1)}%</span></div>
                  </div>
                </div>

                {/* Env */}
                {r.environment && (
                  <div className="p-5 border-b border-border">
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        [`${r.environment.wind_speed_mean} m/s`, "Wind"],
                        [`${r.environment.temp_max.toFixed(0)}°C`, "Temp"],
                        [`${r.environment.erc_mean.toFixed(0)}`, "ERC"],
                        [r.environment.has_prev_fire ? "Yes" : "No", "Prior Fire"],
                      ].map(([val, label]) => (
                        <div key={label} className="bg-bg-warm rounded-lg p-2.5 text-center">
                          <div className="font-mono text-sm font-semibold">{val}</div>
                          <div className="text-[10px] text-text-3">{label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="p-5 flex gap-2">
                  <Link href={`/assess/${r.sample_id}`} className="flex-1 text-center px-4 py-2 bg-fire text-white rounded-lg text-sm font-semibold no-underline hover:bg-fire-dark transition-colors">
                    View Full →
                  </Link>
                  <button onClick={() => handleUnpin(r.sample_id)} className="px-4 py-2 border border-border rounded-lg text-sm font-medium text-text-2 hover:bg-bg-warm transition-colors">
                    Unpin
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <footer className="text-center text-sm text-text-3 pt-12 pb-8 border-t border-border">
        PyroSight · DualBranchUNetEDL · Evidential Deep Learning & Rothermel Physics
      </footer>
    </main>
  );
}
