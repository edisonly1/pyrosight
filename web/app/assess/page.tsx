"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import type { Sample } from "@/lib/types";
import { getSamples } from "@/lib/api";

type Filter = "all" | "fire" | "nofire";

export default function AssessPage() {
  const router = useRouter();
  const [samples, setSamples] = useState<Sample[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSamples()
      .then((res) => {
        setSamples(res.samples);
        if (res.samples.length > 0) setSelectedId(res.samples[0].id);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = samples.filter((s) => {
    if (filter === "fire") return s.has_fire;
    if (filter === "nofire") return !s.has_fire;
    return true;
  });

  const buttons: { key: Filter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "fire", label: "With Fire" },
    { key: "nofire", label: "No Fire" },
  ];

  return (
    <main className="max-w-2xl mx-auto px-6 py-20">
      <p className="font-mono text-[11px] uppercase tracking-widest text-text-3 mb-2">
        Risk Assessment
      </p>
      <h1 className="font-serif text-5xl md:text-6xl font-bold tracking-tight text-text mb-10">
        Assess a Region
      </h1>

      {/* Filter buttons */}
      <div className="flex gap-2 mb-6">
        {buttons.map((b) => (
          <button
            key={b.key}
            onClick={() => {
              setFilter(b.key);
              setSelectedId(null);
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === b.key
                ? "bg-fire text-white"
                : "bg-bg-warm text-text-2 hover:bg-bg-hover"
            }`}
          >
            {b.label}
          </button>
        ))}
      </div>

      {/* Sample selector */}
      <label className="block text-sm text-text-2 mb-2">Select a sample region</label>
      <select
        value={selectedId ?? ""}
        onChange={(e) => setSelectedId(Number(e.target.value))}
        className="w-full border border-border rounded-lg px-4 py-3 bg-bg-white text-text font-mono text-sm focus:outline-none focus:ring-2 focus:ring-fire/40 mb-8"
        disabled={loading}
      >
        {loading && <option>Loading samples...</option>}
        {!loading && filtered.length === 0 && <option>No samples found</option>}
        {filtered.map((s) => (
          <option key={s.id} value={s.id}>
            Region #{s.id} &mdash; {s.has_fire ? `Fire (${(s.fire_fraction * 100).toFixed(1)}%)` : "No fire"} &mdash; ({s.lat.toFixed(2)}, {s.lng.toFixed(2)})
          </option>
        ))}
      </select>

      {/* Action button */}
      <button
        onClick={() => {
          if (selectedId != null) router.push(`/assess/${selectedId}`);
        }}
        disabled={selectedId == null || loading}
        className="w-full py-4 rounded-xl bg-fire text-white font-semibold text-lg transition-colors hover:bg-fire-dark disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Assess Region
      </button>

      <p className="text-xs text-text-3 mt-4 text-center">
        Runs the PyroSight model on the selected region and returns a full fire-spread risk assessment.
      </p>
    </main>
  );
}
