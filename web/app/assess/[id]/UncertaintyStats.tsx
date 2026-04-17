"use client";

export default function UncertaintyStats({ uncertainty }: { uncertainty: number[][] }) {
  let sum = 0;
  let max = 0;
  let count = 0;
  let highCount = 0;

  for (const row of uncertainty) {
    for (const v of row) {
      sum += v;
      count++;
      if (v > max) max = v;
      if (v > 0.3) highCount++;
    }
  }

  const mean = count > 0 ? sum / count : 0;
  const highPct = count > 0 ? highCount / count : 0;

  const stats: { value: string; label: string }[] = [
    { value: (mean * 100).toFixed(1) + "%", label: "Mean Uncertainty" },
    { value: (max * 100).toFixed(1) + "%", label: "Max Uncertainty" },
    { value: (highPct * 100).toFixed(1) + "%", label: "High-Unc Pixels" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map((s) => (
        <div key={s.label} className="text-center">
          <div className="font-mono text-2xl font-semibold text-text">{s.value}</div>
          <div className="text-xs text-text-3 mt-1">{s.label}</div>
        </div>
      ))}
    </div>
  );
}
