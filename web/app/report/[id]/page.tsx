import Link from "next/link";
import type { Assessment, ModelInfo } from "@/lib/types";
import { RISK_HEADLINES, RISK_DESCS, envNarrative, envItems } from "@/lib/utils";
import RiskBadge from "@/components/RiskBadge";
import ReportMap from "./ReportMap";
import PrintButton from "./PrintButton";

async function getData(id: string) {
  const [assessRes, modelRes] = await Promise.all([
    fetch(`http://localhost:8000/api/samples/${id}/assess`, { cache: "no-store" }),
    fetch(`http://localhost:8000/api/model/info`, { cache: "no-store" }),
  ]);
  if (!assessRes.ok) throw new Error("Assessment failed");
  const assessment: Assessment = await assessRes.json();
  const modelInfo: ModelInfo = await modelRes.json();
  return { assessment, modelInfo };
}

export default async function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { assessment: d, modelInfo: mi } = await getData(id);
  const s = d.stats;
  const e = d.environment;
  const now = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  return (
    <>
      {/* Print bar */}
      <div className="no-print flex items-center justify-between px-10 py-4 bg-bg-white border-b border-border">
        <div className="flex items-center gap-3">
          <Link href={`/assess/${id}`} className="text-text-2 text-sm no-underline hover:text-text">← Back</Link>
          <span className="text-text-4">|</span>
          <span className="font-semibold text-sm">Risk Assessment Report</span>
        </div>
        <PrintButton />
      </div>

      <main className="max-w-3xl mx-auto px-10 py-12 print:p-0 print:max-w-none">
        {/* Header */}
        <div className="border-t-4 border-text pt-6 mb-10">
          <h1 className="font-serif text-[28px] font-semibold mb-1">PyroSight Risk Assessment Report</h1>
          <p className="font-mono text-xs text-text-3">Region #{d.sample_id} · Generated {now}</p>
        </div>

        {/* Risk */}
        <div className={`flex items-center gap-5 p-5 rounded-xl border mb-10 ${
          d.risk_level === "CRITICAL" ? "bg-bg-red border-red/15" :
          d.risk_level === "HIGH" ? "bg-bg-fire border-fire/15" :
          d.risk_level === "MODERATE" ? "bg-bg-amber border-amber/15" :
          "bg-bg-green border-green/15"
        }`}>
          <RiskBadge level={d.risk_level} />
          <div className="flex-1">
            <h2 className="font-serif text-xl font-semibold mb-1">{RISK_HEADLINES[d.risk_level]}</h2>
            <p className="text-sm text-text-2">{RISK_DESCS[d.risk_level]}</p>
          </div>
          <div className="text-center">
            <div className="font-mono text-2xl font-semibold">{d.confidence}%</div>
            <div className="text-[10px] text-text-3 uppercase tracking-wider">Confidence</div>
          </div>
        </div>

        {/* Map + Stats */}
        <p className="font-mono text-[11px] uppercase tracking-[2px] text-text-3 mb-4 pb-2 border-b border-border">
          Fire Probability Map & Assessment Summary
        </p>
        <div className="grid grid-cols-[auto_1fr] gap-8 mb-10 items-start">
          <div>
            <ReportMap fireProb={d.fire_prob} />
            <div className="mt-2">
              <div className="h-1.5 rounded-full thermal-gradient" />
              <div className="flex justify-between mt-1 text-[10px] font-mono text-text-3">
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>
          </div>
          <div className="space-y-0">
            {[
              ["Mean Fire Probability", `${(s.mean_fire_prob * 100).toFixed(1)}%`, "text-fire"],
              ["Peak Fire Probability", `${(s.max_fire_prob * 100).toFixed(1)}%`, "text-fire"],
              ["High Risk Area", `${s.high_risk_percent.toFixed(1)}%`, "text-amber"],
              ["Predicted Fire Pixels", `${s.fire_pixel_count}`, ""],
              ["Mean Uncertainty", `${(s.mean_uncertainty * 100).toFixed(1)}%`, "text-blue"],
              ["Model Confidence", `${d.confidence}%`, "text-green"],
            ].map(([label, val, color]) => (
              <div key={label} className="flex justify-between items-center py-3.5 border-b border-bg-warm last:border-b-0">
                <span className="text-sm text-text-2">{label}</span>
                <span className={`font-mono text-sm font-semibold ${color}`}>{val}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Environment */}
        <p className="font-mono text-[11px] uppercase tracking-[2px] text-text-3 mb-4 pb-2 border-b border-border">
          Environmental Conditions
        </p>
        <p className="text-sm text-text-2 mb-4 leading-relaxed">{envNarrative(e)}</p>
        <div className="grid grid-cols-3 gap-3 mb-10">
          {envItems(e).map(([val, label]) => (
            <div key={label} className="bg-bg-warm rounded-lg p-3.5 text-center">
              <div className="font-mono text-base font-semibold">{val}</div>
              <div className="text-[10px] text-text-3 mt-1">{label}</div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-12 pt-4 border-t border-border text-xs text-text-3 font-mono leading-relaxed">
          <p>Model: {mi.name} · {(mi.parameters / 1e6).toFixed(2)}M parameters · Epoch {mi.training.epoch}</p>
          <p>Physics: Rothermel fire spread · CA post-processing · Bayesian evidential fusion</p>
          <p>Dataset: Next Day Wildfire Spread (2012–2020) · {mi.architecture.input_channels} channels · {mi.architecture.image_size}×{mi.architecture.image_size} resolution</p>
          <p className="mt-3">This report is generated by PyroSight, a research system for wildfire risk assessment. Predictions are based on historical satellite data and should not be used as the sole basis for operational fire management decisions.</p>
        </div>
      </main>

    </>
  );
}
