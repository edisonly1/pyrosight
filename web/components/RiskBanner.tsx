import type { RiskLevel } from "@/lib/types";
import { RISK_HEADLINES, RISK_DESCS } from "@/lib/utils";
import RiskBadge from "./RiskBadge";

const BG: Record<RiskLevel, string> = {
  CRITICAL: "bg-bg-red border-red/15",
  HIGH: "bg-bg-fire border-fire/15",
  MODERATE: "bg-bg-amber border-amber/15",
  LOW: "bg-bg-green border-green/15",
};

export default function RiskBanner({ level, confidence }: { level: RiskLevel; confidence: number }) {
  return (
    <div className={`flex items-center gap-6 p-6 rounded-xl border ${BG[level]}`}>
      <RiskBadge level={level} />
      <div className="flex-1">
        <h2 className="font-serif text-[22px] font-semibold mb-1">{RISK_HEADLINES[level]}</h2>
        <p className="text-text-2 text-sm">{RISK_DESCS[level]}</p>
      </div>
      <div className="text-center whitespace-nowrap">
        <div className="font-mono text-[28px] font-semibold">{confidence}%</div>
        <div className="text-[11px] text-text-3 uppercase tracking-wider">Confidence</div>
      </div>
    </div>
  );
}
