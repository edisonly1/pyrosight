import type { RiskLevel } from "@/lib/types";

const STYLES: Record<RiskLevel, string> = {
  CRITICAL: "bg-red text-white",
  HIGH: "bg-fire text-white",
  MODERATE: "bg-amber text-white",
  LOW: "bg-green text-white",
};

export default function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span className={`inline-block font-mono text-[11px] font-semibold tracking-wider px-3.5 py-1.5 rounded-md ${STYLES[level]}`}>
      {level}
    </span>
  );
}
