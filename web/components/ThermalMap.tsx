"use client";

import { useRef, useEffect, useState } from "react";
import { drawThermal, drawUncertainty, drawGrid } from "@/lib/colormaps";

export default function ThermalMap({
  fireProb,
  uncertainty,
  size = 384,
}: {
  fireProb: number[][];
  uncertainty?: number[][];
  size?: number;
}) {
  const fireRef = useRef<HTMLCanvasElement>(null);
  const uncRef = useRef<HTMLCanvasElement>(null);
  const gridRef = useRef<HTMLCanvasElement>(null);
  const [showUnc, setShowUnc] = useState(false);
  const [showGrid, setShowGrid] = useState(false);

  useEffect(() => {
    if (fireRef.current) drawThermal(fireRef.current, fireProb);
    if (uncRef.current && uncertainty) drawUncertainty(uncRef.current, uncertainty);
    if (gridRef.current) drawGrid(gridRef.current);
  }, [fireProb, uncertainty]);

  return (
    <div>
      <div className="bg-white border border-border rounded-xl p-6 shadow-sm">
        <div className="relative mx-auto" style={{ width: size, height: size }}>
          <canvas ref={fireRef} width={64} height={64} className="absolute inset-0 w-full h-full pixelated rounded" />
          <canvas ref={uncRef} width={64} height={64} className="absolute inset-0 w-full h-full pixelated rounded transition-opacity" style={{ opacity: showUnc ? 0.7 : 0 }} />
          <canvas ref={gridRef} width={size} height={size} className="absolute inset-0 w-full h-full rounded transition-opacity" style={{ opacity: showGrid ? 1 : 0 }} />
        </div>
      </div>
      <p className="text-sm text-text-3 italic mt-3 leading-relaxed">
        <strong className="not-italic text-text-2">Fig. 1</strong> — Predicted fire spread probability. Thermal colormap: black (0%) through indigo, magenta, orange to white-hot (100%).
      </p>
      <div className="flex gap-2 mt-3">
        <button
          onClick={() => setShowUnc(!showUnc)}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-lg border transition-colors ${showUnc ? "bg-bg-blue text-blue border-blue/20" : "bg-white text-text-2 border-border hover:bg-bg-warm"}`}
        >
          {showUnc ? "Hide" : "Show"} Uncertainty
        </button>
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-lg border transition-colors ${showGrid ? "bg-bg-warm text-text border-border-l" : "bg-white text-text-2 border-border hover:bg-bg-warm"}`}
        >
          Grid
        </button>
      </div>
      <div className="mt-3">
        <div className="h-2 rounded-full thermal-gradient border border-border" />
        <div className="flex justify-between mt-1 text-[10px] font-mono text-text-3">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
      </div>
    </div>
  );
}
