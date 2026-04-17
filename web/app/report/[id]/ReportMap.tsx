"use client";

import { useRef, useEffect } from "react";
import { drawThermal } from "@/lib/colormaps";

export default function ReportMap({ fireProb }: { fireProb: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) drawThermal(ref.current, fireProb);
  }, [fireProb]);
  return (
    <canvas
      ref={ref}
      width={64}
      height={64}
      className="w-[280px] h-[280px] pixelated rounded-lg border border-border"
    />
  );
}
