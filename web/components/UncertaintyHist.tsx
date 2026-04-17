"use client";

import { useRef, useEffect } from "react";
import { drawHist } from "@/lib/colormaps";

export default function UncertaintyHist({ data }: { data: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) drawHist(ref.current, data);
  }, [data]);
  return (
    <div className="h-14 bg-bg-warm rounded-lg overflow-hidden mt-3">
      <canvas ref={ref} width={300} height={56} className="w-full h-full" />
    </div>
  );
}
