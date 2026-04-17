"use client";

import { useRef, useEffect } from "react";
import { drawViridis } from "@/lib/colormaps";
import { CH_NAMES } from "@/lib/utils";

function ChannelThumb({ name, data }: { name: string; data: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) drawViridis(ref.current, data);
  }, [data]);
  return (
    <div className="flex-shrink-0 flex flex-col items-center gap-1.5">
      <canvas ref={ref} width={64} height={64} className="w-20 h-20 pixelated rounded-md border border-border hover:border-blue transition-colors" />
      <span className="text-[10px] text-text-3 font-medium">{CH_NAMES[name] || name}</span>
    </div>
  );
}

export default function ChannelStrip({ channels }: { channels: Record<string, number[][]> }) {
  return (
    <div className="flex gap-3.5 overflow-x-auto py-1">
      {Object.entries(channels).map(([key, data]) => (
        <ChannelThumb key={key} name={key} data={data} />
      ))}
    </div>
  );
}
