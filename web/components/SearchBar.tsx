"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { geocode } from "@/lib/api";

interface Result {
  lat: number;
  lng: number;
  display_name: string;
}

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const router = useRouter();

  // Debounced search
  const search = useCallback((q: string) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (q.length < 2) { setResults([]); setOpen(false); return; }

    timerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        // Check if it's a lat,lng pair
        const coordMatch = q.match(/^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$/);
        if (coordMatch) {
          const lat = parseFloat(coordMatch[1]);
          const lng = parseFloat(coordMatch[2]);
          if (24.5 <= lat && lat <= 49.5 && -125 <= lng && lng <= -66) {
            setResults([{ lat, lng, display_name: `${lat.toFixed(4)}, ${lng.toFixed(4)}` }]);
            setOpen(true);
            setLoading(false);
            return;
          }
        }
        const r = await geocode(q);
        setResults(r);
        setOpen(r.length > 0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
  }, []);

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const selectResult = (r: Result) => {
    setQuery(r.display_name.split(",")[0]);
    setOpen(false);
    router.push(`/assess/live?lat=${r.lat}&lng=${r.lng}`);
  };

  return (
    <div ref={ref} className="relative">
      <div className="flex items-center gap-2 bg-bg-warm border border-border rounded-lg px-3 py-1.5">
        <svg className="w-4 h-4 text-text-3 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); search(e.target.value); }}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="Search any US location..."
          className="bg-transparent outline-none text-sm text-text placeholder:text-text-3 w-48 lg:w-64"
        />
        {loading && (
          <div className="w-3.5 h-3.5 border-2 border-border border-t-fire rounded-full animate-spin flex-shrink-0" />
        )}
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-bg-white border border-border rounded-lg shadow-lg z-50 overflow-hidden max-w-sm min-w-[280px]">
          {results.map((r, i) => (
            <button
              key={i}
              onClick={() => selectResult(r)}
              className="w-full text-left px-4 py-2.5 text-sm text-text hover:bg-bg-warm transition-colors border-b border-bg-warm last:border-b-0 flex items-start gap-2"
            >
              <span className="text-text-3 mt-0.5 flex-shrink-0">&#x1F4CD;</span>
              <span className="line-clamp-2">{r.display_name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
