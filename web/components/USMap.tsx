"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import maplibregl from "maplibre-gl";
import type { LiveAssessment } from "@/lib/types";
import { assessLive, getActiveFires } from "@/lib/api";
import { RISK_COLORS, RISK_HEADLINES, envNarrative } from "@/lib/utils";

/* ── types ──────────────────────────────────────────────────────── */

interface HistoryItem {
  id: string;
  lat: number;
  lng: number;
  data: LiveAssessment;
  timestamp: Date;
}

/* ── thermal canvas → data URL ──────────────────────────────────── */

/* Render fire_prob as a smooth image overlay */
function fireToDataURL(fireProb: number[][]): string {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(64, 64);

  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const i = (y * 64 + x) * 4;
      const v = fireProb[y][x];

      // Fire colormap: transparent → yellow → orange → red
      let r: number, g: number, b: number;
      if (v < 0.3) {
        const t = v / 0.3;
        r = Math.round(255 * t);
        g = Math.round(200 * t);
        b = Math.round(50 * t);
      } else if (v < 0.6) {
        const t = (v - 0.3) / 0.3;
        r = 255;
        g = Math.round(200 - 120 * t);
        b = Math.round(50 - 40 * t);
      } else {
        const t = (v - 0.6) / 0.4;
        r = Math.round(255 - 55 * t);
        g = Math.round(80 - 60 * t);
        b = Math.round(10 + 20 * t);
      }

      img.data[i] = r;
      img.data[i + 1] = g;
      img.data[i + 2] = b;
      // Alpha: transparent for low values, solid for high
      img.data[i + 3] = v < 0.02 ? 0 : Math.round(Math.min(1, v * 2.5) * 200);
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

/* ── component ──────────────────────────────────────────────────── */

export default function USMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selected, setSelected] = useState<HistoryItem | null>(null);
  const [loading, setLoading] = useState<{ lat: number; lng: number } | null>(null);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [fireCount, setFireCount] = useState<number | null>(null);
  const [freshness, setFreshness] = useState<{ weather?: string; fires?: string } | null>(null);

  /* ── add overlay to map ───────────────────────────────────────── */

  const addOverlay = useCallback((map: maplibregl.Map, item: HistoryItem) => {
    const id = `overlay-${item.id}`;

    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);

    const b = item.data.bbox;
    const imageUrl = fireToDataURL(item.data.fire_prob);

    map.addSource(id, {
      type: "image",
      url: imageUrl,
      coordinates: [
        [b.west, b.north],
        [b.east, b.north],
        [b.east, b.south],
        [b.west, b.south],
      ],
    });

    map.addLayer({
      id,
      type: "raster",
      source: id,
      paint: { "raster-opacity": 0.75, "raster-fade-duration": 300 },
    }, "active-fires");
  }, []);

  /* ── initialize map ───────────────────────────────────────────── */

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          terrain: {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            attribution: "Esri, USGS, NOAA",
          },
        },
        layers: [
          {
            id: "terrain",
            type: "raster",
            source: "terrain",
            paint: {},
          },
        ],
      },
      center: [-98, 39],
      zoom: 4,
      maxBounds: [[-135, 20], [-55, 55]],
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    map.on("load", async () => {
      map.getCanvas().style.cursor = "crosshair";

      // Load active fires
      try {
        const fires = await getActiveFires();
        setFireCount(fires.features.length);

        map.addSource("fires", { type: "geojson", data: fires });

        // Glow layer
        map.addLayer({
          id: "active-fires-glow",
          type: "circle",
          source: "fires",
          paint: {
            "circle-radius": 8,
            "circle-color": "#ff4500",
            "circle-opacity": 0.15,
            "circle-blur": 1,
          },
        });

        // Core dot
        map.addLayer({
          id: "active-fires",
          type: "circle",
          source: "fires",
          paint: {
            "circle-radius": 3,
            "circle-color": "#ff4500",
            "circle-opacity": 0.8,
            "circle-stroke-width": 0.5,
            "circle-stroke-color": "#fff",
          },
        });

        // Fire hover popup
        const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });
        map.on("mouseenter", "active-fires", (e) => {
          map.getCanvas().style.cursor = "pointer";
          if (e.features?.length) {
            const p = e.features[0].properties;
            const coords = (e.features[0].geometry as GeoJSON.Point).coordinates;
            popup
              .setLngLat(coords as [number, number])
              .setHTML(`<div style="font-size:12px;font-family:system-ui;padding:2px 4px;"><strong>Active Fire</strong><br>Confidence: ${p?.confidence || "N/A"}<br>${p?.date || ""} ${p?.time || ""}</div>`)
              .addTo(map);
          }
        });
        map.on("mouseleave", "active-fires", () => {
          map.getCanvas().style.cursor = "crosshair";
          popup.remove();
        });
      } catch (e) {
        console.warn("Failed to load active fires:", e);
      }

      // Click to assess
      map.on("click", (e) => {
        const fires = map.queryRenderedFeatures(e.point, { layers: ["active-fires"] });
        if (fires.length > 0) return; // Don't assess on fire dot click

        const { lat, lng } = e.lngLat;
        if (lat >= 24.5 && lat <= 49.5 && lng >= -125 && lng <= -66) {
          runAssessment(lat, lng);
        }
      });
    });

    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  /* ── run assessment ───────────────────────────────────────────── */

  const runAssessment = useCallback(async (lat: number, lng: number) => {
    const map = mapRef.current;
    if (!map) return;

    setLoading({ lat, lng });
    setSelected(null);

    // Add a pulsing marker at click location
    const marker = new maplibregl.Marker({
      element: createPulsingDot(),
    }).setLngLat([lng, lat]).addTo(map);

    try {
      const data = await assessLive(lat, lng);
      const id = `${lat.toFixed(4)}-${lng.toFixed(4)}-${Date.now()}`;

      const item: HistoryItem = { id, lat, lng, data, timestamp: new Date() };

      // Add overlay to map
      addOverlay(map, item);

      // Update state
      setHistory((prev) => [item, ...prev]);
      setSelected(item);
      setFreshness({ weather: data.data_freshness.weather, fires: data.data_freshness.fire_detections });
    } catch (e) {
      console.error("Assessment failed:", e);
    } finally {
      marker.remove();
      setLoading(null);
    }
  }, [addOverlay]);

  /* ── select history item ──────────────────────────────────────── */

  const selectHistoryItem = useCallback((item: HistoryItem) => {
    setSelected(item);
    mapRef.current?.flyTo({ center: [item.lng, item.lat], zoom: 7, duration: 1000 });
  }, []);

  /* ── render ───────────────────────────────────────────────────── */

  const d = selected?.data;

  return (
    <div className="relative w-full overflow-hidden" style={{ height: "calc(100vh - 64px)" }}>
      {/* Map */}
      <div ref={mapContainer} className="absolute inset-0 w-full h-full" />

      {/* Title + instructions */}
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <h1 className="text-lg font-bold text-text tracking-tight bg-white/85 backdrop-blur-sm px-3 py-1.5 rounded-lg shadow-sm">
          PyroSight
        </h1>
        <p className="text-[11px] text-text-2 mt-1 bg-white/85 backdrop-blur-sm px-3 py-1 rounded-lg shadow-sm">
          Click anywhere to assess wildfire risk · {fireCount !== null ? `${fireCount.toLocaleString()} active fires detected` : "Loading fires..."}
        </p>
      </div>

      {/* Data freshness bar */}
      {freshness && (
        <div className="absolute bottom-4 left-4 z-10 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-1.5 shadow-sm text-[10px] font-mono text-text-3">
          Weather: {freshness.weather} · Fires: {freshness.fires}
        </div>
      )}

      {/* Loading indicator */}
      {loading && (
        <div className="absolute top-4 right-16 z-10 bg-white/90 backdrop-blur-sm rounded-lg px-4 py-2 shadow-sm flex items-center gap-2">
          <div className="w-3.5 h-3.5 border-2 border-border border-t-fire rounded-full animate-spin" />
          <span className="text-xs text-text-2">Fetching satellite data...</span>
        </div>
      )}

      {/* History panel (left) */}
      {history.length > 0 && (
        <div className="absolute top-4 left-4 z-10 mt-16 w-64">
          <button
            onClick={() => setHistoryOpen(!historyOpen)}
            className="w-full text-left bg-white/90 backdrop-blur-sm rounded-t-lg px-3 py-2 shadow-sm text-xs font-semibold text-text border-b border-border/50 flex items-center justify-between"
          >
            <span>Assessments ({history.length})</span>
            <span className="text-text-3">{historyOpen ? "▾" : "▸"}</span>
          </button>
          {historyOpen && (
            <div className="bg-white/90 backdrop-blur-sm rounded-b-lg shadow-sm max-h-64 overflow-y-auto">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => selectHistoryItem(item)}
                  className={`w-full text-left px-3 py-2 text-xs border-b border-border/30 last:border-b-0 hover:bg-bg-warm/50 transition-colors flex items-center gap-2 ${
                    selected?.id === item.id ? "bg-bg-fire/50" : ""
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: RISK_COLORS[item.data.risk_level] }}
                  />
                  <span className="font-mono truncate">
                    {item.lat.toFixed(2)}°, {item.lng.toFixed(2)}°
                  </span>
                  <span className="ml-auto text-text-3 flex-shrink-0">
                    {item.data.risk_level.charAt(0) + item.data.risk_level.slice(1).toLowerCase()}
                  </span>
                </button>
              ))}
              <button
                onClick={() => { setHistory([]); setSelected(null); }}
                className="w-full text-center px-3 py-1.5 text-[10px] text-text-3 hover:text-red transition-colors"
              >
                Clear all
              </button>
            </div>
          )}
        </div>
      )}

      {/* Detail panel (right slide-in) */}
      <div
        className="absolute top-0 right-0 h-full w-[380px] bg-white border-l border-border shadow-xl z-20 overflow-y-auto transition-transform duration-300"
        style={{ transform: d ? "translateX(0)" : "translateX(100%)" }}
      >
        {d && (
          <div className="p-6">
            <button
              onClick={() => setSelected(null)}
              className="absolute top-4 right-4 text-text-3 hover:text-text text-lg leading-none"
            >
              ×
            </button>

            {/* Header */}
            <p className="text-[10px] font-mono text-text-3 uppercase tracking-wider mb-1">Risk Assessment</p>
            <h2 className="text-lg font-bold mb-0.5">
              {selected!.lat.toFixed(4)}°N, {Math.abs(selected!.lng).toFixed(4)}°W
            </h2>
            <p className="text-[11px] text-text-3 mb-4">
              {d.assessment_date} · Weather from {d.data_freshness.weather}
            </p>

            {/* Risk badge */}
            <div className="flex items-center gap-3 mb-5">
              <span
                className="font-mono text-[11px] font-semibold tracking-wider px-3.5 py-1.5 rounded-md text-white"
                style={{ backgroundColor: RISK_COLORS[d.risk_level] }}
              >
                {d.risk_level}
              </span>
              <span className="font-mono text-xl font-bold">{d.confidence}%</span>
              <span className="text-[10px] text-text-3 uppercase">confidence</span>
            </div>

            {/* Narrative */}
            <p className="text-sm text-text-2 leading-relaxed mb-5">
              {RISK_HEADLINES[d.risk_level]}. {envNarrative(d.environment)}
            </p>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-2.5 mb-5">
              {[
                ["Mean Fire", `${(d.stats.mean_fire_prob * 100).toFixed(1)}%`, "text-fire"],
                ["Peak Fire", `${(d.stats.max_fire_prob * 100).toFixed(1)}%`, "text-fire"],
                ["High Risk", `${d.stats.high_risk_percent.toFixed(1)}%`, "text-amber"],
                ["Uncertainty", `${(d.stats.mean_uncertainty * 100).toFixed(1)}%`, "text-blue"],
                ["Wind", `${d.environment.wind_speed_mean} m/s`, "text-text"],
                ["Max Temp", `${d.environment.temp_max.toFixed(0)}°C`, "text-text"],
                ["ERC", `${d.environment.erc_mean.toFixed(0)}`, "text-text"],
                ["Prior Fire", d.environment.has_prev_fire ? "Yes" : "No", d.environment.has_prev_fire ? "text-red" : "text-green"],
              ].map(([label, val, color]) => (
                <div key={label} className="bg-bg-warm rounded-lg px-3 py-2">
                  <p className="text-[9px] text-text-3 uppercase tracking-wider">{label}</p>
                  <p className={`text-sm font-semibold font-mono ${color}`}>{val}</p>
                </div>
              ))}
            </div>

            {/* Thermal thumbnail */}
            <div className="mb-5">
              <p className="text-[9px] text-text-3 uppercase tracking-wider mb-2">Fire Probability Map</p>
              <PanelThermal fireProb={d.fire_prob} />
              <div className="h-1.5 rounded-full mt-2" style={{ background: "linear-gradient(90deg, #1e1b4b, #7c3aed, #db2777, #f97316, #facc15, #fefce8)" }} />
              <div className="flex justify-between text-[9px] text-text-3 font-mono mt-0.5">
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>

            {/* Environment */}
            <div className="grid grid-cols-3 gap-2 mb-5">
              {[
                [d.environment.humidity.toFixed(1) + " g/kg", "Humidity"],
                [d.environment.precipitation.toFixed(2) + " mm", "Precip"],
                [Math.round(d.environment.ndvi_mean).toString(), "NDVI"],
                [d.environment.elevation_range[0].toFixed(0) + "–" + d.environment.elevation_range[1].toFixed(0) + "m", "Elevation"],
                [d.environment.temp_min.toFixed(0) + "°C", "Min Temp"],
                [d.stats.fire_pixel_count.toString(), "Fire Pixels"],
              ].map(([val, label]) => (
                <div key={label} className="bg-bg-warm rounded px-2 py-1.5 text-center">
                  <p className="font-mono text-xs font-semibold">{val}</p>
                  <p className="text-[8px] text-text-3">{label}</p>
                </div>
              ))}
            </div>

            {/* Coverage info */}
            <div className="text-[10px] text-text-3 font-mono bg-bg-warm rounded-lg px-3 py-2">
              Coverage: {(d.bbox.north - d.bbox.south).toFixed(2)}° × {(d.bbox.east - d.bbox.west).toFixed(2)}° (~64 × 64 km)
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── panel thermal thumbnail ──────────────────────────────────── */

function PanelThermal({ fireProb }: { fireProb: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    // Import dynamically to avoid circular issues
    import("@/lib/colormaps").then(({ drawThermal: dt }) => {
      if (ref.current) dt(ref.current, fireProb);
    });
  }, [fireProb]);
  return (
    <canvas
      ref={ref}
      width={64}
      height={64}
      className="w-full rounded-lg border border-border"
      style={{ imageRendering: "pixelated" }}
    />
  );
}

/* ── pulsing dot element ──────────────────────────────────────── */

function createPulsingDot(): HTMLDivElement {
  const el = document.createElement("div");
  el.style.cssText = `
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(232, 89, 12, 0.4);
    border: 2px solid #E8590C;
    animation: pulse-ring 1.2s ease-in-out infinite;
  `;
  const style = document.createElement("style");
  style.textContent = `
    @keyframes pulse-ring {
      0% { transform: scale(1); opacity: 1; }
      100% { transform: scale(2.5); opacity: 0; }
    }
  `;
  el.appendChild(style);
  return el;
}
