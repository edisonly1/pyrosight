import type { Environment, RiskLevel } from "./types";

export const RISK_HEADLINES: Record<RiskLevel, string> = {
  CRITICAL: "Critical fire spread risk detected",
  HIGH: "High fire spread risk detected",
  MODERATE: "Moderate fire activity possible",
  LOW: "Low risk — minimal fire spread expected",
};

export const RISK_DESCS: Record<RiskLevel, string> = {
  CRITICAL: "Significant fire spread predicted with high confidence. Multiple areas show >80% probability of next-day fire activity. Immediate monitoring recommended.",
  HIGH: "Notable fire spread likely in several areas. Enhanced monitoring and preparedness advised.",
  MODERATE: "Some fire activity possible but limited in extent. Standard monitoring should continue.",
  LOW: "Environmental conditions do not favor significant fire spread in the next 24 hours.",
};

export const RISK_COLORS: Record<RiskLevel, string> = {
  CRITICAL: "#DC2626",
  HIGH: "#E8590C",
  MODERATE: "#D97706",
  LOW: "#16A34A",
};

export const RISK_BG: Record<RiskLevel, string> = {
  CRITICAL: "bg-bg-red",
  HIGH: "bg-bg-fire",
  MODERATE: "bg-bg-amber",
  LOW: "bg-bg-green",
};

export const CH_NAMES: Record<string, string> = {
  elevation: "Elevation",
  th: "Temperature",
  vs: "Wind Speed",
  tmmn: "Min Temp",
  tmmx: "Max Temp",
  sph: "Humidity",
  pr: "Precipitation",
  pdsi: "PDSI",
  NDVI: "NDVI",
  population: "Population",
  erc: "Fire Energy",
  PrevFireMask: "Prior Fire",
};

export function envNarrative(env: Environment): string {
  const parts: string[] = [];
  if (env.wind_speed_mean > 6) parts.push(`Strong winds at ${env.wind_speed_mean} m/s`);
  else if (env.wind_speed_mean > 3) parts.push(`Moderate winds at ${env.wind_speed_mean} m/s`);
  else parts.push(`Light winds at ${env.wind_speed_mean} m/s`);

  parts.push(`with temperatures reaching ${env.temp_max.toFixed(0)}°C`);

  if (env.precipitation < 0.01) parts.push("and no measurable precipitation");
  else parts.push(`and ${env.precipitation.toFixed(1)} mm precipitation`);

  if (env.erc_mean > 70)
    parts.push(`. The Energy Release Component reads ${env.erc_mean.toFixed(0)}, indicating extremely dry fuel conditions.`);
  else if (env.erc_mean > 40)
    parts.push(`. The Energy Release Component reads ${env.erc_mean.toFixed(0)}, indicating moderate fire danger.`);
  else
    parts.push(`. The Energy Release Component reads ${env.erc_mean.toFixed(0)}, indicating lower fire danger.`);

  return parts.join(" ");
}

export function envItems(env: Environment): [string, string][] {
  return [
    [`${env.wind_speed_mean} m/s`, "Wind Speed"],
    [`${env.temp_max.toFixed(0)}°C`, "Max Temp"],
    [`${env.temp_min.toFixed(0)}°C`, "Min Temp"],
    [`${env.humidity.toFixed(1)} g/kg`, "Humidity"],
    [`${env.precipitation.toFixed(2)} mm`, "Precip"],
    [`${env.erc_mean.toFixed(0)}`, "Fire Energy"],
    [`${Math.round(env.ndvi_mean)}`, "NDVI"],
    [`${env.elevation_range[0].toFixed(0)}–${env.elevation_range[1].toFixed(0)}m`, "Elevation"],
    [env.has_prev_fire ? "YES" : "NO", "Prior Fire"],
  ];
}
