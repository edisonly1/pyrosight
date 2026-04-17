export interface Sample {
  id: number;
  has_fire: boolean;
  fire_pixel_count: number;
  valid_pixel_count: number;
  fire_fraction: number;
  lat: number;
  lng: number;
}

export interface SamplesResponse {
  count: number;
  samples: Sample[];
}

export interface RiskStats {
  risk_level: RiskLevel;
  confidence: number;
  mean_fire_prob: number;
  max_fire_prob: number;
  high_risk_percent: number;
  mean_uncertainty: number;
  fire_pixel_count: number;
}

export interface Environment {
  elevation_range: [number, number];
  wind_speed_mean: number;
  temp_min: number;
  temp_max: number;
  humidity: number;
  precipitation: number;
  ndvi_mean: number;
  erc_mean: number;
  has_prev_fire: boolean;
}

export interface Assessment {
  sample_id: number;
  fire_prob: number[][];
  uncertainty: number[][];
  valid_mask?: boolean[][];
  input_channels?: Record<string, number[][]>;
  risk_level: RiskLevel;
  confidence: number;
  environment: Environment;
  stats: RiskStats;
}

export interface BatchResult {
  sample_id: number;
  risk_level: RiskLevel;
  confidence: number;
  stats: RiskStats;
  environment: Environment;
  lat: number;
  lng: number;
}

export interface BatchSummary {
  assessed: number;
  risk_distribution: Record<RiskLevel, number>;
  avg_fire_prob: number;
  top_risk: { sample_id: number; risk_level: RiskLevel; max_fire_prob: number; confidence: number }[];
}

export interface BatchResponse {
  results: BatchResult[];
  summary: BatchSummary;
}

export interface ModelInfo {
  name: string;
  parameters: number;
  architecture: {
    encoder_widths: number[];
    bottleneck_channels: number;
    num_classes: number;
    input_channels: number;
    image_size: number;
  };
  training: { epoch: number; best_f1: number | null };
  feature_keys: string[];
  physics: { rothermel: boolean; ca_postprocess: boolean; evidential_fusion: boolean };
}

export interface LiveAssessment {
  location: { lat: number; lng: number };
  assessment_date: string;
  data_freshness: { weather: string; fire_detections: string };
  bbox: { west: number; south: number; east: number; north: number };
  fire_prob: number[][];
  uncertainty: number[][];
  input_channels?: Record<string, number[][]>;
  risk_level: RiskLevel;
  confidence: number;
  environment: Environment;
  stats: RiskStats;
}

export type RiskLevel = "CRITICAL" | "HIGH" | "MODERATE" | "LOW";
