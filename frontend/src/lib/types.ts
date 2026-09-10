export interface StageProbabilities {
  [key: string]: number;
}

export interface KeyFeatures {
  frp: number | null;

  distance_to_industry_m: number | null;
  industrial_feature_count_5km: number | null;
  industrial_proximity_score: number | null;

  tree_cover_pct: number | null;
  cropland_pct: number | null;
  built_up_pct: number | null;

  detections_30d: number | null;
  active_days_30d: number | null;
  persistence_score: number | null;

  nearest_major_road_distance_m?: number | null;
  nearest_major_road_class?: string | null;
  nearest_major_road_name?: string | null;
  nearest_major_road_ref?: string | null;
  major_road_count_1km?: number | null;
  road_context_source?: string | null;

  nearest_industrial_type: string | null;
  nearest_industrial_name: string | null;
  nearest_industrial_facility_distance_m: number | null;

  nearest_specialized_industrial_type: string | null;
  nearest_specialized_industrial_name: string | null;
  nearest_specialized_industrial_distance_m: number | null;

  refinery_count: number | null;
  power_plant_count: number | null;
  steel_metal_plant_count: number | null;
  factory_count: number | null;
  mine_quarry_count: number | null;
  flare_count: number | null;
}

export interface DataQuality {
  industrial_context_ok: boolean;

  landcover_ok: boolean;

  persistence_status: "NOT_REQUIRED" | "REQUESTED" | "OK" | "FAILED" | string;

  persistence_ok: boolean | null;

  road_context_status?: string;
  road_context_ok?: boolean | null;

  industrial_context_error: string | null;
  landcover_error: string | null;
  persistence_error: string | null;
  road_context_error?: string | null;
}

export interface Detection {
  latitude: number;
  longitude: number;
  frp: number;

  confidence: string | null;
  daynight: string | null;
  satellite: string | null;
  source?: string | null;

  acquisition_utc: string;

  stage_a_prediction: string;
  stage_a_confidence: number;
  stage_a_probabilities: StageProbabilities;

  stage_b_prediction: string | null;
  stage_b_confidence: number | null;
  stage_b_probabilities: StageProbabilities | null;

  final_classification: string;

  explanation: string[] | string;
  explanations?: string[];

  key_features: KeyFeatures;

  data_quality: DataQuality;
}

export interface ChennaiClassificationResponse {
  application: string;

  study_area: string;

  bounding_box: string;

  source: string;

  days: number;

  firms_detection_count: number;

  processed_detection_count?: number;

  classified_count: number;

  failed_count: number;

  classification_summary: Record<string, number>;

  detections: Detection[];

  failures?: Array<{
    row_index?: number;
    latitude?: number;
    longitude?: number;
    error: string;
  }>;
}
