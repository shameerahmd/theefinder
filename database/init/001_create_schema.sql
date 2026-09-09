CREATE SCHEMA IF NOT EXISTS extensions;

CREATE EXTENSION IF NOT EXISTS postgis
WITH SCHEMA extensions;

SET search_path TO public, extensions;

CREATE TABLE IF NOT EXISTS thermal_detections (
    id BIGSERIAL PRIMARY KEY,

    source VARCHAR(64) NOT NULL,
    satellite VARCHAR(32),
    acquisition_utc TIMESTAMPTZ NOT NULL,

    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom extensions.geometry(Point, 4326) NOT NULL,

    frp DOUBLE PRECISION,
    confidence VARCHAR(16),
    daynight VARCHAR(4),

    stage_a_prediction VARCHAR(64),
    stage_a_confidence DOUBLE PRECISION,
    stage_a_probabilities JSONB,

    stage_b_prediction VARCHAR(64),
    stage_b_confidence DOUBLE PRECISION,
    stage_b_probabilities JSONB,

    final_classification VARCHAR(64) NOT NULL,

    distance_to_industry_m DOUBLE PRECISION,
    industrial_feature_count_5km DOUBLE PRECISION,
    industrial_proximity_score DOUBLE PRECISION,

    tree_cover_pct DOUBLE PRECISION,
    cropland_pct DOUBLE PRECISION,
    built_up_pct DOUBLE PRECISION,

    detections_30d DOUBLE PRECISION,
    active_days_30d DOUBLE PRECISION,
    persistence_score DOUBLE PRECISION,

    industrial_context_ok BOOLEAN,
    landcover_ok BOOLEAN,
    persistence_status VARCHAR(32),
    persistence_ok BOOLEAN,

    explanation JSONB,
    data_quality JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_thermal_detection
        UNIQUE (
            source,
            satellite,
            acquisition_utc,
            latitude,
            longitude
        )
);

CREATE INDEX IF NOT EXISTS idx_thermal_detections_geom
    ON thermal_detections
    USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_thermal_detections_acquisition
    ON thermal_detections (acquisition_utc DESC);

CREATE INDEX IF NOT EXISTS idx_thermal_detections_classification
    ON thermal_detections (final_classification);

CREATE INDEX IF NOT EXISTS idx_thermal_detections_stage_a
    ON thermal_detections (stage_a_prediction);

CREATE INDEX IF NOT EXISTS idx_thermal_detections_stage_b
    ON thermal_detections (stage_b_prediction);
