-- Companion enrichment schema for Immich (same Postgres instance).
-- Designed to be additive and avoid Immich migrations: keep in separate "enrich" schema.
-- Key assets by Immich assetId (UUID) and/or content_sha256 for cross-system joins.
-- Apply manually if DB already initialized: psql -U "$DB_USERNAME" -d "$DB_DATABASE_NAME" -f /path/to/enrich_schema.sql

BEGIN;

CREATE SCHEMA IF NOT EXISTS enrich;

-- Canonical asset identity / linkage
CREATE TABLE IF NOT EXISTS enrich.asset (
    id                  BIGSERIAL PRIMARY KEY,
    immich_asset_id     UUID,                       -- Link to Immich asset
    content_sha256      CHAR(64),                   -- SHA256 of file content
    source_system       TEXT,                       -- e.g. immich, icloud, takeout, android, local
    source_ref          TEXT,                       -- Source-specific identifier
    original_path       TEXT,                       -- Original filesystem path
    filename            TEXT,
    mime_type           TEXT,
    file_size           BIGINT,
    imported_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_enrich_asset_immich_asset_id
    ON enrich.asset (immich_asset_id) WHERE immich_asset_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_enrich_asset_content_sha256
    ON enrich.asset (content_sha256) WHERE content_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_enrich_asset_source
    ON enrich.asset (source_system, source_ref);
CREATE INDEX IF NOT EXISTS ix_enrich_asset_original_path
    ON enrich.asset (original_path);

-- Full raw metadata dump (ExifTool -j -a -G1 -s, source sidecars, etc.)
CREATE TABLE IF NOT EXISTS enrich.asset_metadata_raw (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    extractor           TEXT NOT NULL,              -- exiftool, immich, sidecar, etc.
    raw_json            JSONB NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_asset_metadata_raw_asset_extracted
    ON enrich.asset_metadata_raw (asset_id, extracted_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_metadata_raw_extractor
    ON enrich.asset_metadata_raw (extractor);
CREATE INDEX IF NOT EXISTS idx_metadata_gin
    ON enrich.asset_metadata_raw USING GIN (raw_json);

-- Normalized/core fields for fast queries (derived from Immich + raw metadata)
CREATE TABLE IF NOT EXISTS enrich.asset_core (
    asset_id            BIGINT PRIMARY KEY REFERENCES enrich.asset(id) ON DELETE CASCADE,
    captured_at         TIMESTAMPTZ,
    camera_make         TEXT,
    camera_model        TEXT,
    lens_model          TEXT,
    focal_length        DOUBLE PRECISION,
    aperture            DOUBLE PRECISION,
    iso                 INTEGER,
    exposure_time       DOUBLE PRECISION,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    altitude            DOUBLE PRECISION,
    orientation         INTEGER,
    width               INTEGER,
    height              INTEGER,
    duration            DOUBLE PRECISION,           -- For video
    is_video            BOOLEAN DEFAULT FALSE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_asset_core_captured_at ON enrich.asset_core (captured_at);
CREATE INDEX IF NOT EXISTS ix_asset_core_device ON enrich.asset_core (camera_make, camera_model);
CREATE INDEX IF NOT EXISTS ix_asset_core_location ON enrich.asset_core (latitude, longitude);

-- Local VLM/AI enrichment results (captions, objects/scenes, OCR, embeddings refs)
CREATE TABLE IF NOT EXISTS enrich.asset_vlm (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    model_name          TEXT NOT NULL,
    model_runtime       TEXT,                       -- ollama, openai-api, etc.
    prompt_version      TEXT,
    enriched_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    caption_short       TEXT,
    caption_detailed    TEXT,
    objects             JSONB,                      -- [{label, confidence, bbox}]
    scenes              JSONB,                      -- [{label, confidence}]
    activities          JSONB,
    attributes          JSONB,
    ocr_text            TEXT,
    ocr_blocks          JSONB,
    embedding_model     TEXT,
    quality_score       DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS ix_asset_vlm_asset_enriched
    ON enrich.asset_vlm (asset_id, enriched_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_vlm_model
    ON enrich.asset_vlm (model_name, prompt_version);

-- Face enrichment results (prefer Immich faces where available; store supplemental/local results here)
CREATE TABLE IF NOT EXISTS enrich.asset_faces (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    provider            TEXT NOT NULL,            -- e.g. immich, insightface, local
    person_label        TEXT,                     -- local label/cluster id; avoid sensitive claims
    confidence          DOUBLE PRECISION,
    bbox                JSONB,                    -- {x,y,w,h} or polygon in image coords
    landmarks           JSONB,
    embedding_model     TEXT,
    embedding           JSONB,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_response        JSONB
);

CREATE INDEX IF NOT EXISTS ix_asset_faces_asset ON enrich.asset_faces (asset_id);
CREATE INDEX IF NOT EXISTS ix_asset_faces_person ON enrich.asset_faces (provider, person_label);

-- Sidecar/output tracking (XMP/JSON non-destructive writes)
CREATE TABLE IF NOT EXISTS enrich.asset_sidecar (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    sidecar_path        TEXT NOT NULL,
    status              TEXT NOT NULL,            -- written, skipped_unsafe, error
    fields_written      JSONB,                    -- which fields were written
    error_message       TEXT,
    written_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_asset_sidecar_asset ON enrich.asset_sidecar (asset_id, written_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_sidecar_status ON enrich.asset_sidecar (status);

-- Convenience view: latest VLM enrichment per asset
CREATE OR REPLACE VIEW enrich.asset_latest_vlm AS
SELECT DISTINCT ON (asset_id)
    id, asset_id, model_name, model_runtime, prompt_version, enriched_at,
    caption_detailed, caption_short, objects, scenes, activities, attributes,
    ocr_text, ocr_blocks, embedding_model, quality_score
FROM enrich.asset_vlm
ORDER BY asset_id, enriched_at DESC, id DESC;

-- Body pose detection results
CREATE TABLE IF NOT EXISTS enrich.asset_body_poses (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    landmark_set        TEXT NOT NULL,            -- mediapipe_pose, yolov8_pose, rtmpose, openpose
    landmarks_2d        JSONB NOT NULL,           -- [[x, y, visibility], ...] normalized 0-1
    landmarks_3d        JSONB,                    -- [[x, y, z], ...] world coordinates (meters)
    bbox                JSONB,                    -- [x1, y1, x2, y2] normalized
    det_score           DOUBLE PRECISION,
    model_name          TEXT,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_response        JSONB
);

CREATE INDEX IF NOT EXISTS ix_asset_body_poses_asset ON enrich.asset_body_poses (asset_id);
CREATE INDEX IF NOT EXISTS ix_asset_body_poses_model ON enrich.asset_body_poses (model_name);

-- Body measurements (per-asset and aggregated per-performer)
CREATE TABLE IF NOT EXISTS enrich.body_measurements (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT REFERENCES enrich.asset(id) ON DELETE CASCADE,
    performer_id        BIGINT REFERENCES enrich.performers(id) ON DELETE CASCADE,
    measurement_type    TEXT NOT NULL,            -- height, bust, waist, hips, shoulder_width, etc.
    value               DOUBLE PRECISION NOT NULL,
    unit                TEXT NOT NULL DEFAULT 'cm', -- cm, mm, inches, ratio, percent
    confidence          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    method              TEXT NOT NULL DEFAULT 'landmark_2d', -- landmark_2d, landmark_3d, vlm_estimate, manual
    reference_scale     DOUBLE PRECISION,         -- Known reference for absolute scaling
    landmarks_used      JSONB,                    -- Indices of landmarks used
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS ix_body_measurements_asset ON enrich.body_measurements (asset_id);
CREATE INDEX IF NOT EXISTS ix_body_measurements_performer ON enrich.body_measurements (performer_id);
CREATE INDEX IF NOT EXISTS ix_body_measurements_type ON enrich.body_measurements (measurement_type);

-- Facial biometrics (aggregated per performer)
CREATE TABLE IF NOT EXISTS enrich.facial_biometrics (
    id                          BIGSERIAL PRIMARY KEY,
    performer_id                BIGINT NOT NULL REFERENCES enrich.performers(id) ON DELETE CASCADE,
    identity_embedding          JSONB NOT NULL,           -- Mean ArcFace embedding (512-dim)
    identity_embedding_std      JSONB,                    -- Per-dimension std
    identity_quality_mean       DOUBLE PRECISION DEFAULT 0.0,
    shape_coefficients          JSONB,                    -- 3DMM shape params
    expression_coefficients_mean JSONB,                   -- Mean expression params
    expression_coefficients_std  JSONB,                   -- Std of expression params
    canonical_landmarks_68      JSONB,                    -- Mean 68 landmarks
    canonical_landmarks_68_cov  JSONB,                    -- Covariance matrix
    canonical_landmarks_468     JSONB,                    -- Mean 468 landmarks
    facial_measurements         JSONB DEFAULT '{}',       -- {type: {mean, std, min, max}}
    skin_tone_lab               JSONB,                    -- Mean L*a*b*
    skin_texture_descriptor     JSONB,                    -- Texture embedding
    source_image_count          INTEGER DEFAULT 0,
    pose_coverage               JSONB DEFAULT '{}',       -- yaw bins -> count
    expression_coverage         JSONB DEFAULT '{}',       -- expression labels -> count
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_versions              JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_facial_biometrics_performer ON enrich.facial_biometrics (performer_id);

-- Performers / known persons (extended from enrich_persons)
CREATE TABLE IF NOT EXISTS enrich.performers (
    id                      BIGSERIAL PRIMARY KEY,
    name                    TEXT NOT NULL,                -- Stage name / identifier
    aliases                 JSONB DEFAULT '[]',
    face_embedding          JSONB,                        -- Canonical face embedding (512-dim)
    face_embedding_quality  DOUBLE PRECISION DEFAULT 0.0,
    measurements            JSONB DEFAULT '{}',           -- Aggregated {type: {mean, std, min, max, median, count}}
    measurement_confidence  JSONB DEFAULT '{}',           -- type -> mean confidence
    body_type               TEXT,                         -- ectomorph/mesomorph/endomorph/etc
    height_cm               DOUBLE PRECISION,
    height_confidence       DOUBLE PRECISION DEFAULT 0.0,
    weight_kg_estimate      DOUBLE PRECISION,
    facial_biometrics_id    BIGINT REFERENCES enrich.facial_biometrics(id) ON DELETE SET NULL,
    asset_count             INTEGER DEFAULT 0,
    first_seen              TIMESTAMPTZ,
    last_seen               TIMESTAMPTZ,
    source_tags             JSONB DEFAULT '[]',           -- Studio, site, etc.
    notes                   TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_performers_name ON enrich.performers (name);
CREATE INDEX IF NOT EXISTS ix_performers_face_embedding ON enrich.performers USING GIN (face_embedding);

-- Performer-asset linkage (many-to-many)
CREATE TABLE IF NOT EXISTS enrich.performer_assets (
    performer_id    BIGINT NOT NULL REFERENCES enrich.performers(id) ON DELETE CASCADE,
    asset_id        BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    matched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    match_method    TEXT,                               -- face, metadata, manual
    match_score     DOUBLE PRECISION,
    PRIMARY KEY (performer_id, asset_id)
);

CREATE INDEX IF NOT EXISTS ix_performer_assets_asset ON enrich.performer_assets (asset_id);

COMMIT;