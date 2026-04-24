-- =============================================================
-- Wind Turbine Predictive Maintenance — TimescaleDB Schema
-- =============================================================
-- Aligned with BitirmeAraRapor.docx §C.5
--
-- Tables:
--   sensors           — static sensor metadata catalog
--   feature_vectors   — 426-dim feature vectors (DOUBLE PRECISION[])
--   anomaly_results   — LSTM Autoencoder reconstruction error output
--   anomaly_logs      — auto-generated alarm log via trigger
--
-- Hypertables on: feature_vectors, anomaly_results, anomaly_logs
-- =============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================
-- 1. SENSORS — Static metadata catalog (report §C.5.2, Tablo 25)
-- =============================================================
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id        SERIAL PRIMARY KEY,
    sensor_type      TEXT NOT NULL CHECK (sensor_type IN (
                         'accelerometer', 'temperature', 'tachometer',
                         'anemometer', 'wind_vane')),
    channel_name     TEXT NOT NULL UNIQUE,
    location         TEXT,
    axis             TEXT,
    sampling_freq_hz DOUBLE PRECISION NOT NULL CHECK (sampling_freq_hz > 0),
    window_group     TEXT NOT NULL CHECK (window_group IN (
                         'bearing', 'nacelle', 'tower', 'slow')),
    window_seconds   DOUBLE PRECISION NOT NULL CHECK (window_seconds IN (1.0, 5.0)),
    feature_count    INTEGER NOT NULL CHECK (feature_count IN (15, 16))
);

-- =============================================================
-- 2. FEATURE_VECTORS — Main time-series table (report §C.5.1.3)
--    features = DOUBLE PRECISION[426]  (NOT JSONB, NOT 426 columns)
-- =============================================================
CREATE TABLE IF NOT EXISTS feature_vectors (
    time            TIMESTAMPTZ         NOT NULL,
    turbine_id      TEXT                NOT NULL,
    scenario_label  TEXT                DEFAULT 'unknown',
    features        DOUBLE PRECISION[]  NOT NULL
);

SELECT create_hypertable('feature_vectors', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_fv_turbine_time
    ON feature_vectors (turbine_id, time DESC);

-- =============================================================
-- 3. ANOMALY_RESULTS — LSTM Autoencoder output (report §C.5.1.1)
-- =============================================================
CREATE TABLE IF NOT EXISTS anomaly_results (
    time                 TIMESTAMPTZ     NOT NULL,
    turbine_id           TEXT            NOT NULL,
    reconstruction_error DOUBLE PRECISION NOT NULL CHECK (reconstruction_error >= 0),
    threshold            DOUBLE PRECISION NOT NULL CHECK (threshold > 0),
    is_anomaly           BOOLEAN         NOT NULL,
    model_version        TEXT            DEFAULT 'v1'
);

SELECT create_hypertable('anomaly_results', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ar_turbine_time
    ON anomaly_results (turbine_id, time DESC);

-- =============================================================
-- 4. ANOMALY_LOGS — Auto-generated from trigger (report §C.5.3.2)
--    Severity: LOW (score/threshold 1.0–1.5),
--              MEDIUM (1.5–2.0), HIGH (≥2.0)
-- =============================================================
CREATE TABLE IF NOT EXISTS anomaly_logs (
    log_id               BIGSERIAL,
    time                 TIMESTAMPTZ     NOT NULL,
    turbine_id           TEXT            NOT NULL,
    reconstruction_error DOUBLE PRECISION NOT NULL,
    threshold            DOUBLE PRECISION NOT NULL,
    severity             TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    acknowledged         BOOLEAN         DEFAULT FALSE,
    PRIMARY KEY (log_id, time)
);

SELECT create_hypertable('anomaly_logs', 'time', if_not_exists => TRUE);

-- =============================================================
-- 5. TRIGGER: Auto-log anomalies (report Tablo 26, T-1)
--    trg_auto_log_anomaly
-- =============================================================
CREATE OR REPLACE FUNCTION fn_auto_log_anomaly()
RETURNS TRIGGER AS $$
DECLARE
    sev   TEXT;
    ratio DOUBLE PRECISION;
BEGIN
    IF NEW.is_anomaly = TRUE THEN
        ratio := NEW.reconstruction_error / NULLIF(NEW.threshold, 0);
        IF ratio >= 2.0 THEN
            sev := 'HIGH';
        ELSIF ratio >= 1.5 THEN
            sev := 'MEDIUM';
        ELSE
            sev := 'LOW';
        END IF;

        INSERT INTO anomaly_logs (
            time, turbine_id, reconstruction_error, threshold, severity
        ) VALUES (
            NEW.time, NEW.turbine_id, NEW.reconstruction_error,
            NEW.threshold, sev
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_auto_log_anomaly ON anomaly_results;
CREATE TRIGGER trg_auto_log_anomaly
    AFTER INSERT ON anomaly_results
    FOR EACH ROW
    EXECUTE FUNCTION fn_auto_log_anomaly();

-- =============================================================
-- 6. STORED PROCEDURES (report Tablo 26, SP-1 through SP-4)
-- =============================================================

-- SP-1: Get latest N feature vectors for LSTM model input (20 × 426)
CREATE OR REPLACE FUNCTION sp_get_latest_features(
    p_turbine_id TEXT,
    p_limit      INTEGER DEFAULT 20
)
RETURNS TABLE (
    time     TIMESTAMPTZ,
    features DOUBLE PRECISION[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT fv.time, fv.features
    FROM   feature_vectors fv
    WHERE  fv.turbine_id = p_turbine_id
    ORDER  BY fv.time DESC
    LIMIT  p_limit;
END;
$$ LANGUAGE plpgsql;

-- SP-2: Get feature vectors by time range and optional scenario
CREATE OR REPLACE FUNCTION sp_get_features_by_range(
    p_turbine_id TEXT,
    p_start      TIMESTAMPTZ,
    p_end        TIMESTAMPTZ,
    p_scenario   TEXT DEFAULT NULL
)
RETURNS TABLE (
    time           TIMESTAMPTZ,
    scenario_label TEXT,
    features       DOUBLE PRECISION[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT fv.time, fv.scenario_label, fv.features
    FROM   feature_vectors fv
    WHERE  fv.turbine_id = p_turbine_id
      AND  fv.time BETWEEN p_start AND p_end
      AND  (p_scenario IS NULL OR fv.scenario_label = p_scenario)
    ORDER  BY fv.time ASC;
END;
$$ LANGUAGE plpgsql;

-- SP-3: Anomaly summary statistics by severity
CREATE OR REPLACE FUNCTION sp_anomaly_summary(
    p_turbine_id TEXT,
    p_start      TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    p_end        TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    severity   TEXT,
    total_count BIGINT,
    avg_score  DOUBLE PRECISION,
    max_score  DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT al.severity,
           COUNT(*)::BIGINT,
           AVG(al.reconstruction_error),
           MAX(al.reconstruction_error)
    FROM   anomaly_logs al
    WHERE  al.turbine_id = p_turbine_id
      AND  al.time BETWEEN p_start AND p_end
    GROUP  BY al.severity
    ORDER  BY al.severity;
END;
$$ LANGUAGE plpgsql;

-- SP-4: Model performance statistics
CREATE OR REPLACE FUNCTION sp_model_stats(
    p_turbine_id TEXT,
    p_start      TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    p_end        TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    total_inferences BIGINT,
    anomaly_count    BIGINT,
    normal_count     BIGINT,
    avg_error        DOUBLE PRECISION,
    max_error        DOUBLE PRECISION,
    anomaly_rate     DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT COUNT(*)::BIGINT                             AS total_inferences,
           COUNT(*) FILTER (WHERE ar.is_anomaly)::BIGINT AS anomaly_count,
           COUNT(*) FILTER (WHERE NOT ar.is_anomaly)::BIGINT AS normal_count,
           AVG(ar.reconstruction_error)                 AS avg_error,
           MAX(ar.reconstruction_error)                 AS max_error,
           CASE WHEN COUNT(*) > 0
                THEN COUNT(*) FILTER (WHERE ar.is_anomaly)::DOUBLE PRECISION
                     / COUNT(*)
                ELSE 0.0
           END                                          AS anomaly_rate
    FROM   anomaly_results ar
    WHERE  ar.turbine_id = p_turbine_id
      AND  ar.time BETWEEN p_start AND p_end;
END;
$$ LANGUAGE plpgsql;

-- =============================================================
-- 7. COMPRESSION POLICIES (report Tablo 26, SP-5)
--    Enable after initial data accumulation.
-- =============================================================
-- ALTER TABLE feature_vectors SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'turbine_id'
-- );
-- SELECT add_compression_policy('feature_vectors', INTERVAL '7 days');
--
-- ALTER TABLE anomaly_results SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'turbine_id'
-- );
-- SELECT add_compression_policy('anomaly_results', INTERVAL '14 days');

-- =============================================================
-- 8. SEED: Insert sensor metadata (28 channels)
-- =============================================================
INSERT INTO sensors (sensor_type, channel_name, location, axis, sampling_freq_hz, window_group, window_seconds, feature_count)
VALUES
    -- Bearing: 6 channels @ 74 kHz, 1s window, 16 features/ch
    ('accelerometer', 'brng_f_x', 'front_bearing', 'x', 74000, 'bearing', 1.0, 16),
    ('accelerometer', 'brng_f_y', 'front_bearing', 'y', 74000, 'bearing', 1.0, 16),
    ('accelerometer', 'brng_f_z', 'front_bearing', 'z', 74000, 'bearing', 1.0, 16),
    ('accelerometer', 'brng_r_x', 'rear_bearing',  'x', 74000, 'bearing', 1.0, 16),
    ('accelerometer', 'brng_r_y', 'rear_bearing',  'y', 74000, 'bearing', 1.0, 16),
    ('accelerometer', 'brng_r_z', 'rear_bearing',  'z', 74000, 'bearing', 1.0, 16),
    -- Nacelle: 3 channels @ 37 kHz, 1s window, 15 features/ch
    ('accelerometer', 'Nacl_x', 'nacelle', 'x', 37000, 'nacelle', 1.0, 15),
    ('accelerometer', 'Nacl_y', 'nacelle', 'y', 37000, 'nacelle', 1.0, 15),
    ('accelerometer', 'Nacl_z', 'nacelle', 'z', 37000, 'nacelle', 1.0, 15),
    -- Tower/Tach: 13 channels @ 2960 Hz, 5s window, 15 features/ch
    ('tachometer',    'tach',    'rotor',        NULL, 2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_f_x', 'tower_bottom', 'x',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_f_y', 'tower_bottom', 'y',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_f_z', 'tower_bottom', 'z',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_r_x', 'tower_bottom', 'x',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_r_y', 'tower_bottom', 'y',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'bot_r_z', 'tower_bottom', 'z',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_l_x', 'tower_top',    'x',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_l_y', 'tower_top',    'y',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_l_z', 'tower_top',    'z',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_r_x', 'tower_top',    'x',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_r_y', 'tower_top',    'y',  2960, 'tower', 5.0, 15),
    ('accelerometer', 'top_r_z', 'tower_top',    'z',  2960, 'tower', 5.0, 15),
    -- Slow: 6 channels @ 1480 Hz, 5s window, 15 features/ch
    ('temperature',   'tmp_amb',    'nacelle_ambient', NULL, 1480, 'slow', 5.0, 15),
    ('temperature',   'tmp_brng_f', 'front_bearing',   NULL, 1480, 'slow', 5.0, 15),
    ('temperature',   'tmp_brng_r', 'rear_bearing',    NULL, 1480, 'slow', 5.0, 15),
    ('anemometer',    'anm_mst',    'mast_9m',         NULL, 1480, 'slow', 5.0, 15),
    ('anemometer',    'anm_roof',   'roof_12.5m',      NULL, 1480, 'slow', 5.0, 15),
    ('wind_vane',     'van',        'nacelle_top',      NULL, 1480, 'slow', 5.0, 15)
ON CONFLICT (channel_name) DO NOTHING;
