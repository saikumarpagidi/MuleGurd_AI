-- ============================================================
-- MuleGuard.AI — PostgreSQL Schema
--
-- Design rule: scoring_decisions and audit_log are APPEND-ONLY.
-- No UPDATE. No DELETE. Enforced at the database engine level
-- via PostgreSQL RULEs — independent of application-layer checks.
-- ============================================================


-- ── TABLE 1: SCORING DECISIONS ────────────────────────────
-- Immutable record of every model score and decision produced.
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scoring_decisions (
    id                    BIGSERIAL PRIMARY KEY,
    transaction_id        VARCHAR(128)   NOT NULL UNIQUE,
    source_token          CHAR(64)       NOT NULL,
    destination_token     CHAR(64)       NOT NULL,
    amount                NUMERIC(18,2)  NOT NULL CHECK (amount > 0),
    channel               VARCHAR(32)    NOT NULL,
    final_risk_score      NUMERIC(6,4)   NOT NULL CHECK (final_risk_score BETWEEN 0 AND 1),
    decision              VARCHAR(10)    NOT NULL CHECK (decision IN ('APPROVE','REVIEW','BLOCK')),
    tgn_score             NUMERIC(6,4)   CHECK (tgn_score BETWEEN 0 AND 1),
    lgbm_score            NUMERIC(6,4)   CHECK (lgbm_score BETWEEN 0 AND 1),
    anomaly_score         NUMERIC(6,4)   CHECK (anomaly_score BETWEEN 0 AND 1),
    device_score          NUMERIC(6,4)   CHECK (device_score BETWEEN 0 AND 1),
    ring_flag             BOOLEAN        NOT NULL DEFAULT FALSE,
    ring_role             VARCHAR(10)    CHECK (ring_role IN ('HUB','BRIDGE','MULE','NONE')),
    shap_top_features     JSONB,
    explanation           TEXT,
    explanation_mode      VARCHAR(20)    NOT NULL DEFAULT 'rule_based'
                          CHECK (explanation_mode IN ('llm','rule_based','none')),
    degraded_mode         BOOLEAN        NOT NULL DEFAULT FALSE,
    processing_ms         INTEGER        CHECK (processing_ms > 0),
    scored_at             TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);


-- ── TABLE 2: ALERTS ───────────────────────────────────────
-- Created when a scoring decision results in REVIEW or BLOCK.
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  VARCHAR(128) NOT NULL REFERENCES scoring_decisions(transaction_id),
    alert_priority  VARCHAR(10)  NOT NULL CHECK (alert_priority IN ('HIGH','CRITICAL')),
    status          VARCHAR(20)  NOT NULL DEFAULT 'OPEN'
                    CHECK (status IN ('OPEN','ASSIGNED','RESOLVED','FALSE_POSITIVE')),
    assigned_to     VARCHAR(128),
    notes           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    resolved_by     VARCHAR(128)
);


-- ── TABLE 3: AUDIT LOG ────────────────────────────────────
-- Append-only, tamper-evident record of every system event.
-- PostgreSQL RULEs enforce immutability at the engine level.
-- SHA-256 checksum enables offline tamper detection.
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      VARCHAR(64)  NOT NULL,
    entity_type     VARCHAR(64)  NOT NULL,
    entity_id       VARCHAR(128) NOT NULL,
    actor           VARCHAR(128) NOT NULL DEFAULT 'system',
    payload         JSONB,
    row_checksum    CHAR(64),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);


-- ── TABLE 4: MODEL REGISTRY ───────────────────────────────
-- Tracks every trained model version that has been deployed or evaluated.
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id               BIGSERIAL PRIMARY KEY,
    model_name       VARCHAR(64)  NOT NULL,
    model_type       VARCHAR(32)  NOT NULL CHECK (model_type IN ('LightGBM','Autoencoder','TGN','GNN')),
    auc_score        NUMERIC(6,4),
    fpr              NUMERIC(6,4),
    fnr              NUMERIC(6,4),
    artifact_path    TEXT,
    training_dataset VARCHAR(64),
    feature_count    INTEGER,
    is_active        BOOLEAN      NOT NULL DEFAULT FALSE,
    deployed_at      TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_sd_source_token      ON scoring_decisions(source_token);
CREATE INDEX IF NOT EXISTS idx_sd_destination_token ON scoring_decisions(destination_token);
CREATE INDEX IF NOT EXISTS idx_sd_scored_at         ON scoring_decisions(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_sd_decision          ON scoring_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_sd_ring_flag         ON scoring_decisions(ring_flag) WHERE ring_flag = TRUE;

CREATE INDEX IF NOT EXISTS idx_alerts_status        ON alerts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_priority      ON alerts(alert_priority, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_entity         ON audit_log(entity_type, entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event_type     ON audit_log(event_type, created_at DESC);


-- ============================================================
-- IMMUTABILITY ENFORCEMENT
-- PostgreSQL RULEs intercept UPDATE and DELETE before execution.
-- These rules apply regardless of the calling role or application.
-- ============================================================

CREATE OR REPLACE RULE no_update_audit_log AS
    ON UPDATE TO audit_log DO INSTEAD NOTHING;

CREATE OR REPLACE RULE no_delete_audit_log AS
    ON DELETE TO audit_log DO INSTEAD NOTHING;

CREATE OR REPLACE RULE no_update_scoring_decisions AS
    ON UPDATE TO scoring_decisions DO INSTEAD NOTHING;

CREATE OR REPLACE RULE no_delete_scoring_decisions AS
    ON DELETE TO scoring_decisions DO INSTEAD NOTHING;


-- ============================================================
-- BOOTSTRAP — Initial audit entry to confirm INSERT works
-- ============================================================
INSERT INTO audit_log (event_type, entity_type, entity_id, actor, payload)
VALUES (
    'SCHEMA_INITIALIZED',
    'database',
    'muleguard',
    'system',
    '{"schema": "muleguard"}'
);
