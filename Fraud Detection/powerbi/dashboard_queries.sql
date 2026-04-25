-- ============================================================
-- Power BI AML Compliance Dashboard — DirectQuery SQL
-- ============================================================
-- Database: PostgreSQL (GCP Cloud SQL)
-- Schema:   fraud_analytics
-- Refresh:  Every 15 minutes via Power BI scheduled refresh
--
-- UAE CBUAE AML Compliance Dashboard panels:
--   1. Fraud rate by hour/day
--   2. Top merchant categories by fraud rate
--   3. SHAP reason distribution
--   4. Risk tier breakdown
--   5. Model drift KPI cards
--   6. AML alert queue
--   7. Daily fraud amount exposure
--   8. Geographic risk heatmap (if location data available)
-- ============================================================

-- ============================================================
-- TABLE SCHEMA (for reference)
-- ============================================================
/*
CREATE TABLE fraud_analytics.predictions (
    id                  BIGSERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) NOT NULL UNIQUE,
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fraud_score         NUMERIC(6,4) NOT NULL,
    is_fraud            BOOLEAN NOT NULL,
    risk_tier           VARCHAR(16) NOT NULL,   -- LOW/MEDIUM/HIGH/CRITICAL
    confidence          NUMERIC(6,4),
    recommended_action  VARCHAR(64),
    compliance_ref      VARCHAR(128),
    model_version       VARCHAR(64),
    -- Input features (key ones stored for drift monitoring)
    amount              NUMERIC(14,2),
    merchant_category   VARCHAR(64),
    channel             VARCHAR(32),
    currency            VARCHAR(8) DEFAULT 'USD',
    is_international    BOOLEAN,
    -- SHAP top reasons (stored as JSONB)
    shap_reasons        JSONB,
    -- Audit
    latency_ms          NUMERIC(8,2),
    api_key_hash        VARCHAR(64),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_predicted_at ON fraud_analytics.predictions(predicted_at DESC);
CREATE INDEX idx_predictions_risk_tier    ON fraud_analytics.predictions(risk_tier);
CREATE INDEX idx_predictions_is_fraud     ON fraud_analytics.predictions(is_fraud) WHERE is_fraud = TRUE;
CREATE INDEX idx_predictions_merchant     ON fraud_analytics.predictions(merchant_category);

CREATE TABLE fraud_analytics.model_drift_log (
    id              BIGSERIAL PRIMARY KEY,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(64),
    psi_score       NUMERIC(8,4),
    drift_detected  BOOLEAN,
    status          VARCHAR(16),   -- healthy/warning/critical
    feature_psi     JSONB,
    recommendation  TEXT
);
*/


-- ============================================================
-- QUERY 1: Fraud Rate Time Series (Power BI Line Chart)
-- Panel: "Fraud Rate Trend" — refreshes every 15 min
-- ============================================================
SELECT
    DATE_TRUNC('hour', predicted_at AT TIME ZONE 'Asia/Dubai')     AS hour_bucket_ast,
    EXTRACT(HOUR FROM predicted_at AT TIME ZONE 'Asia/Dubai')      AS hour_of_day,
    TO_CHAR(predicted_at AT TIME ZONE 'Asia/Dubai', 'YYYY-MM-DD HH24:00') AS label,
    COUNT(*)                                                        AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                      AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*),
        4
    )                                                               AS fraud_rate_pct,
    ROUND(AVG(fraud_score)::NUMERIC, 4)                            AS avg_fraud_score,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fraud_score)::NUMERIC, 4)
                                                                    AS p95_fraud_score
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '48 hours'
GROUP BY
    1, 2, 3
ORDER BY
    1 ASC;


-- ============================================================
-- QUERY 2: Merchant Category Risk Table (Power BI Bar/Table)
-- Panel: "Top Flagged Merchants" — refreshes every 30 min
-- ============================================================
SELECT
    COALESCE(merchant_category, 'unknown')                          AS merchant_category,
    COUNT(*)                                                        AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                      AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*),
        4
    )                                                               AS fraud_rate_pct,
    ROUND(AVG(fraud_score)::NUMERIC, 4)                            AS avg_fraud_score,
    ROUND(SUM(CASE WHEN is_fraud THEN COALESCE(amount, 0) ELSE 0 END)::NUMERIC, 2)
                                                                    AS fraud_exposure_usd,
    -- Risk tier (for conditional formatting in Power BI)
    CASE
        WHEN (SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0)) >= 0.025
            THEN 'CRITICAL'
        WHEN (SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0)) >= 0.010
            THEN 'HIGH'
        WHEN (SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0)) >= 0.005
            THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                             AS risk_tier
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '7 days'
    AND merchant_category IS NOT NULL
GROUP BY
    1
ORDER BY
    fraud_rate_pct DESC
LIMIT 20;


-- ============================================================
-- QUERY 3: SHAP Reason Distribution (Power BI Treemap)
-- Panel: "Model Explainability — Top Risk Drivers"
-- ============================================================
WITH shap_exploded AS (
    SELECT
        transaction_id,
        is_fraud,
        jsonb_array_elements(shap_reasons)                          AS reason
    FROM
        fraud_analytics.predictions
    WHERE
        predicted_at >= NOW() - INTERVAL '24 hours'
        AND shap_reasons IS NOT NULL
        AND is_fraud = TRUE   -- Only explain fraud flags for compliance
),
shap_parsed AS (
    SELECT
        reason->>'feature'                                          AS feature_name,
        (reason->>'impact')::FLOAT                                  AS impact,
        reason->>'direction'                                        AS direction,
        transaction_id
    FROM shap_exploded
)
SELECT
    feature_name,
    COUNT(DISTINCT transaction_id)                                  AS flagged_transaction_count,
    ROUND(AVG(impact)::NUMERIC, 4)                                 AS avg_shap_impact,
    ROUND(
        100.0 * SUM(CASE WHEN direction = 'increases_risk' THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                                               AS pct_increases_risk,
    -- Rank for Power BI treemap sizing
    RANK() OVER (ORDER BY COUNT(DISTINCT transaction_id) DESC)      AS importance_rank
FROM
    shap_parsed
WHERE
    feature_name IS NOT NULL
GROUP BY
    feature_name
ORDER BY
    flagged_transaction_count DESC
LIMIT 15;


-- ============================================================
-- QUERY 4: Risk Tier Summary KPI Cards (Power BI Card Visuals)
-- Panel: "Risk Distribution" — donut + KPI cards
-- ============================================================
SELECT
    risk_tier,
    COUNT(*)                                                        AS transaction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)             AS pct_of_total,
    ROUND(AVG(fraud_score)::NUMERIC, 4)                            AS avg_fraud_score,
    ROUND(SUM(COALESCE(amount, 0))::NUMERIC, 2)                   AS total_amount_usd,
    -- Power BI colour coding
    CASE risk_tier
        WHEN 'LOW'      THEN '#27AE60'
        WHEN 'MEDIUM'   THEN '#F39C12'
        WHEN 'HIGH'     THEN '#E67E22'
        WHEN 'CRITICAL' THEN '#C0392B'
    END                                                             AS colour_hex
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '24 hours'
GROUP BY
    risk_tier
ORDER BY
    CASE risk_tier
        WHEN 'LOW' THEN 1 WHEN 'MEDIUM' THEN 2
        WHEN 'HIGH' THEN 3 WHEN 'CRITICAL' THEN 4
    END;


-- ============================================================
-- QUERY 5: Model Drift KPI Card (Power BI Gauge)
-- Panel: "Model Health Monitor"
-- ============================================================
SELECT
    logged_at                                                       AS last_check,
    model_version,
    ROUND(psi_score::NUMERIC, 4)                                   AS psi_score,
    drift_detected,
    status,
    recommendation,
    -- Gauge value (0–100 scale for Power BI gauge)
    LEAST(100, ROUND((psi_score / 0.25) * 100))                   AS psi_gauge_pct,
    -- Traffic light status for conditional formatting
    CASE status
        WHEN 'healthy'  THEN '#27AE60'
        WHEN 'warning'  THEN '#F39C12'
        WHEN 'critical' THEN '#C0392B'
        ELSE '#95A5A6'
    END                                                             AS status_colour
FROM
    fraud_analytics.model_drift_log
ORDER BY
    logged_at DESC
LIMIT 1;


-- ============================================================
-- QUERY 6: AML Alert Queue (Power BI Table with drill-through)
-- Panel: "Compliance Officer Alert Queue"
-- Required by: CBUAE AML Circular 2/2024 Article 8
-- ============================================================
SELECT
    transaction_id,
    predicted_at AT TIME ZONE 'Asia/Dubai'                         AS flagged_at_ast,
    TO_CHAR(predicted_at AT TIME ZONE 'Asia/Dubai', 'YYYY-MM-DD HH24:MI:SS')
                                                                    AS flagged_at_str,
    ROUND(fraud_score::NUMERIC, 4)                                 AS fraud_score,
    risk_tier,
    ROUND(COALESCE(amount, 0)::NUMERIC, 2)                        AS amount_usd,
    COALESCE(merchant_category, 'unknown')                         AS merchant_category,
    COALESCE(currency, 'USD')                                      AS currency,
    COALESCE(channel, 'unknown')                                   AS channel,
    is_international,
    compliance_ref,
    recommended_action,
    model_version,
    -- SHAP top reason (first element)
    shap_reasons->0->>'feature'                                    AS top_shap_feature,
    shap_reasons->0->>'direction'                                  AS top_shap_direction,
    ROUND((shap_reasons->0->>'impact')::FLOAT::NUMERIC, 4)        AS top_shap_impact,
    -- Age of alert in minutes
    ROUND(EXTRACT(EPOCH FROM (NOW() - predicted_at)) / 60)        AS alert_age_minutes,
    -- SLA breach flag (CBUAE: CRITICAL alerts must be reviewed within 120 min)
    CASE
        WHEN risk_tier = 'CRITICAL'
             AND EXTRACT(EPOCH FROM (NOW() - predicted_at)) > 7200
             THEN TRUE
        ELSE FALSE
    END                                                             AS sla_breached
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '48 hours'
    AND risk_tier IN ('HIGH', 'CRITICAL')
ORDER BY
    fraud_score DESC,
    predicted_at DESC
LIMIT 500;


-- ============================================================
-- QUERY 7: Daily Fraud Exposure Summary (Power BI Bar Chart)
-- Panel: "Daily Fraud Amount Exposure (USD)"
-- ============================================================
SELECT
    DATE_TRUNC('day', predicted_at AT TIME ZONE 'Asia/Dubai')::DATE   AS date_ast,
    TO_CHAR(predicted_at AT TIME ZONE 'Asia/Dubai', 'Mon DD')          AS date_label,
    COUNT(*)                                                            AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                          AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud THEN COALESCE(amount, 0) ELSE 0 END)::NUMERIC, 2)
                                                                        AS fraud_exposure_usd,
    ROUND(AVG(CASE WHEN is_fraud THEN COALESCE(amount, 0) END)::NUMERIC, 2)
                                                                        AS avg_fraud_amount,
    ROUND(MAX(CASE WHEN is_fraud THEN COALESCE(amount, 0) ELSE 0 END)::NUMERIC, 2)
                                                                        AS max_fraud_amount,
    ROUND(AVG(fraud_score)::NUMERIC, 4)                                AS avg_fraud_score
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '30 days'
GROUP BY
    1, 2
ORDER BY
    1 ASC;


-- ============================================================
-- QUERY 8: Hourly Heatmap Data (Power BI Matrix)
-- Panel: "Fraud Heatmap — Day × Hour"
-- ============================================================
SELECT
    TO_CHAR(predicted_at AT TIME ZONE 'Asia/Dubai', 'Dy')             AS day_of_week,
    EXTRACT(DOW FROM predicted_at AT TIME ZONE 'Asia/Dubai')::INT      AS day_num,
    EXTRACT(HOUR FROM predicted_at AT TIME ZONE 'Asia/Dubai')::INT     AS hour_of_day,
    COUNT(*)                                                            AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                          AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        3
    )                                                                   AS fraud_rate_pct
FROM
    fraud_analytics.predictions
WHERE
    predicted_at >= NOW() - INTERVAL '30 days'
GROUP BY
    1, 2, 3
ORDER BY
    2 ASC, 3 ASC;


-- ============================================================
-- QUERY 9: Model Performance Rolling (Power BI Line Chart)
-- Panel: "Model Accuracy Over Time" — requires labelled outcomes
-- ============================================================
-- Note: Requires joining predictions with confirmed fraud outcomes
-- from core banking system (webhook or batch job)
WITH labelled AS (
    SELECT
        p.transaction_id,
        p.predicted_at,
        p.fraud_score,
        p.is_fraud                                                  AS predicted_fraud,
        COALESCE(o.confirmed_fraud, p.is_fraud)                    AS actual_fraud
    FROM
        fraud_analytics.predictions p
        LEFT JOIN fraud_analytics.fraud_outcomes o
            ON p.transaction_id = o.transaction_id
    WHERE
        p.predicted_at >= NOW() - INTERVAL '30 days'
),
daily_metrics AS (
    SELECT
        DATE_TRUNC('day', predicted_at AT TIME ZONE 'Asia/Dubai')::DATE AS date_ast,
        COUNT(*)                                                    AS n_predictions,
        -- True positives, false positives, etc.
        SUM(CASE WHEN predicted_fraud AND actual_fraud THEN 1 ELSE 0 END)  AS tp,
        SUM(CASE WHEN predicted_fraud AND NOT actual_fraud THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN NOT predicted_fraud AND actual_fraud THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN NOT predicted_fraud AND NOT actual_fraud THEN 1 ELSE 0 END) AS tn
    FROM labelled
    GROUP BY 1
)
SELECT
    date_ast,
    n_predictions,
    tp, fp, fn, tn,
    -- Precision = TP / (TP + FP)
    ROUND(tp::NUMERIC / NULLIF(tp + fp, 0), 4)                    AS precision_score,
    -- Recall = TP / (TP + FN)
    ROUND(tp::NUMERIC / NULLIF(tp + fn, 0), 4)                    AS recall_score,
    -- F1 = 2 × (Precision × Recall) / (Precision + Recall)
    ROUND(
        2.0 * (tp::FLOAT / NULLIF(tp + fp, 0)) * (tp::FLOAT / NULLIF(tp + fn, 0))
        / NULLIF(
            (tp::FLOAT / NULLIF(tp + fp, 0)) + (tp::FLOAT / NULLIF(tp + fn, 0)),
            0
        )::NUMERIC, 4
    )                                                               AS f1_score
FROM
    daily_metrics
ORDER BY
    date_ast ASC;


-- ============================================================
-- CBUAE REGULATORY REPORTING VIEW
-- ============================================================
-- Materialised view for monthly CBUAE AML report submission
-- ============================================================
CREATE OR REPLACE VIEW fraud_analytics.cbuae_monthly_report AS
SELECT
    DATE_TRUNC('month', predicted_at)::DATE                         AS report_month,
    COUNT(*)                                                        AS total_transactions_screened,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                      AS total_fraud_flags,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4
    )                                                               AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud THEN COALESCE(amount, 0) ELSE 0 END)::NUMERIC, 2)
                                                                    AS total_fraud_exposure_usd,
    COUNT(CASE WHEN risk_tier = 'CRITICAL' THEN 1 END)             AS critical_alerts,
    COUNT(CASE WHEN risk_tier = 'HIGH' THEN 1 END)                 AS high_alerts,
    COUNT(CASE WHEN recommended_action = 'block_and_report_cbuae' THEN 1 END)
                                                                    AS transactions_reported_to_cbuae,
    MAX(model_version)                                              AS model_version_used,
    'UAE VARA / CBUAE AML Circular 2/2024'                         AS regulatory_framework
FROM
    fraud_analytics.predictions
GROUP BY
    1
ORDER BY
    1 DESC;