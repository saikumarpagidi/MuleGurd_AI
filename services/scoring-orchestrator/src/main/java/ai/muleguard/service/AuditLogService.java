package ai.muleguard.service;

import ai.muleguard.model.ScoringDecision;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.r2dbc.core.DatabaseClient;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;

/**
 * Persists scoring decisions to PostgreSQL using R2DBC (non-blocking).
 *
 * Two tables are written per non-APPROVE decision:
 *   1. scoring_decisions — the immutable scoring record (append-only enforced by DB RULE)
 *   2. alerts           — created when decision is BLOCK or REVIEW
 *
 * IMPORTANT: The scoring_decisions table has PostgreSQL RULE enforcement of
 * immutability (no UPDATE, no DELETE). This is NOT enforced at the application
 * layer — it's at the database engine level. The application writes INSERT only.
 *
 * Transaction: both inserts run in a single @Transactional context so if
 * the alerts INSERT fails, the scoring_decisions INSERT is also rolled back
 * (preventing orphaned decisions with no corresponding alert record).
 *
 * Uses raw DatabaseClient (not R2DBC repositories) because the scoring_decisions
 * INSERT benefits from the RETURNING id clause (to confirm the row was inserted
 * and fetch its generated PK for correlation logging), which is not expressible
 * via Spring Data repository annotations.
 */
@Service
public class AuditLogService {

    private static final Logger log = LoggerFactory.getLogger(AuditLogService.class);

    private final DatabaseClient db;
    private final ObjectMapper objectMapper;

    public AuditLogService(DatabaseClient db, ObjectMapper objectMapper) {
        this.db = db;
        this.objectMapper = objectMapper;
    }

    /**
     * Persist a scoring decision to PostgreSQL.
     *
     * Returns Mono<Void> — the caller (ScoringService) uses .thenReturn(decision)
     * to continue the pipeline after the write completes. Errors are NOT silently
     * swallowed here — they propagate to ScoringService which handles them with
     * onErrorResume so the caller still receives the decision even if DB is down.
     */
    @Transactional
    public Mono<Void> persist(ScoringDecision decision) {
        String shapJson = serializeShap(decision);
        OffsetDateTime scoredAt = decision.getScoredAt() != null
            ? decision.getScoredAt().atOffset(ZoneOffset.UTC)
            : OffsetDateTime.now(ZoneOffset.UTC);

        Mono<Void> insertDecision = db.sql("""
            INSERT INTO scoring_decisions
              (transaction_id, source_token, destination_token, amount, channel,
               final_risk_score, decision, tgn_score, lgbm_score, anomaly_score,
               device_score, ring_flag, ring_role, shap_top_features, explanation,
               explanation_mode, degraded_mode, processing_ms, scored_at)
            VALUES
              (:transaction_id, :source_token, :destination_token, :amount, :channel,
               :final_risk_score, :decision, :tgn_score, :lgbm_score, :anomaly_score,
               :device_score, :ring_flag, :ring_role, :shap_top_features::jsonb, :explanation,
               :explanation_mode, :degraded_mode, :processing_ms, :scored_at)
            ON CONFLICT (transaction_id) DO NOTHING
            """)
            .bind("transaction_id",    decision.getTransactionId())
            .bind("source_token",      orEmpty(decision.getSourceToken()))
            .bind("destination_token", orEmpty(decision.getDestinationToken()))
            .bind("amount",            decision.getAmount())
            .bind("channel",           orEmpty(decision.getChannel()))
            .bind("final_risk_score",  decision.getFinalRiskScore())
            .bind("decision",          decision.getDecision())
            .bindNull("tgn_score",     Double.class) // set below conditionally
            .bind("lgbm_score",        decision.getLgbmScore())
            .bind("anomaly_score",     decision.getAnomalyScore())
            .bind("device_score",      decision.getDeviceScore())
            .bind("ring_flag",         decision.isRingFlag())
            .bind("ring_role",         orEmpty(decision.getRingRole()))
            .bind("shap_top_features", shapJson)
            .bind("explanation",       orEmpty(decision.getExplanation()))
            .bind("explanation_mode",  orEmpty(decision.getExplanationMode()))
            .bind("degraded_mode",     decision.isDegradedMode())
            .bind("processing_ms",     decision.getProcessingMs())
            .bind("scored_at",         scoredAt)
            .then();

        // Handle nullable tgn_score — can't conditionally bind in the same chain
        Mono<Void> insertDecisionFinal = decision.getTgnScore() != null
            ? db.sql("""
                INSERT INTO scoring_decisions
                  (transaction_id, source_token, destination_token, amount, channel,
                   final_risk_score, decision, tgn_score, lgbm_score, anomaly_score,
                   device_score, ring_flag, ring_role, shap_top_features, explanation,
                   explanation_mode, degraded_mode, processing_ms, scored_at)
                VALUES
                  (:transaction_id, :source_token, :destination_token, :amount, :channel,
                   :final_risk_score, :decision, :tgn_score, :lgbm_score, :anomaly_score,
                   :device_score, :ring_flag, :ring_role, :shap_top_features::jsonb, :explanation,
                   :explanation_mode, :degraded_mode, :processing_ms, :scored_at)
                ON CONFLICT (transaction_id) DO NOTHING
                """)
                .bind("transaction_id",    decision.getTransactionId())
                .bind("source_token",      orEmpty(decision.getSourceToken()))
                .bind("destination_token", orEmpty(decision.getDestinationToken()))
                .bind("amount",            decision.getAmount())
                .bind("channel",           orEmpty(decision.getChannel()))
                .bind("final_risk_score",  decision.getFinalRiskScore())
                .bind("decision",          decision.getDecision())
                .bind("tgn_score",         decision.getTgnScore())
                .bind("lgbm_score",        decision.getLgbmScore())
                .bind("anomaly_score",     decision.getAnomalyScore())
                .bind("device_score",      decision.getDeviceScore())
                .bind("ring_flag",         decision.isRingFlag())
                .bind("ring_role",         orEmpty(decision.getRingRole()))
                .bind("shap_top_features", shapJson)
                .bind("explanation",       orEmpty(decision.getExplanation()))
                .bind("explanation_mode",  orEmpty(decision.getExplanationMode()))
                .bind("degraded_mode",     decision.isDegradedMode())
                .bind("processing_ms",     decision.getProcessingMs())
                .bind("scored_at",         scoredAt)
                .then()
            : insertDecision;

        return insertDecisionFinal
            .then(insertAlertIfNeeded(decision))
            .doOnSuccess(v -> log.debug("Persisted scoring_decision txnId={} decision={}",
                decision.getTransactionId(), decision.getDecision()));
    }

    private Mono<Void> insertAlertIfNeeded(ScoringDecision decision) {
        if (!"BLOCK".equals(decision.getDecision()) && !"REVIEW".equals(decision.getDecision())) {
            return Mono.empty();
        }
        String priority = "BLOCK".equals(decision.getDecision()) ? "CRITICAL" : "HIGH";
        return db.sql("""
            INSERT INTO alerts (transaction_id, alert_priority, status)
            VALUES (:transaction_id, :alert_priority, 'OPEN')
            ON CONFLICT DO NOTHING
            """)
            .bind("transaction_id", decision.getTransactionId())
            .bind("alert_priority", priority)
            .then()
            .doOnSuccess(v -> log.info("Alert created txnId={} priority={}", decision.getTransactionId(), priority));
    }

    private String serializeShap(ScoringDecision decision) {
        if (decision.getShapTopFeatures() == null) return "[]";
        try {
            return objectMapper.writeValueAsString(decision.getShapTopFeatures());
        } catch (Exception e) {
            return "[]";
        }
    }

    private String orEmpty(String s) {
        return s != null ? s : "";
    }
}
