package ai.muleguard.model;

import com.fasterxml.jackson.databind.JsonNode;

import java.time.Instant;

/**
 * Authoritative scoring decision produced by the Sprint 6 Risk Fusion Engine.
 *
 * This is the FINAL decision — not to be confused with ai-engine's
 * DecisionPreliminary (LightGBM+AE only). This record includes TGN weights
 * and is what gets written to PostgreSQL scoring_decisions table and
 * published to muleguard.alerts Kafka topic.
 *
 * Field names map 1:1 to the scoring_decisions table columns in
 * infrastructure/postgres/schema.sql.
 */
public class ScoringDecision {

    private String transactionId;
    private String sourceToken;
    private String destinationToken;
    private double amount;
    private String channel;

    // ── Risk scores from individual models ─────────────────────────────────
    private Double tgnScore;         // null if graph-service was in degraded mode
    private Double isMuleProbability; // from TGN v4
    private double lgbmScore;
    private double anomalyScore;
    private double deviceScore;

    // ── Fusion result ────────────────────────────────────────────────────────
    private double finalRiskScore;   // weighted ensemble: 0.50×TGN + 0.30×LGBM + 0.15×AE + 0.05×Device
    private String decision;         // APPROVE | REVIEW | BLOCK

    // ── Ring signals ─────────────────────────────────────────────────────────
    private boolean ringFlag;
    private String ringRole;         // HUB | BRIDGE | MULE | NONE

    // ── Explainability ───────────────────────────────────────────────────────
    private JsonNode shapTopFeatures;  // JSON array of top 5 SHAP features from ai-engine
    private String explanation;
    private String explanationMode = "none";  // llm | rule_based | none

    // ── Metadata ─────────────────────────────────────────────────────────────
    private boolean degradedMode;    // true if any upstream service used circuit breaker fallback
    private int processingMs;
    private Instant scoredAt = Instant.now();

    // ── Static factory methods ───────────────────────────────────────────────

    /** Creates a BLOCK decision from the fusion result. */
    public static ScoringDecision block(String txnId, double riskScore) {
        ScoringDecision d = new ScoringDecision();
        d.transactionId = txnId;
        d.finalRiskScore = riskScore;
        d.decision = "BLOCK";
        return d;
    }

    /** Creates a fully degraded decision (all upstreams failed). */
    public static ScoringDecision degradedFallback(ScoringRequest request) {
        ScoringDecision d = new ScoringDecision();
        d.transactionId = request.getTransactionId();
        d.sourceToken = request.getFromAccount();
        d.destinationToken = request.getToAccount();
        d.amount = request.getAmount();
        d.channel = request.getPaymentFormat();
        d.finalRiskScore = 0.5;  // conservative neutral when completely degraded
        d.decision = "REVIEW";
        d.degradedMode = true;
        d.ringFlag = false;
        d.ringRole = "NONE";
        d.tgnScore = null;
        d.isMuleProbability = null;
        d.lgbmScore = 0.5;
        d.anomalyScore = 0.5;
        d.deviceScore = 0.0;
        d.explanationMode = "none";
        return d;
    }

    // ── Getters / Setters ────────────────────────────────────────────────────

    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }

    public String getSourceToken() { return sourceToken; }
    public void setSourceToken(String sourceToken) { this.sourceToken = sourceToken; }

    public String getDestinationToken() { return destinationToken; }
    public void setDestinationToken(String destinationToken) { this.destinationToken = destinationToken; }

    public double getAmount() { return amount; }
    public void setAmount(double amount) { this.amount = amount; }

    public String getChannel() { return channel; }
    public void setChannel(String channel) { this.channel = channel; }

    public Double getTgnScore() { return tgnScore; }
    public void setTgnScore(Double tgnScore) { this.tgnScore = tgnScore; }

    public Double getIsMuleProbability() { return isMuleProbability; }
    public void setIsMuleProbability(Double isMuleProbability) { this.isMuleProbability = isMuleProbability; }

    public double getLgbmScore() { return lgbmScore; }
    public void setLgbmScore(double lgbmScore) { this.lgbmScore = lgbmScore; }

    public double getAnomalyScore() { return anomalyScore; }
    public void setAnomalyScore(double anomalyScore) { this.anomalyScore = anomalyScore; }

    public double getDeviceScore() { return deviceScore; }
    public void setDeviceScore(double deviceScore) { this.deviceScore = deviceScore; }

    public double getFinalRiskScore() { return finalRiskScore; }
    public void setFinalRiskScore(double finalRiskScore) { this.finalRiskScore = finalRiskScore; }

    public String getDecision() { return decision; }
    public void setDecision(String decision) { this.decision = decision; }

    public boolean isRingFlag() { return ringFlag; }
    public void setRingFlag(boolean ringFlag) { this.ringFlag = ringFlag; }

    public String getRingRole() { return ringRole; }
    public void setRingRole(String ringRole) { this.ringRole = ringRole; }

    public JsonNode getShapTopFeatures() { return shapTopFeatures; }
    public void setShapTopFeatures(JsonNode shapTopFeatures) { this.shapTopFeatures = shapTopFeatures; }

    public String getExplanation() { return explanation; }
    public void setExplanation(String explanation) { this.explanation = explanation; }

    public String getExplanationMode() { return explanationMode; }
    public void setExplanationMode(String explanationMode) { this.explanationMode = explanationMode; }

    public boolean isDegradedMode() { return degradedMode; }
    public void setDegradedMode(boolean degradedMode) { this.degradedMode = degradedMode; }

    public int getProcessingMs() { return processingMs; }
    public void setProcessingMs(int processingMs) { this.processingMs = processingMs; }

    public Instant getScoredAt() { return scoredAt; }
    public void setScoredAt(Instant scoredAt) { this.scoredAt = scoredAt; }
}
