package ai.muleguard.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;

/**
 * DTO matching the ai-engine POST /score response schema.
 *
 * Field names use camelCase Java style; Jackson maps from the snake_case
 * JSON response via @JsonProperty annotations. This is the complete response
 * from the Python ai-engine (LightGBM + Autoencoder + SHAP).
 *
 * The authoritative fusion happens here in the orchestrator — the
 * decision_preliminary field from ai-engine is informational only.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class LGBMScore {

    @JsonProperty("transaction_id")
    private String transactionId;

    @JsonProperty("lgbm_score")
    private double lgbmScore;

    @JsonProperty("ae_score")
    private double aeScore;

    @JsonProperty("ae_reconstruction_error")
    private double aeReconstructionError;

    @JsonProperty("shap_top_5")
    private JsonNode shapTop5;

    @JsonProperty("decision_preliminary")
    private String decisionPreliminary;

    @JsonProperty("feature_source")
    private String featureSource;

    @JsonProperty("model_versions")
    private JsonNode modelVersions;

    @JsonProperty("inference_latency_ms")
    private double inferenceLatencyMs;

    /** Whether this result was produced by a circuit-breaker fallback. */
    private boolean degraded = false;

    // ── Static factory: degraded fallback ───────────────────────────────────

    /**
     * Returns a safe degraded-mode score when the ai-engine circuit breaker opens.
     * Uses a neutral lgbm_score=0.5 so the fusion engine doesn't blindly block
     * or approve on a dead upstream — it degrades gracefully.
     */
    public static LGBMScore degraded(String transactionId) {
        LGBMScore s = new LGBMScore();
        s.transactionId = transactionId;
        s.lgbmScore = 0.5;
        s.aeScore = 0.5;
        s.aeReconstructionError = 0.0;
        s.decisionPreliminary = "REVIEW";
        s.featureSource = "degraded";
        s.inferenceLatencyMs = 0.0;
        s.degraded = true;
        return s;
    }

    // ── Getters / Setters ────────────────────────────────────────────────────

    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }

    public double getLgbmScore() { return lgbmScore; }
    public void setLgbmScore(double lgbmScore) { this.lgbmScore = lgbmScore; }

    public double getAeScore() { return aeScore; }
    public void setAeScore(double aeScore) { this.aeScore = aeScore; }

    public double getAeReconstructionError() { return aeReconstructionError; }
    public void setAeReconstructionError(double aeReconstructionError) { this.aeReconstructionError = aeReconstructionError; }

    public JsonNode getShapTop5() { return shapTop5; }
    public void setShapTop5(JsonNode shapTop5) { this.shapTop5 = shapTop5; }

    public String getDecisionPreliminary() { return decisionPreliminary; }
    public void setDecisionPreliminary(String decisionPreliminary) { this.decisionPreliminary = decisionPreliminary; }

    public String getFeatureSource() { return featureSource; }
    public void setFeatureSource(String featureSource) { this.featureSource = featureSource; }

    public JsonNode getModelVersions() { return modelVersions; }
    public void setModelVersions(JsonNode modelVersions) { this.modelVersions = modelVersions; }

    public double getInferenceLatencyMs() { return inferenceLatencyMs; }
    public void setInferenceLatencyMs(double inferenceLatencyMs) { this.inferenceLatencyMs = inferenceLatencyMs; }

    public boolean isDegraded() { return degraded; }
    public void setDegraded(boolean degraded) { this.degraded = degraded; }
}
