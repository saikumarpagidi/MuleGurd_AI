package ai.muleguard.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * DTO matching the graph-service POST /graph-score response schema.
 *
 * tgn_score may be null if the graph-service is in full degraded mode
 * (Neo4j unreachable AND TGN model not loaded). The fusion engine handles
 * null by renormalizing the remaining model weights.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class TGNScore {

    @JsonProperty("tgn_score")
    private Double tgnScore;   // null in full degraded mode

    @JsonProperty("is_mule_probability")
    private Double isMuleProbability;


    @JsonProperty("ring_role")
    private String ringRole = "NONE";

    @JsonProperty("ring_probability")
    private double ringProbability = 0.5;

    @JsonProperty("in_scc_ring")
    private boolean inSccRing = false;

    @JsonProperty("community_id")
    private int communityId = -1;

    @JsonProperty("pagerank")
    private double pagerank = 0.0;

    @JsonProperty("processing_time_ms")
    private double processingTimeMs;

    @JsonProperty("degraded_mode")
    private boolean degradedMode = false;

    @JsonProperty("model_version")
    private String modelVersion = "unknown";

    // ── Static factory: degraded fallback ───────────────────────────────────

    /**
     * Returned when the graph-service circuit breaker opens.
     *
     * tgn_score=null signals the fusion engine to redistribute the 0.50 TGN
     * weight across the remaining models rather than using a placeholder value
     * that could bias the final risk score up or down.
     */
    public static TGNScore degraded() {
        TGNScore s = new TGNScore();
        s.tgnScore = null;
        s.ringRole = "NONE";
        s.ringProbability = 0.5;
        s.inSccRing = false;
        s.isMuleProbability = null;
        s.degradedMode = true;
        s.modelVersion = "circuit_breaker_open";
        return s;
    }

    // ── Getters / Setters ────────────────────────────────────────────────────

    public Double getTgnScore() { return tgnScore; }
    public void setTgnScore(Double tgnScore) { this.tgnScore = tgnScore; }

    public Double getIsMuleProbability() { return isMuleProbability; }
    public void setIsMuleProbability(Double isMuleProbability) { this.isMuleProbability = isMuleProbability; }

    public String getRingRole() { return ringRole; }
    public void setRingRole(String ringRole) { this.ringRole = ringRole; }

    public double getRingProbability() { return ringProbability; }
    public void setRingProbability(double ringProbability) { this.ringProbability = ringProbability; }

    public boolean isInSccRing() { return inSccRing; }
    public void setInSccRing(boolean inSccRing) { this.inSccRing = inSccRing; }

    public int getCommunityId() { return communityId; }
    public void setCommunityId(int communityId) { this.communityId = communityId; }

    public double getPagerank() { return pagerank; }
    public void setPagerank(double pagerank) { this.pagerank = pagerank; }

    public double getProcessingTimeMs() { return processingTimeMs; }
    public void setProcessingTimeMs(double processingTimeMs) { this.processingTimeMs = processingTimeMs; }

    public boolean isDegradedMode() { return degradedMode; }
    public void setDegradedMode(boolean degradedMode) { this.degradedMode = degradedMode; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }
}
