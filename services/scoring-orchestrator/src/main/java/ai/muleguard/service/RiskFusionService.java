package ai.muleguard.service;

import ai.muleguard.config.ScoringProperties;
import ai.muleguard.model.LGBMScore;
import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import ai.muleguard.model.TGNScore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Risk Fusion Engine — Sprint 6 core logic.
 *
 * Implements the weighted ensemble formula from the enterprise architecture doc:
 *
 *   finalRisk = 0.50 × TGN + 0.30 × LightGBM + 0.15 × Autoencoder + 0.05 × Device
 *
 * Weights are loaded from application.yml (muleguard.weights.*) and can be
 * overridden per environment via env vars, enabling A/B testing of different
 * fusion configurations without a code deploy.
 *
 * GRACEFUL DEGRADATION — null TGN score handling:
 *   When graph-service circuit breaker is open, TGNScore.tgnScore=null.
 *   In this case the TGN weight (0.50) is redistributed proportionally among
 *   the remaining models:
 *
 *     lgbm_effective  = 0.30 / (1 - 0.50) = 0.60
 *     ae_effective    = 0.15 / (1 - 0.50) = 0.30
 *     device_effective= 0.05 / (1 - 0.50) = 0.10
 *
 *   The renormalized weights still sum to 1.0. The decision is still issued
 *   (not blocked), but ScoringDecision.degradedMode=true is set so downstream
 *   consumers and dashboards can display a caveat.
 *
 * DECISION THRESHOLDS:
 *   APPROVE  → finalRisk < 0.45
 *   REVIEW   → 0.45 ≤ finalRisk < 0.75
 *   BLOCK    → finalRisk ≥ 0.75
 *
 * This class is PURE — no I/O, no reactive types. It takes already-resolved
 * DTOs and returns a ScoringDecision synchronously. This makes unit testing
 * trivial (RiskFusionServiceTest.java).
 */
@Service
public class RiskFusionService {

    private static final Logger log = LoggerFactory.getLogger(RiskFusionService.class);

    private final ScoringProperties props;

    public RiskFusionService(ScoringProperties props) {
        this.props = props;
    }

    /**
     * Fuse model scores into a final risk decision.
     *
     * @param request   Original scoring request (provides transaction metadata)
     * @param lgbmScore AI Engine response (LightGBM + AE scores)
     * @param tgnScore  Graph Service response (TGN score, possibly null if degraded)
     * @return          Authoritative ScoringDecision ready for PostgreSQL write
     */
    public ScoringDecision fuse(ScoringRequest request, LGBMScore lgbmScore, TGNScore tgnScore) {
        ScoringProperties.Weights w = props.getWeights();

        double wTgn  = w.getTgn();
        double wLgbm = w.getLgbm();
        double wAe   = w.getAutoencoder();
        double wDev  = w.getDevice();

        boolean tgnDegraded = tgnScore == null || tgnScore.isDegradedMode() || tgnScore.getTgnScore() == null;

        double finalRisk;
        boolean degraded = lgbmScore.isDegraded() || tgnDegraded;

        if (tgnDegraded) {
            // Renormalize: distribute TGN's weight among remaining models
            double remaining = 1.0 - wTgn;
            if (remaining <= 0) remaining = 1.0;
            double wLgbmEff = wLgbm / remaining;
            double wAeEff   = wAe   / remaining;
            double wDevEff  = wDev  / remaining;

            // Device signal: derived from JA3 hash presence (0.0 if no device data)
            double deviceSignal = computeDeviceSignal(request);

            finalRisk = wLgbmEff * lgbmScore.getLgbmScore()
                      + wAeEff   * lgbmScore.getAeScore()
                      + wDevEff  * deviceSignal;

            log.warn("TGN degraded for txnId={} — renormalized weights: lgbm={:.2f} ae={:.2f} dev={:.2f}",
                request.getTransactionId(), wLgbmEff, wAeEff, wDevEff);
        } else {
            double deviceSignal = computeDeviceSignal(request);
            finalRisk = wTgn  * tgnScore.getTgnScore()
                      + wLgbm * lgbmScore.getLgbmScore()
                      + wAe   * lgbmScore.getAeScore()
                      + wDev  * deviceSignal;
        }

        // Clamp to [0, 1] — floating point arithmetic should keep it there,
        // but defensive clamping prevents any edge-case from sending 1.001 to Postgres
        finalRisk = Math.max(0.0, Math.min(1.0, finalRisk));

        String decision = computeDecision(finalRisk);
        boolean ringFlag = tgnScore != null && (tgnScore.isInSccRing() ||
                           "HUB".equals(tgnScore.getRingRole()) || "MULE".equals(tgnScore.getRingRole()));

        ScoringDecision result = new ScoringDecision();
        result.setTransactionId(request.getTransactionId());
        result.setSourceToken(request.getFromAccount());
        result.setDestinationToken(request.getToAccount());
        result.setAmount(request.getAmount());
        result.setChannel(request.getPaymentFormat());
        result.setFinalRiskScore(finalRisk);
        result.setDecision(decision);
        result.setLgbmScore(lgbmScore.getLgbmScore());
        result.setAnomalyScore(lgbmScore.getAeScore());
        result.setDeviceScore(computeDeviceSignal(request));
        result.setTgnScore(tgnDegraded ? null : tgnScore.getTgnScore());
        result.setIsMuleProbability(tgnDegraded ? null : tgnScore.getIsMuleProbability());
        result.setRingFlag(ringFlag);
        result.setRingRole(tgnScore != null ? tgnScore.getRingRole() : "NONE");
        result.setShapTopFeatures(lgbmScore.getShapTop5());
        result.setDegradedMode(degraded);
        result.setExplanationMode("none");  // Sprint 7 will fill this

        log.info("FUSION txnId={} | tgn={} lgbm={:.4f} ae={:.4f} -> final={:.4f} | decision={}{}",
            request.getTransactionId(),
            tgnDegraded ? "DEGRADED" : String.format("%.4f", tgnScore.getTgnScore()),
            lgbmScore.getLgbmScore(),
            lgbmScore.getAeScore(),
            finalRisk,
            decision,
            degraded ? " [DEGRADED]" : "");

        return result;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private String computeDecision(double riskScore) {
        double blockThreshold  = props.getThresholds().getBlock();
        double reviewThreshold = props.getThresholds().getReview();
        if (riskScore >= blockThreshold)  return "BLOCK";
        if (riskScore >= reviewThreshold) return "REVIEW";
        return "APPROVE";
    }

    /**
     * Derive a device risk signal from the JA3 fingerprint hash.
     *
     * In the full implementation (Sprint 10), this would query a Redis-based
     * JA3 reputation store that tracks historically suspicious device fingerprints.
     * For now, we use a simple heuristic: presence of a ja3Hash = 0.1 (unknown),
     * absence = 0.0 (no device data). This contributes at most 0.05 × 0.1 = 0.005
     * to the final risk score, effectively a tie-breaker.
     */
    private double computeDeviceSignal(ScoringRequest request) {
        if (request.getJa3Hash() == null || request.getJa3Hash().isBlank()) {
            return 0.0;
        }
        // All unknown JA3 hashes contribute a small positive signal.
        // Known-bad hashes (Redis lookup) would return 1.0 here in production.
        return 0.1;
    }
}
