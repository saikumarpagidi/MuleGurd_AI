package ai.muleguard.model;

/**
 * Extracted Autoencoder sub-result from the LGBMScore response.
 * The ai-engine returns both LightGBM and Autoencoder scores in a single
 * /score call — this DTO holds just the AE portion for clarity in the
 * fusion engine's method signatures.
 */
public class AnomalyScore {

    private double aeScore;
    private double aeReconstructionError;
    private boolean degraded;

    public AnomalyScore() {}

    public AnomalyScore(double aeScore, double aeReconstructionError, boolean degraded) {
        this.aeScore = aeScore;
        this.aeReconstructionError = aeReconstructionError;
        this.degraded = degraded;
    }

    /** Extract from the combined LGBMScore response. */
    public static AnomalyScore from(LGBMScore lgbmScore) {
        return new AnomalyScore(
            lgbmScore.getAeScore(),
            lgbmScore.getAeReconstructionError(),
            lgbmScore.isDegraded()
        );
    }

    public double getAeScore() { return aeScore; }
    public void setAeScore(double aeScore) { this.aeScore = aeScore; }

    public double getAeReconstructionError() { return aeReconstructionError; }
    public void setAeReconstructionError(double v) { this.aeReconstructionError = v; }

    public boolean isDegraded() { return degraded; }
    public void setDegraded(boolean degraded) { this.degraded = degraded; }
}
