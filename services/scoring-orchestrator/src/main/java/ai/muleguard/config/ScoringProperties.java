package ai.muleguard.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Strongly-typed binding for all muleguard.* properties in application.yml.
 *
 * Using @ConfigurationProperties instead of scattered @Value annotations
 * means all tunable parameters are in one place, validated at startup,
 * and easily overridable per environment via env vars.
 *
 * The weights sum constraint (must equal 1.0) is enforced at build time
 * in RiskFusionService — this class is a pure data container.
 */
@Configuration
@ConfigurationProperties(prefix = "muleguard")
public class ScoringProperties {

    private final Services services = new Services();
    private final Weights weights = new Weights();
    private final Thresholds thresholds = new Thresholds();
    private final Kafka kafka = new Kafka();

    // ── Nested classes ───────────────────────────────────────────────────────

    public static class ServiceConfig {
        private String url;
        private long timeoutMs;

        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public long getTimeoutMs() { return timeoutMs; }
        public void setTimeoutMs(long timeoutMs) { this.timeoutMs = timeoutMs; }
    }

    public static class Services {
        private ServiceConfig aiEngine = new ServiceConfig();
        private ServiceConfig graphService = new ServiceConfig();
        private ServiceConfig featureService = new ServiceConfig();

        public ServiceConfig getAiEngine() { return aiEngine; }
        public void setAiEngine(ServiceConfig c) { this.aiEngine = c; }
        public ServiceConfig getGraphService() { return graphService; }
        public void setGraphService(ServiceConfig c) { this.graphService = c; }
        public ServiceConfig getFeatureService() { return featureService; }
        public void setFeatureService(ServiceConfig c) { this.featureService = c; }
    }

    public static class Weights {
        private double tgn = 0.50;
        private double lgbm = 0.30;
        private double autoencoder = 0.15;
        private double device = 0.05;

        public double getTgn() { return tgn; }
        public void setTgn(double tgn) { this.tgn = tgn; }
        public double getLgbm() { return lgbm; }
        public void setLgbm(double lgbm) { this.lgbm = lgbm; }
        public double getAutoencoder() { return autoencoder; }
        public void setAutoencoder(double autoencoder) { this.autoencoder = autoencoder; }
        public double getDevice() { return device; }
        public void setDevice(double device) { this.device = device; }
    }

    public static class Thresholds {
        private double review = 0.45;
        private double block = 0.75;

        public double getReview() { return review; }
        public void setReview(double review) { this.review = review; }
        public double getBlock() { return block; }
        public void setBlock(double block) { this.block = block; }
    }

    public static class KafkaTopics {
        private String transactions = "muleguard.transactions";
        private String alerts = "muleguard.alerts";

        public String getTransactions() { return transactions; }
        public void setTransactions(String transactions) { this.transactions = transactions; }
        public String getAlerts() { return alerts; }
        public void setAlerts(String alerts) { this.alerts = alerts; }
    }

    public static class Kafka {
        private KafkaTopics topics = new KafkaTopics();

        public KafkaTopics getTopics() { return topics; }
        public void setTopics(KafkaTopics topics) { this.topics = topics; }
    }

    // ── Root getters ─────────────────────────────────────────────────────────

    public Services getServices() { return services; }
    public Weights getWeights() { return weights; }
    public Thresholds getThresholds() { return thresholds; }
    public Kafka getKafka() { return kafka; }
}
