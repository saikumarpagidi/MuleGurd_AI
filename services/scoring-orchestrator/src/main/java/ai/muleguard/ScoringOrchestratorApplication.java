package ai.muleguard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.kafka.annotation.EnableKafka;

/**
 * MuleGuard.AI — Scoring Orchestrator
 *
 * Coordinates parallel inference across three AI model services
 * using Project Reactor's non-blocking WebFlux framework.
 *
 * Scoring pipeline (Sprint 6):
 *   1. Receive transaction from Kafka (muleguard.transactions) or HTTP POST
 *   2. Fetch account features from Redis (via feature-service) [async]
 *   3. Fetch graph neighbourhood from Neo4j (via graph-service) [async]
 *   4. Execute AI Engine (LightGBM + Autoencoder + SHAP) [async]
 *   5. Execute TGN via graph-service [async]
 *   Steps 2-5 run in parallel via Mono.zip() — total wall-clock = max(latencies)
 *   6. Fuse scores: 0.50×TGN + 0.30×LightGBM + 0.15×Autoencoder + 0.05×Device
 *   7. Persist decision to PostgreSQL scoring_decisions (R2DBC)
 *   8. If BLOCK or REVIEW: publish to muleguard.alerts Kafka topic
 *
 * Resilience4j circuit breakers protect all downstream model service calls.
 * If a service fails, the orchestrator renormalizes weights and continues.
 */
@SpringBootApplication
@EnableKafka
public class ScoringOrchestratorApplication {

    public static void main(String[] args) {
        SpringApplication.run(ScoringOrchestratorApplication.class, args);
    }
}
