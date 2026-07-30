package ai.muleguard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * MuleGuard.AI — Scoring Orchestrator
 *
 * Coordinates parallel inference across three AI model services
 * using Project Reactor's non-blocking WebFlux framework.
 *
 * Scoring pipeline:
 *   1. Fetch account features from Redis (via feature-service)
 *   2. Fetch graph neighbourhood from Neo4j (via graph-service)
 *   3. Execute LightGBM, Autoencoder, and TGN models in parallel
 *   4. Fuse scores: 0.50×TGN + 0.30×LightGBM + 0.15×Autoencoder + 0.05×JA3
 *   5. Persist decision to PostgreSQL audit log
 */
@SpringBootApplication
public class ScoringOrchestratorApplication {

    public static void main(String[] args) {
        SpringApplication.run(ScoringOrchestratorApplication.class, args);
    }
}
