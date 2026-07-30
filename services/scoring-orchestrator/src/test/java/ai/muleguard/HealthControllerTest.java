package ai.muleguard;

import ai.muleguard.controller.HealthController;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.WebFluxTest;
import org.springframework.test.web.reactive.server.WebTestClient;

/**
 * Scoring Orchestrator — Health Endpoint Tests
 */
@WebFluxTest(controllers = HealthController.class)
class HealthControllerTest {

    @Autowired
    private WebTestClient webTestClient;

    @Test
    void healthEndpoint_returns200() {
        webTestClient.get()
            .uri("/api/v1/health")
            .exchange()
            .expectStatus().isOk();
    }

    @Test
    void healthEndpoint_reportsStatusOk() {
        webTestClient.get()
            .uri("/api/v1/health")
            .exchange()
            .expectStatus().isOk()
            .expectBody()
            .jsonPath("$.status").isEqualTo("ok");
    }

    @Test
    void healthEndpoint_identifiesService() {
        webTestClient.get()
            .uri("/api/v1/health")
            .exchange()
            .expectStatus().isOk()
            .expectBody()
            .jsonPath("$.service").isEqualTo("scoring-orchestrator");
    }

    @Test
    void healthEndpoint_returnsJsonContentType() {
        webTestClient.get()
            .uri("/api/v1/health")
            .exchange()
            .expectHeader()
            .contentType("application/json");
    }
}
