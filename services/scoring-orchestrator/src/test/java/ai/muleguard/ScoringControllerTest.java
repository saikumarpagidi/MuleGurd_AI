package ai.muleguard;

import ai.muleguard.controller.ScoringController;
import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import ai.muleguard.service.ScoringService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.WebFluxTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@WebFluxTest(ScoringController.class)
class ScoringControllerTest {

    @Autowired
    private WebTestClient webClient;

    @MockBean
    private ScoringService scoringService;

    @Test
    void testScoreEndpoint() {
        ScoringRequest request = new ScoringRequest();
        request.setTransactionId("txn-1");

        ScoringDecision decision = new ScoringDecision();
        decision.setTransactionId("txn-1");
        decision.setDecision("APPROVE");
        decision.setFinalRiskScore(0.1);

        when(scoringService.score(any(ScoringRequest.class))).thenReturn(Mono.just(decision));

        webClient.post().uri("/api/v1/score")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue("{\"transactionId\":\"txn-1\",\"amount\":100.0}")
            .exchange()
            .expectStatus().isOk()
            .expectBody()
            .jsonPath("$.transactionId").isEqualTo("txn-1")
            .jsonPath("$.decision").isEqualTo("APPROVE");
    }
}
