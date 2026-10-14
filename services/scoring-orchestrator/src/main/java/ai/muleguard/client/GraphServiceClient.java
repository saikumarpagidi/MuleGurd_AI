package ai.muleguard.client;

import ai.muleguard.model.ScoringRequest;
import ai.muleguard.model.TGNScore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/**
 * Reactive HTTP client for the graph-service's POST /graph-score endpoint.
 *
 * The graph-service has the tightest latency budget (25ms) since it must
 * complete Neo4j lookup + TGN inference within that window. The circuit
 * breaker here is especially important: if Neo4j is degraded, the graph-
 * service will return 503s quickly, and this CB ensures those are absorbed
 * without propagating to the orchestrator's latency.
 *
 * When the CB trips, TGNScore.degraded() is returned with tgnScore=null.
 * RiskFusionService sees null and renormalizes: remaining weights become
 * 0.60 LGBM, 0.30 AE, 0.10 Device (maintaining relative ratios).
 */
@Component
public class GraphServiceClient {

    private static final Logger log = LoggerFactory.getLogger(GraphServiceClient.class);

    private final WebClient webClient;
    private final CircuitBreaker circuitBreaker;
    private final ObjectMapper objectMapper;

    public GraphServiceClient(
        @Qualifier("graphServiceWebClient") WebClient webClient,
        @Qualifier("graphServiceBreaker") CircuitBreaker circuitBreaker,
        ObjectMapper objectMapper
    ) {
        this.webClient = webClient;
        this.circuitBreaker = circuitBreaker;
        this.objectMapper = objectMapper;
    }

    /**
     * POST /graph-score → Mono<TGNScore>
     *
     * Subscribed in parallel with AIEngineClient.score() via Mono.zip().
     */
    public Mono<TGNScore> graphScore(ScoringRequest request) {
        ObjectNode body = buildRequestBody(request);

        return webClient.post()
            .uri("/graph-score")
            .bodyValue(body)
            .retrieve()
            .onStatus(status -> status.is5xxServerError(),
                resp -> resp.bodyToMono(String.class)
                    .map(b -> new RuntimeException("graph-service error " + resp.statusCode() + ": " + b)))
            .bodyToMono(TGNScore.class)
            .transformDeferred(CircuitBreakerOperator.of(circuitBreaker))
            .onErrorResume(ex -> {
                log.warn("graph-service call failed (txnId={}) — TGN degraded: {}",
                    request.getTransactionId(), ex.getMessage());
                return Mono.just(TGNScore.degraded());
            });
    }

    private ObjectNode buildRequestBody(ScoringRequest req) {
        ObjectNode body = objectMapper.createObjectNode();
        // Include transaction_id for end-to-end traceability in graph-service logs
        if (req.getTransactionId() != null) {
            body.put("transaction_id", req.getTransactionId());
        }
        body.put("from_account", req.getFromAccount());
        body.put("to_account", req.getToAccount());
        body.put("amount", req.getAmount());
        // LocalDateTime.toString() gives "2022-09-01T00:01" (no seconds, no Z).
        // Appending "Z" directly produces "2022-09-01T00:01Z" — invalid ISO-8601.
        // formatTimestamp() ensures a full "yyyy-MM-ddTHH:mm:ssZ" string.
        body.put("timestamp", formatTimestamp(req));
        body.put("payment_format", req.getPaymentFormat() != null ? req.getPaymentFormat() : "OTHER");
        body.put("currency", req.getCurrency() != null ? req.getCurrency() : "USD");
        return body;
    }

    /**
     * Produce a well-formed ISO-8601 datetime string from an Instant.
     * Instant.toString() always produces "yyyy-MM-ddTHH:mm:ssZ" — no further
     * formatting needed.
     */
    private String formatTimestamp(ScoringRequest req) {
        if (req.getTimestamp() == null) {
            return "2026-01-01T00:00:00Z";
        }
        return req.getTimestamp().toString();   // e.g. "2022-09-01T00:01:00Z"
    }
}
