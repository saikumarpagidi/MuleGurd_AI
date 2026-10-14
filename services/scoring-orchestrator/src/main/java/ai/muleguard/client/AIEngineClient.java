package ai.muleguard.client;

import ai.muleguard.model.LGBMScore;
import ai.muleguard.model.ScoringRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

/**
 * Reactive HTTP client for the ai-engine's POST /score endpoint.
 *
 * Wraps every call with the "aiEngine" Resilience4j circuit breaker.
 * If the ai-engine is unavailable (connection refused, timeout, 5xx),
 * the circuit breaker trips and the fallback returns LGBMScore.degraded()
 * immediately without waiting for the timeout — this is the key latency
 * protection mechanism for the 50ms P95 orchestrator gate.
 *
 * Request format mirrors Python ai-engine's TransactionInput schema:
 *   {transaction_id, from_account, to_account, amount, timestamp,
 *    payment_format, currency}
 */
@Component
public class AIEngineClient {

    private static final Logger log = LoggerFactory.getLogger(AIEngineClient.class);

    private final WebClient webClient;
    private final CircuitBreaker circuitBreaker;
    private final ObjectMapper objectMapper;

    public AIEngineClient(
        @Qualifier("aiEngineWebClient") WebClient webClient,
        @Qualifier("aiEngineBreaker") CircuitBreaker circuitBreaker,
        ObjectMapper objectMapper
    ) {
        this.webClient = webClient;
        this.circuitBreaker = circuitBreaker;
        this.objectMapper = objectMapper;
    }

    /**
     * POST /score → Mono<LGBMScore>
     *
     * The Mono is non-blocking and subscribed by ScoringService via Mono.zip().
     * Circuit breaker is applied as a reactive operator — no thread blocking.
     */
    public Mono<LGBMScore> score(ScoringRequest request) {
        ObjectNode body = buildRequestBody(request);

        return webClient.post()
            .uri("/score")
            .bodyValue(body)
            .retrieve()
            .onStatus(HttpStatus.SERVICE_UNAVAILABLE::equals,
                resp -> resp.bodyToMono(String.class)
                    .map(b -> new RuntimeException("ai-engine 503: " + b)))
            .onStatus(status -> status.is5xxServerError(),
                resp -> resp.bodyToMono(String.class)
                    .map(b -> new RuntimeException("ai-engine error " + resp.statusCode() + ": " + b)))
            .bodyToMono(LGBMScore.class)
            .transformDeferred(CircuitBreakerOperator.of(circuitBreaker))
            .onErrorResume(ex -> {
                log.warn("ai-engine call failed (txnId={}) — using degraded fallback: {}",
                    request.getTransactionId(), ex.getMessage());
                return Mono.just(LGBMScore.degraded(request.getTransactionId()));
            });
    }

    private ObjectNode buildRequestBody(ScoringRequest req) {
        ObjectNode body = objectMapper.createObjectNode();
        body.put("transaction_id", req.getTransactionId());
        body.put("from_account", req.getFromAccount());
        body.put("to_account", req.getToAccount());
        body.put("amount", req.getAmount());
        // ISO-8601 timestamp string
        String ts = req.getTimestamp() != null ? req.getTimestamp().toString() + "Z" : "2026-01-01T00:00:00Z";
        body.put("timestamp", ts);
        body.put("payment_format", req.getPaymentFormat() != null ? req.getPaymentFormat() : "OTHER");
        body.put("currency", req.getCurrency() != null ? req.getCurrency() : "USD");
        return body;
    }
}
