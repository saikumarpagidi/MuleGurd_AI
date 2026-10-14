package ai.muleguard.client;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Collections;
import java.util.Map;

/**
 * Reactive HTTP client for the feature-service's GET /features/{accountId} endpoint.
 *
 * The feature-service reads pre-computed account features from Redis (Feast online
 * store) and is the fastest of the three parallel calls (5ms budget). If it fails,
 * the ai-engine falls back to on-the-fly feature computation from the raw transaction
 * payload — this path was built in Sprint 4C (feature_store.py).
 *
 * On circuit-breaker open: returns empty Map which the ai-engine interprets as
 * "no Redis features available, use on-the-fly" — the same behaviour as a Redis miss.
 */
@Component
public class FeatureServiceClient {

    private static final Logger log = LoggerFactory.getLogger(FeatureServiceClient.class);

    private final WebClient webClient;
    private final CircuitBreaker circuitBreaker;

    public FeatureServiceClient(
        @Qualifier("featureServiceWebClient") WebClient webClient,
        @Qualifier("featureServiceBreaker") CircuitBreaker circuitBreaker
    ) {
        this.webClient = webClient;
        this.circuitBreaker = circuitBreaker;
    }

    /**
     * GET /features/{accountId} → Mono<Map<String, Double>>
     *
     * Subscribed in parallel via Mono.zip() in ScoringService.
     * The returned map is passed to the ai-engine request body as pre-computed
     * features, reducing on-the-fly computation latency.
     *
     * Returns empty map on any error — ai-engine handles the missing-features case.
     */
    public Mono<Map<String, Double>> fetchFeatures(String accountId) {
        return webClient.get()
            .uri("/features/{accountId}", accountId)
            .retrieve()
            .bodyToMono(new ParameterizedTypeReference<Map<String, Double>>() {})
            .transformDeferred(CircuitBreakerOperator.of(circuitBreaker))
            .onErrorResume(ex -> {
                log.debug("feature-service call failed for account={}: {} — ai-engine will use on-the-fly features",
                    accountId, ex.getMessage());
                return Mono.just(Collections.emptyMap());
            });
    }
}
