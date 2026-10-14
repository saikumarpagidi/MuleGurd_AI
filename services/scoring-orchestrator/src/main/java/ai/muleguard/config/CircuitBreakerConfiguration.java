package ai.muleguard.config;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * Resilience4j Circuit Breaker configuration for all three downstream model services.
 *
 * Each service gets an identically configured circuit breaker (matching the
 * Resilience4j YAML config in application.yml). The @Bean methods here serve
 * as programmatic fallback references for the client classes — the YAML config
 * takes precedence when the Spring Boot auto-configuration picks it up.
 *
 * Circuit Breaker policy (per service):
 *   - Sliding window: 10 calls (COUNT_BASED)
 *   - Failure threshold: 50% (5 failures in any 10-call window opens the CB)
 *   - Wait duration in open state: 30 seconds
 *   - Half-open: 3 test calls before fully closing
 *   - Automatic transition: open → half-open after wait duration
 *
 * Fallback behaviour (applied in the client classes):
 *   - aiEngine open → LGBMScore.degraded(transactionId) [lgbm=0.5, ae=0.5]
 *   - graphService open → TGNScore.degraded() [tgnScore=null, degraded=true]
 *   - featureService open → empty feature map (ai-engine uses on-the-fly fallback)
 */
@Configuration
public class CircuitBreakerConfiguration {

    private static final io.github.resilience4j.circuitbreaker.CircuitBreakerConfig SHARED_CONFIG =
        io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(10)
            .failureRateThreshold(50.0f)
            .waitDurationInOpenState(Duration.ofSeconds(30))
            .permittedNumberOfCallsInHalfOpenState(3)
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            // Record all exceptions as failures — no filtering
            .recordExceptions(Exception.class)
            .build();

    @Bean
    public CircuitBreakerRegistry circuitBreakerRegistry() {
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(SHARED_CONFIG);

        // Pre-register named circuit breakers so the YAML config picks them up
        // and the application health endpoint reports their state correctly.
        registry.circuitBreaker("aiEngine", SHARED_CONFIG);
        registry.circuitBreaker("graphService", SHARED_CONFIG);
        registry.circuitBreaker("featureService", SHARED_CONFIG);

        return registry;
    }

    @Bean("aiEngineBreaker")
    public CircuitBreaker aiEngineBreaker(CircuitBreakerRegistry registry) {
        return registry.circuitBreaker("aiEngine");
    }

    @Bean("graphServiceBreaker")
    public CircuitBreaker graphServiceBreaker(CircuitBreakerRegistry registry) {
        return registry.circuitBreaker("graphService");
    }

    @Bean("featureServiceBreaker")
    public CircuitBreaker featureServiceBreaker(CircuitBreakerRegistry registry) {
        return registry.circuitBreaker("featureService");
    }
}
