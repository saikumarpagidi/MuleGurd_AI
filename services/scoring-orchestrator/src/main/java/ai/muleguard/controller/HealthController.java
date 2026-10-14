package ai.muleguard.controller;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

/**
 * Custom health endpoint to expose detailed circuit breaker state.
 * 
 * While Actuator provides /actuator/health, this custom endpoint
 * formats the health check specifically for k8s liveness/readiness probes,
 * allowing DevOps to immediately see if the orchestrator is running in
 * degraded mode due to a downstream failure.
 */
@RestController
public class HealthController {

    private final CircuitBreakerRegistry circuitBreakerRegistry;

    public HealthController(CircuitBreakerRegistry circuitBreakerRegistry) {
        this.circuitBreakerRegistry = circuitBreakerRegistry;
    }

    @GetMapping({"/health", "/api/v1/health"})
    public Mono<ResponseEntity<Map<String, Object>>> getHealth() {
        Map<String, Object> response = new HashMap<>();
        response.put("service", "scoring-orchestrator");
        response.put("status", "ok");
        
        Map<String, String> cbStates = new HashMap<>();
        boolean anyDegraded = false;
        
        for (CircuitBreaker cb : circuitBreakerRegistry.getAllCircuitBreakers()) {
            String state = cb.getState().name();
            cbStates.put(cb.getName(), state);
            if (!"CLOSED".equals(state)) {
                anyDegraded = true;
            }
        }
        
        response.put("circuitBreakers", cbStates);
        response.put("degradedMode", anyDegraded);
        
        return Mono.just(ResponseEntity.ok(response));
    }
}
