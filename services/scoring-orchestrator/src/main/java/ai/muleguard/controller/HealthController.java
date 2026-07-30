package ai.muleguard.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Health endpoint for the Scoring Orchestrator.
 *
 * Returns the operational status of this service.
 * Used by Docker Compose health checks, Kubernetes liveness probes,
 * and the integration test suite.
 *
 * When the full scoring pipeline is active, this endpoint will additionally
 * report the circuit breaker state for each downstream model service.
 */
@RestController
@RequestMapping("/api/v1")
public class HealthController {

    @Value("${muleguard.service.name:scoring-orchestrator}")
    private String serviceName;

    /**
     * Returns service health status.
     *
     * @return Reactive map containing status and service name.
     */
    @GetMapping("/health")
    public Mono<Map<String, Object>> health() {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "ok");
        response.put("service", serviceName);
        return Mono.just(response);
    }
}
