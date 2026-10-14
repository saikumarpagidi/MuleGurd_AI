package ai.muleguard.controller;

import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import ai.muleguard.service.ScoringService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

/**
 * REST endpoint for synchronous scoring.
 * 
 * Provides an alternative to the Kafka ingestion path for systems that
 * require an immediate blocking response (e.g., synchronous payment gateways).
 * The pipeline itself is still fully reactive and non-blocking under the hood.
 */
@RestController
@RequestMapping({"/api/v1", ""})
public class ScoringController {

    private static final Logger log = LoggerFactory.getLogger(ScoringController.class);

    private final ScoringService scoringService;

    public ScoringController(ScoringService scoringService) {
        this.scoringService = scoringService;
    }

    /**
     * POST /api/v1/score
     * 
     * Accepts a transaction payload and returns the authoritative fusion decision.
     */
    @PostMapping("/score")
    public Mono<ResponseEntity<ScoringDecision>> scoreTransaction(@RequestBody ScoringRequest request) {
        return scoringService.score(request)
            .map(ResponseEntity::ok)
            .defaultIfEmpty(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build());
    }

    /**
     * Fallback exception handler for unexpected pipeline failures.
     * 
     * In normal operation, the scoringService catches downstream errors and returns
     * a degraded fallback decision. If something completely breaks the pipeline itself,
     * this ensures we don't leak stack traces to the client.
     */
    @ExceptionHandler(Exception.class)
    public Mono<ResponseEntity<ScoringDecision>> handleException(Exception ex) {
        log.error("Unhandled exception in ScoringController: {}", ex.getMessage());
        
        // Return a completely degraded fallback so the client can still proceed safely
        ScoringRequest dummy = new ScoringRequest();
        dummy.setTransactionId("fallback-" + System.currentTimeMillis());
        return Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .body(ScoringDecision.degradedFallback(dummy)));
    }
}
