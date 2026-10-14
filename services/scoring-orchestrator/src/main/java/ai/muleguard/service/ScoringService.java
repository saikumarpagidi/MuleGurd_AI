package ai.muleguard.service;

import ai.muleguard.client.AIEngineClient;
import ai.muleguard.client.FeatureServiceClient;
import ai.muleguard.client.GraphServiceClient;
import ai.muleguard.model.LGBMScore;
import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import ai.muleguard.model.TGNScore;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Instant;

/**
 * Sprint 6 Core Scoring Pipeline — the heart of MuleGuard.AI.
 *
 * Executes three AI model calls IN PARALLEL using Reactor's Mono.zip(),
 * fuses the results via RiskFusionService, persists to PostgreSQL via
 * AuditLogService, and publishes alerts to Kafka via KafkaProducerService.
 *
 * Pipeline wall-clock = max(ai_engine_latency, graph_service_latency, feature_service_latency)
 *                     ≈ max(200ms, 25ms, 5ms)
 *                     = 200ms (LightGBM+AE+SHAP is the bottleneck)
 *
 * The P95 < 50ms gate in the k6 load test is achievable because the LightGBM
 * model inference itself is < 10ms; the 200ms timeout is a ceiling, not an
 * average. With Redis features (Sprint 4C), typical latency is 10-30ms.
 *
 * Event flow:
 *   KafkaConsumerService (Kafka topic) ─┐
 *   ScoringController (HTTP POST)      ─┼─→ ScoringService.score()
 *                                           │
 *                                           ├─ AIEngineClient.score()     ──┐
 *                                           ├─ GraphServiceClient.score() ──┼─ Mono.zip()
 *                                           ├─ FeatureServiceClient()     ──┘
 *                                           │        ↓
 *                                           ├─ RiskFusionService.fuse()
 *                                           ├─ AuditLogService.persist()
 *                                           └─ KafkaProducerService.publish() (if BLOCK/REVIEW)
 */
@Service
public class ScoringService {

    private static final Logger log = LoggerFactory.getLogger(ScoringService.class);

    private final AIEngineClient aiEngineClient;
    private final GraphServiceClient graphServiceClient;
    private final FeatureServiceClient featureServiceClient;
    private final RiskFusionService riskFusionService;
    private final AuditLogService auditLogService;
    private final KafkaProducerService kafkaProducerService;
    private final MeterRegistry meterRegistry;

    public ScoringService(
        AIEngineClient aiEngineClient,
        GraphServiceClient graphServiceClient,
        FeatureServiceClient featureServiceClient,
        RiskFusionService riskFusionService,
        AuditLogService auditLogService,
        KafkaProducerService kafkaProducerService,
        MeterRegistry meterRegistry
    ) {
        this.aiEngineClient = aiEngineClient;
        this.graphServiceClient = graphServiceClient;
        this.featureServiceClient = featureServiceClient;
        this.riskFusionService = riskFusionService;
        this.auditLogService = auditLogService;
        this.kafkaProducerService = kafkaProducerService;
        this.meterRegistry = meterRegistry;
    }

    /**
     * Execute the full parallel scoring pipeline for a single transaction.
     *
     * Returns Mono<ScoringDecision> which is lazy — nothing executes until subscribed.
     * The orchestrator's HTTP controller subscribes automatically via @ResponseBody.
     * The Kafka consumer subscribes explicitly via .subscribe() after receiving a message.
     *
     * Error behaviour:
     *   - Individual model failures are absorbed by circuit breakers (see client classes)
     *   - If ALL three clients fail simultaneously, the entire Mono errors, which is
     *     caught by the controller's @ExceptionHandler → returns 503 with a degraded decision
     *   - AuditLog write failure is logged but does NOT propagate back to the caller —
     *     a write failure should not cause the caller to retry (which would cause a duplicate)
     */
    public Mono<ScoringDecision> score(ScoringRequest request) {
        long startNanos = System.nanoTime();
        Timer.Sample timerSample = Timer.start(meterRegistry);

        return Mono.zip(
            aiEngineClient.score(request),
            graphServiceClient.graphScore(request),
            featureServiceClient.fetchFeatures(request.getFromAccount())
        )
        .map(tuple -> {
            LGBMScore lgbmScore = tuple.getT1();
            TGNScore  tgnScore  = tuple.getT2();
            // Feature map (T3) is used by ai-engine internally; the orchestrator
            // receives the already-scored result from ai-engine, so T3 is consumed
            // only to ensure feature pre-fetching ran in parallel. Future sprint:
            // pass features directly to the ai-engine in the request body.

            ScoringDecision decision = riskFusionService.fuse(request, lgbmScore, tgnScore);
            int processingMs = (int) ((System.nanoTime() - startNanos) / 1_000_000L);
            decision.setProcessingMs(processingMs);
            decision.setScoredAt(Instant.now());

            if (processingMs > 50) {
                log.warn("LATENCY BREACH txnId={} processingMs={} (P95 gate=50ms)",
                    request.getTransactionId(), processingMs);
            }

            return decision;
        })
        .flatMap(decision -> auditLogService.persist(decision)
            .thenReturn(decision)
            .onErrorResume(ex -> {
                // DB write failure: log and continue — do not propagate to caller
                log.error("AuditLog write failed for txnId={}: {}", request.getTransactionId(), ex.getMessage());
                return Mono.just(decision);
            })
        )
        .flatMap(decision -> {
            // Publish alert for BLOCK and REVIEW decisions
            if ("BLOCK".equals(decision.getDecision()) || "REVIEW".equals(decision.getDecision())) {
                return kafkaProducerService.publishAlert(decision)
                    .thenReturn(decision)
                    .onErrorResume(ex -> {
                        log.error("Kafka alert publish failed for txnId={}: {}", request.getTransactionId(), ex.getMessage());
                        return Mono.just(decision);
                    });
            }
            return Mono.just(decision);
        })
        .doOnSuccess(decision -> {
            timerSample.stop(meterRegistry.timer("scoring.pipeline.duration",
                "decision", decision.getDecision(),
                "degraded", String.valueOf(decision.isDegradedMode())));
        })
        .onErrorResume(ex -> {
            log.error("Complete pipeline failure for txnId={}: {}", request.getTransactionId(), ex.getMessage());
            return Mono.just(ScoringDecision.degradedFallback(request));
        });
    }
}
