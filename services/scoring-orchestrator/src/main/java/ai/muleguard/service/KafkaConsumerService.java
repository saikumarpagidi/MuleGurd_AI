package ai.muleguard.service;

import ai.muleguard.model.ScoringRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;

/**
 * Consumes transaction events from the muleguard.transactions Kafka topic.
 *
 * This is the primary real-time ingestion path for Sprint 6.
 * The listener uses MANUAL_IMMEDIATE ack mode (configured in KafkaConfig),
 * meaning the Kafka offset is only committed AFTER the full scoring pipeline
 * and database write succeed. This guarantees at-least-once processing semantics:
 * if the orchestrator crashes mid-pipeline, the transaction will be redelivered.
 */
@Service
public class KafkaConsumerService {

    private static final Logger log = LoggerFactory.getLogger(KafkaConsumerService.class);

    private final ScoringService scoringService;

    public KafkaConsumerService(ScoringService scoringService) {
        this.scoringService = scoringService;
    }

    /**
     * Listener for the transactions topic.
     *
     * Subscribes to the reactive scoring pipeline. The pipeline returns Mono<ScoringDecision>,
     * which we explicitly subscribe to here. The acknowledgement is only triggered
     * in the doOnSuccess callback of the Mono, ensuring DB persistence is complete.
     */
    @KafkaListener(
        topics = "${muleguard.kafka.topics.transactions}",
        groupId = "scoring-orchestrator",
        containerFactory = "kafkaListenerContainerFactory"
    )
    public void consumeTransaction(@Payload ScoringRequest request, Acknowledgment ack) {
        log.debug("Received transaction from Kafka: {}", request.getTransactionId());

        scoringService.score(request)
            .doOnSuccess(decision -> {
                log.debug("Successfully processed Kafka txnId={}", request.getTransactionId());
                ack.acknowledge();  // Commit offset
            })
            .doOnError(ex -> {
                // In a production system, we might route this to a Dead Letter Queue (DLQ)
                // after N retries. For now, we log it and do NOT acknowledge, causing a
                // redelivery on the next poll.
                log.error("Failed to process Kafka txnId={}: {}", request.getTransactionId(), ex.getMessage());
            })
            .subscribe(); // Trigger the reactive pipeline
    }
}
