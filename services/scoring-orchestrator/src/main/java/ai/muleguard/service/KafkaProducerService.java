package ai.muleguard.service;

import ai.muleguard.config.ScoringProperties;
import ai.muleguard.model.ScoringDecision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.concurrent.CompletableFuture;

/**
 * Publishes scored decisions to the muleguard.alerts Kafka topic.
 *
 * Only BLOCK and REVIEW decisions are published (APPROVE decisions generate
 * no alerts to avoid flooding the alert consumer with low-value messages).
 *
 * The partition key is the source_token (from_account hash) so that all
 * transactions from the same account are ordered within the same partition.
 * This ensures downstream alert consumers see a chronologically ordered
 * history of decisions per account, which is important for ring-ring pattern
 * detection in Sprint 8.
 *
 * Idempotent producer (configured in KafkaConfig) ensures that retries on
 * network failure don't produce duplicate alert records.
 */
@Service
public class KafkaProducerService {

    private static final Logger log = LoggerFactory.getLogger(KafkaProducerService.class);

    private final KafkaTemplate<String, ScoringDecision> kafkaTemplate;
    private final ScoringProperties props;

    public KafkaProducerService(KafkaTemplate<String, ScoringDecision> kafkaTemplate,
                                 ScoringProperties props) {
        this.kafkaTemplate = kafkaTemplate;
        this.props = props;
    }

    /**
     * Publish a BLOCK or REVIEW decision to muleguard.alerts.
     *
     * Returns Mono<Void> so the ScoringService pipeline can flatMap it
     * without blocking. Errors are caught by ScoringService.onErrorResume.
     *
     * @param decision The fused scoring decision to publish
     */
    public Mono<Void> publishAlert(ScoringDecision decision) {
        String topic = props.getKafka().getTopics().getAlerts();
        String key = decision.getSourceToken();  // partition by sender account

        CompletableFuture<SendResult<String, ScoringDecision>> future =
            kafkaTemplate.send(topic, key, decision).toCompletableFuture();

        return Mono.fromFuture(future)
            .doOnSuccess(result -> log.info(
                "Published alert txnId={} decision={} partition={} offset={}",
                decision.getTransactionId(), decision.getDecision(),
                result.getRecordMetadata().partition(),
                result.getRecordMetadata().offset()))
            .doOnError(ex -> log.error(
                "Failed to publish alert for txnId={}: {}", decision.getTransactionId(), ex.getMessage()))
            .then();
    }
}
