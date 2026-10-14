package ai.muleguard.config;

import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.core.*;
import org.springframework.kafka.listener.ContainerProperties;
import org.springframework.kafka.support.serializer.JsonDeserializer;
import org.springframework.kafka.support.serializer.JsonSerializer;

import java.util.HashMap;
import java.util.Map;

/**
 * Kafka consumer and producer configuration.
 *
 * Consumer (muleguard.transactions topic):
 *   - Deserializes JSON → ScoringRequest
 *   - Concurrency=3 (3 listener threads to handle burst traffic)
 *   - AckMode=MANUAL_IMMEDIATE: only commit offset after successful DB write
 *     (ensures no scoring decision is lost on crash between consume and persist)
 *
 * Producer (muleguard.alerts topic):
 *   - Serializes ScoringDecision → JSON
 *   - acks=all: every alert write must be confirmed by all Kafka replicas
 *   - Idempotent producer enabled to prevent duplicate alert messages on retry
 */
@Configuration
public class KafkaConfig {

    @Value("${spring.kafka.bootstrap-servers:localhost:9092}")
    private String bootstrapServers;

    // ── Consumer ─────────────────────────────────────────────────────────────

    @Bean
    public ConsumerFactory<String, ScoringRequest> consumerFactory() {
        JsonDeserializer<ScoringRequest> deserializer = new JsonDeserializer<>(ScoringRequest.class);
        deserializer.setRemoveTypeHeaders(false);
        deserializer.addTrustedPackages("ai.muleguard.model");
        deserializer.setUseTypeMapperForKey(true);

        Map<String, Object> props = new HashMap<>();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "scoring-orchestrator");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);  // manual ack
        props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, 100);
        props.put(ConsumerConfig.FETCH_MAX_WAIT_MS_CONFIG, 10);  // low latency

        return new DefaultKafkaConsumerFactory<>(props, new StringDeserializer(), deserializer);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, ScoringRequest> kafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, ScoringRequest> factory =
            new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        factory.setConcurrency(3);  // 3 parallel listener threads
        // MANUAL_IMMEDIATE: ack immediately when acknowledge() is called in the listener
        factory.getContainerProperties().setAckMode(ContainerProperties.AckMode.MANUAL_IMMEDIATE);
        return factory;
    }

    // ── Producer ─────────────────────────────────────────────────────────────

    @Bean
    public ProducerFactory<String, ScoringDecision> producerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        // Reliability: all replicas must acknowledge (acks=all)
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        // Idempotent producer: prevents duplicate alert messages on network retries
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(ProducerConfig.RETRIES_CONFIG, 3);
        props.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 1);
        return new DefaultKafkaProducerFactory<>(props);
    }

    @Bean
    public KafkaTemplate<String, ScoringDecision> kafkaTemplate() {
        return new KafkaTemplate<>(producerFactory());
    }
}
