package ai.muleguard.config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * WebClient configuration for the three downstream model service calls.
 *
 * Each service gets its own WebClient bean with:
 *   - Named Reactor Netty connection pool (max 100 connections, 1000 pending)
 *   - Connect, read, and response timeouts matching the per-service SLA
 *   - Codec max-in-memory-size set to 2MB (SHAP payloads can be large)
 *
 * AI Engine (LGBM + AE + SHAP): 200ms response timeout
 * Graph Service (TGN):          25ms response timeout
 * Feature Service (Redis):      5ms response timeout
 *
 * These clients are injected into the service clients (AIEngineClient,
 * GraphServiceClient, FeatureServiceClient) which layer circuit breakers on top.
 */
@Configuration
public class WebClientConfig {

    private static final int MAX_IN_MEMORY_SIZE_BYTES = 2 * 1024 * 1024; // 2 MB

    private final ScoringProperties props;

    public WebClientConfig(ScoringProperties props) {
        this.props = props;
    }

    // ── AI Engine WebClient ──────────────────────────────────────────────────

    @Bean("aiEngineWebClient")
    public WebClient aiEngineWebClient() {
        long timeoutMs = props.getServices().getAiEngine().getTimeoutMs();
        return buildWebClient(
            props.getServices().getAiEngine().getUrl(),
            "ai-engine-pool",
            timeoutMs
        );
    }

    // ── Graph Service WebClient ──────────────────────────────────────────────

    @Bean("graphServiceWebClient")
    public WebClient graphServiceWebClient() {
        long timeoutMs = props.getServices().getGraphService().getTimeoutMs();
        return buildWebClient(
            props.getServices().getGraphService().getUrl(),
            "graph-service-pool",
            timeoutMs
        );
    }

    // ── Feature Service WebClient ────────────────────────────────────────────

    @Bean("featureServiceWebClient")
    public WebClient featureServiceWebClient() {
        long timeoutMs = props.getServices().getFeatureService().getTimeoutMs();
        return buildWebClient(
            props.getServices().getFeatureService().getUrl(),
            "feature-service-pool",
            timeoutMs
        );
    }

    // ── Shared builder ───────────────────────────────────────────────────────

    private WebClient buildWebClient(String baseUrl, String poolName, long responseTimeoutMs) {
        ConnectionProvider provider = ConnectionProvider.builder(poolName)
            .maxConnections(100)
            .maxIdleTime(Duration.ofSeconds(20))
            .maxLifeTime(Duration.ofSeconds(60))
            .pendingAcquireMaxCount(1000)
            .pendingAcquireTimeout(Duration.ofMillis(responseTimeoutMs))
            .build();

        HttpClient httpClient = HttpClient.create(provider)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, (int) Math.min(responseTimeoutMs, 5000))
            .responseTimeout(Duration.ofMillis(responseTimeoutMs))
            .doOnConnected(conn -> conn
                .addHandlerLast(new ReadTimeoutHandler(responseTimeoutMs, TimeUnit.MILLISECONDS))
                .addHandlerLast(new WriteTimeoutHandler(responseTimeoutMs, TimeUnit.MILLISECONDS))
            );

        ExchangeStrategies strategies = ExchangeStrategies.builder()
            .codecs(configurer -> configurer.defaultCodecs()
                .maxInMemorySize(MAX_IN_MEMORY_SIZE_BYTES))
            .build();

        return WebClient.builder()
            .baseUrl(baseUrl)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .exchangeStrategies(strategies)
            .defaultHeader("Content-Type", "application/json")
            .defaultHeader("Accept", "application/json")
            .build();
    }
}
