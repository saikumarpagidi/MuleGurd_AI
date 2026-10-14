package ai.muleguard;

import ai.muleguard.client.AIEngineClient;
import ai.muleguard.model.LGBMScore;
import ai.muleguard.model.ScoringRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@SpringBootTest
class GracefulDegradationTest {

    @Autowired
    private CircuitBreakerRegistry registry;
    
    @Autowired
    private ObjectMapper objectMapper;

    @MockBean(name = "aiEngineWebClient")
    private WebClient aiEngineWebClient;

    @Test
    void testAiEngineCircuitBreakerFallback() {
        CircuitBreaker cb = registry.circuitBreaker("aiEngine");
        
        // Ensure CB is closed for the test start
        cb.transitionToClosedState();
        
        // Mock a WebClient structure that throws an error when retrieved
        WebClient.RequestBodyUriSpec uriSpec = org.mockito.Mockito.mock(WebClient.RequestBodyUriSpec.class);
        WebClient.RequestBodySpec bodySpec = org.mockito.Mockito.mock(WebClient.RequestBodySpec.class);
        WebClient.RequestHeadersSpec headersSpec = org.mockito.Mockito.mock(WebClient.RequestHeadersSpec.class);
        WebClient.ResponseSpec responseSpec = org.mockito.Mockito.mock(WebClient.ResponseSpec.class);
        
        when(aiEngineWebClient.post()).thenReturn(uriSpec);
        when(uriSpec.uri(any(String.class))).thenReturn(bodySpec);
        when(bodySpec.bodyValue(any())).thenReturn(headersSpec);
        when(headersSpec.retrieve()).thenReturn(responseSpec);
        
        // Simulate connection refused
        when(responseSpec.bodyToMono(LGBMScore.class))
            .thenReturn(Mono.error(new RuntimeException("Connection refused")));

        AIEngineClient client = new AIEngineClient(aiEngineWebClient, cb, objectMapper);
        
        ScoringRequest req = new ScoringRequest();
        req.setTransactionId("txn-999");
        
        // Trigger enough failures to open the circuit breaker (threshold is 50% of 10 calls = 5)
        for (int i = 0; i < 6; i++) {
            client.score(req).block();
        }
        
        // Circuit breaker should now be OPEN
        org.junit.jupiter.api.Assertions.assertEquals(CircuitBreaker.State.OPEN, cb.getState());
        
        // Test the fallback mechanism
        StepVerifier.create(client.score(req))
            .expectNextMatches(score -> 
                score.isDegraded() && 
                score.getLgbmScore() == 0.5 && 
                "txn-999".equals(score.getTransactionId()))
            .verifyComplete();
    }
}
