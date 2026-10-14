package ai.muleguard;

import ai.muleguard.config.ScoringProperties;
import ai.muleguard.model.LGBMScore;
import ai.muleguard.model.ScoringDecision;
import ai.muleguard.model.ScoringRequest;
import ai.muleguard.model.TGNScore;
import ai.muleguard.service.RiskFusionService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RiskFusionServiceTest {

    private ScoringProperties props;
    private RiskFusionService service;
    private ScoringRequest request;

    @BeforeEach
    void setUp() {
        props = new ScoringProperties();
        service = new RiskFusionService(props);
        
        request = new ScoringRequest();
        request.setTransactionId("txn-123");
        request.setFromAccount("src");
        request.setToAccount("dst");
        request.setAmount(100.0);
    }

    @Test
    void testNormalFusion() {
        LGBMScore lgbm = new LGBMScore();
        lgbm.setLgbmScore(0.8);
        lgbm.setAeScore(0.6);
        
        TGNScore tgn = new TGNScore();
        tgn.setTgnScore(0.9);
        tgn.setRingRole("MULE");
        
        ScoringDecision decision = service.fuse(request, lgbm, tgn);
        
        // 0.5*0.9 (0.45) + 0.3*0.8 (0.24) + 0.15*0.6 (0.09) + 0.05*0.0 (0.0) = 0.78
        assertEquals(0.78, decision.getFinalRiskScore(), 0.001);
        assertEquals("BLOCK", decision.getDecision());
        assertTrue(decision.isRingFlag());
        assertEquals("MULE", decision.getRingRole());
        assertFalse(decision.isDegradedMode());
    }

    @Test
    void testDegradedFusion_TgnMissing() {
        LGBMScore lgbm = new LGBMScore();
        lgbm.setLgbmScore(0.8); // weight 0.3 -> effective 0.6
        lgbm.setAeScore(0.6);   // weight 0.15 -> effective 0.3
        
        TGNScore tgn = TGNScore.degraded(); // tgnScore is null
        
        ScoringDecision decision = service.fuse(request, lgbm, tgn);
        
        // Renormalized: 0.6 * 0.8 + 0.3 * 0.6 + 0.1 * 0.0 = 0.48 + 0.18 = 0.66
        assertEquals(0.66, decision.getFinalRiskScore(), 0.001);
        assertEquals("REVIEW", decision.getDecision());
        assertTrue(decision.isDegradedMode());
        assertNull(decision.getTgnScore());
    }
}
