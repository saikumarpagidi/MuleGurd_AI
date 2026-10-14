package ai.muleguard.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.Instant;

/**
 * Wire-format for a transaction event published on muleguard.transactions Kafka topic
 * and accepted by POST /api/v1/score.
 *
 * Field names match the Python ai-engine's TransactionInput schema (snake_case)
 * to allow direct JSON pass-through between services without field mapping.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ScoringRequest {

    private String transactionId;
    private String fromAccount;
    private String toAccount;
    private double amount;
    private Instant timestamp;

    private String paymentFormat = "OTHER";
    private String currency = "USD";
    /** JA3 fingerprint hash for device-signal scoring (optional). */
    private String ja3Hash;

    // ── Constructors ────────────────────────────────────────────────────────

    public ScoringRequest() {}

    public ScoringRequest(String transactionId, String fromAccount, String toAccount,
                          double amount, Instant timestamp,
                          String paymentFormat, String currency) {
        this.transactionId = transactionId;
        this.fromAccount = fromAccount;
        this.toAccount = toAccount;
        this.amount = amount;
        this.timestamp = timestamp;
        this.paymentFormat = paymentFormat;
        this.currency = currency;
    }

    // ── Getters / Setters ───────────────────────────────────────────────────

    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }

    public String getFromAccount() { return fromAccount; }
    public void setFromAccount(String fromAccount) { this.fromAccount = fromAccount; }

    public String getToAccount() { return toAccount; }
    public void setToAccount(String toAccount) { this.toAccount = toAccount; }

    public double getAmount() { return amount; }
    public void setAmount(double amount) { this.amount = amount; }

    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }

    public String getPaymentFormat() { return paymentFormat; }
    public void setPaymentFormat(String paymentFormat) { this.paymentFormat = paymentFormat; }

    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }

    public String getJa3Hash() { return ja3Hash; }
    public void setJa3Hash(String ja3Hash) { this.ja3Hash = ja3Hash; }

    @Override
    public String toString() {
        return "ScoringRequest{txnId='" + transactionId + "', from='" + fromAccount +
               "', to='" + toAccount + "', amount=" + amount + "}";
    }
}
