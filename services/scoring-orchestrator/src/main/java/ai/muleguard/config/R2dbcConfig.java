package ai.muleguard.config;

import io.r2dbc.spi.ConnectionFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.r2dbc.config.EnableR2dbcAuditing;
import org.springframework.data.r2dbc.repository.config.EnableR2dbcRepositories;
import org.springframework.r2dbc.connection.R2dbcTransactionManager;
import org.springframework.r2dbc.core.DatabaseClient;
import org.springframework.transaction.ReactiveTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * R2DBC (Reactive PostgreSQL) configuration.
 *
 * The ConnectionFactory is auto-configured from spring.r2dbc.* in application.yml
 * (URL, username, password, pool settings). This class adds:
 *   - Reactive transaction management
 *   - DatabaseClient bean for raw SQL (used by AuditLogService for RETURNING clause)
 *   - @EnableR2dbcRepositories scan
 *   - @EnableR2dbcAuditing for @CreatedDate / @LastModifiedDate support
 *
 * The connection pool is configured entirely in application.yml:
 *   spring.r2dbc.pool.initial-size=5
 *   spring.r2dbc.pool.max-size=20
 *   spring.r2dbc.pool.max-idle-time=30m
 */
@Configuration
@EnableR2dbcRepositories(basePackages = "ai.muleguard.repository")
@EnableR2dbcAuditing
@EnableTransactionManagement
public class R2dbcConfig {

    /**
     * Reactive transaction manager backed by R2DBC ConnectionFactory.
     * Enables @Transactional on reactive methods (Mono/Flux return types).
     */
    @Bean
    public ReactiveTransactionManager reactiveTransactionManager(ConnectionFactory connectionFactory) {
        return new R2dbcTransactionManager(connectionFactory);
    }

    /**
     * DatabaseClient for raw SQL execution (used in AuditLogService to execute
     * the RETURNING clause on INSERT and to verify immutability RULE enforcement).
     * Spring Data R2DBC repositories are the preferred approach; DatabaseClient
     * is used only for queries that can't be expressed via the repository API.
     */
    @Bean
    public DatabaseClient databaseClient(ConnectionFactory connectionFactory) {
        return DatabaseClient.create(connectionFactory);
    }
}
