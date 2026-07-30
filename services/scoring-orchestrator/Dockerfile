# Scoring Orchestrator — Multi-stage Docker build
# Stage 1: Compile with Maven
# Stage 2: Minimal JRE runtime image

FROM maven:3.9-eclipse-temurin-21-alpine AS build

WORKDIR /build

# Copy POM first — Docker caches this layer when pom.xml is unchanged,
# avoiding redundant dependency downloads on code-only changes.
COPY pom.xml .
RUN mvn dependency:go-offline -B -q

COPY src ./src
RUN mvn package -DskipTests -B -q


FROM eclipse-temurin:21-jre-alpine AS runtime

WORKDIR /app

# Run as a non-root user
RUN addgroup -S muleguard && adduser -S muleguard -G muleguard
COPY --from=build /build/target/*.jar app.jar
RUN chown muleguard:muleguard app.jar
USER muleguard

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=10s --retries=10 --start-period=90s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/api/v1/health || exit 1

ENTRYPOINT ["java", \
    "-XX:+UseContainerSupport", \
    "-XX:MaxRAMPercentage=75.0", \
    "-Djava.security.egd=file:/dev/./urandom", \
    "-jar", "app.jar"]
