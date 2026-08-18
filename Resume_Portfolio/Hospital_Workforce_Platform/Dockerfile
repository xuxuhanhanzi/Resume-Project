# syntax=docker/dockerfile:1.7
FROM maven:3.9.9-eclipse-temurin-21 AS build
WORKDIR /workspace
COPY pom.xml .
COPY src src
RUN --mount=type=cache,target=/root/.m2/repository \
    attempt=1; \
    while [ "$attempt" -le 5 ]; do \
      mvn --batch-mode \
        -Dmaven.wagon.http.retryHandler.count=5 \
        -Dmaven.wagon.http.retryHandler.requestSentEnabled=true \
        -P!tests \
        -Dmaven.test.skip=true package && break; \
      echo "Maven package failed on attempt $attempt; retrying with the shared cache"; \
      attempt=$((attempt + 1)); \
    done; \
    [ "$attempt" -le 5 ] && \
    mvn --batch-mode \
      -Dmaven.wagon.http.retryHandler.count=5 \
      org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get \
      -Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.3 && \
    mkdir -p /opt/maven && \
    cp -a /root/.m2/repository /opt/maven/repository

FROM build AS test-runtime
ENTRYPOINT ["mvn", "--batch-mode", "-Dmaven.repo.local=/opt/maven/repository", "-Dmaven.wagon.http.retryHandler.count=5", "test"]

FROM eclipse-temurin:21-jre
WORKDIR /app
RUN useradd --system --create-home appuser
COPY --from=build /workspace/target/hospital-workforce-platform-0.1.0-SNAPSHOT.jar app.jar
USER appuser
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
