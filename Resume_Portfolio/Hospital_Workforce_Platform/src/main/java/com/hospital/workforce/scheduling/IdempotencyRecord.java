package com.hospital.workforce.scheduling;

import jakarta.persistence.*;
import java.time.Instant;
@Entity @Table(name = "idempotency_record", uniqueConstraints = @UniqueConstraint(name = "uk_idempotency_key", columnNames = "request_key")) class IdempotencyRecord {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(name = "request_key", nullable = false, length = 128) private String requestKey;
    @Column(nullable = false) private Long resourceId;
    @Column(nullable = false) private Instant createdAt = Instant.now();
    protected IdempotencyRecord() { } IdempotencyRecord(String requestKey, Long resourceId) { this.requestKey = requestKey; this.resourceId = resourceId; }
    Long getResourceId() { return resourceId; }
}
