package com.hospital.workforce.audit;

import jakarta.persistence.*;
import java.time.Instant;

@Entity @Table(name = "audit_log")
public class AuditLog {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private String actor;
    @Column(nullable = false) private String action;
    @Column(nullable = false) private String resourceType;
    @Column(nullable = false) private String resourceId;
    @Column(length = 2000) private String detail;
    @Column(nullable = false) private Instant createdAt = Instant.now();
    protected AuditLog() { }
    AuditLog(String actor, String action, String resourceType, String resourceId, String detail) { this.actor = actor; this.action = action; this.resourceType = resourceType; this.resourceId = resourceId; this.detail = detail; }
    public Long getId() { return id; } public String getActor() { return actor; } public String getAction() { return action; }
    public String getResourceType() { return resourceType; } public String getResourceId() { return resourceId; }
    public String getDetail() { return detail; } public Instant getCreatedAt() { return createdAt; }
}
