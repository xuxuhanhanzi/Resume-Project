package com.hospital.workforce.notification;

import jakarta.persistence.*;
import java.time.Instant;
@Entity @Table(name = "notification") public class Notification {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long employeeId;
    @Column(nullable = false, length = 80) private String type;
    @Column(nullable = false, length = 1000) private String message;
    @Column(nullable = false) private Instant createdAt = Instant.now();
    @Column(nullable = false) private boolean readFlag = false;
    protected Notification() { } Notification(Long employeeId, String type, String message) { this.employeeId = employeeId; this.type = type; this.message = message; }
    public Long getId() { return id; } public Long getEmployeeId() { return employeeId; } public String getType() { return type; } public String getMessage() { return message; } public Instant getCreatedAt() { return createdAt; } public boolean isReadFlag() { return readFlag; }
}
