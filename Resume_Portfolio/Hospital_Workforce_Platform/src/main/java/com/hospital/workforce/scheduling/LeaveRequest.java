package com.hospital.workforce.scheduling;

import com.hospital.workforce.common.DomainConflictException;
import jakarta.persistence.*;
import java.time.Instant;
@Entity @Table(name = "leave_request") public class LeaveRequest {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long employeeId;
    @Column(nullable = false) private Instant startAt;
    @Column(nullable = false) private Instant endAt;
    @Column(nullable = false, length = 500) private String reason;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private LeaveStatus status = LeaveStatus.PENDING;
    @Version private Long version;
    protected LeaveRequest() { } LeaveRequest(Long employeeId, Instant startAt, Instant endAt, String reason) { this.employeeId = employeeId; this.startAt = startAt; this.endAt = endAt; this.reason = reason; }
    public Long getId() { return id; } public Long getEmployeeId() { return employeeId; } public Instant getStartAt() { return startAt; } public Instant getEndAt() { return endAt; } public LeaveStatus getStatus() { return status; }
    public void approve() { if (status != LeaveStatus.PENDING) throw new DomainConflictException("Only a pending leave request can be approved"); status = LeaveStatus.APPROVED; }
}
