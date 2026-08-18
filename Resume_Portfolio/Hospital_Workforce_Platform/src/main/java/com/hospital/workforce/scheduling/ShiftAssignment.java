package com.hospital.workforce.scheduling;

import com.hospital.workforce.common.DomainConflictException;
import jakarta.persistence.*;
import java.time.Instant;

@Entity @Table(name = "shift_assignment", uniqueConstraints = @UniqueConstraint(name = "uk_shift_employee_start", columnNames = {"employee_id", "start_at"}))
public class ShiftAssignment {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long employeeId;
    @Column(nullable = false) private Long departmentId;
    @Column(nullable = false) private Instant startAt;
    @Column(nullable = false) private Instant endAt;
    @Column(nullable = false, length = 80) private String requiredQualification;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private ShiftStatus status = ShiftStatus.DRAFT;
    @Column(nullable = false) private Instant createdAt = Instant.now();
    @Version private Long version;
    protected ShiftAssignment() { }
    ShiftAssignment(Long employeeId, Long departmentId, Instant startAt, Instant endAt, String requiredQualification) { this.employeeId = employeeId; this.departmentId = departmentId; this.startAt = startAt; this.endAt = endAt; this.requiredQualification = requiredQualification; }
    public Long getId() { return id; } public Long getEmployeeId() { return employeeId; } public Long getDepartmentId() { return departmentId; }
    public Instant getStartAt() { return startAt; } public Instant getEndAt() { return endAt; } public ShiftStatus getStatus() { return status; }
    public String getRequiredQualification() { return requiredQualification; } public Long getVersion() { return version; }
    public void publish() { if (status != ShiftStatus.DRAFT) throw new IllegalStateException("Only a draft shift can be published"); status = ShiftStatus.PUBLISHED; }
    public void cancel() { if (status == ShiftStatus.CANCELLED) throw new IllegalStateException("Shift is already cancelled"); status = ShiftStatus.CANCELLED; }
    public void reassignTo(Long newEmployeeId, Long newDepartmentId) { if (status != ShiftStatus.PUBLISHED) throw new DomainConflictException("Only a published shift can be reassigned"); employeeId = newEmployeeId; departmentId = newDepartmentId; }
}
