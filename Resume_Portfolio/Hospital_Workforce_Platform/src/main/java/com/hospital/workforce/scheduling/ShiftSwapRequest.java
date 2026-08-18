package com.hospital.workforce.scheduling;

import com.hospital.workforce.common.DomainConflictException;
import jakarta.persistence.*;
@Entity @Table(name = "shift_swap_request") public class ShiftSwapRequest {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long shiftId;
    @Column(nullable = false) private Long targetEmployeeId;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private LeaveStatus status = LeaveStatus.PENDING;
    protected ShiftSwapRequest() { } ShiftSwapRequest(Long shiftId, Long targetEmployeeId) { this.shiftId = shiftId; this.targetEmployeeId = targetEmployeeId; }
    public Long getId() { return id; } public Long getShiftId() { return shiftId; } public Long getTargetEmployeeId() { return targetEmployeeId; } public LeaveStatus getStatus() { return status; }
    public void approve() { if (status != LeaveStatus.PENDING) throw new DomainConflictException("Only a pending swap request can be approved"); status = LeaveStatus.APPROVED; }
}
