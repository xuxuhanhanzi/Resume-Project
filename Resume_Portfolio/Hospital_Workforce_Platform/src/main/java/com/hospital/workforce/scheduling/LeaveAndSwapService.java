package com.hospital.workforce.scheduling;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import com.hospital.workforce.workforce.*;
import jakarta.transaction.Transactional;
import java.time.Instant;
import org.springframework.stereotype.Service;
@Service class LeaveAndSwapService {
    private final LeaveRequestRepository leaves; private final ShiftSwapRequestRepository swaps; private final ShiftRepository shifts; private final WorkforceService workforce; private final AuditService audit;
    LeaveAndSwapService(LeaveRequestRepository leaves, ShiftSwapRequestRepository swaps, ShiftRepository shifts, WorkforceService workforce, AuditService audit) { this.leaves = leaves; this.swaps = swaps; this.shifts = shifts; this.workforce = workforce; this.audit = audit; }
    @Transactional LeaveRequest requestLeave(Long employeeId, Instant startAt, Instant endAt, String reason) { if (!startAt.isBefore(endAt)) throw new IllegalArgumentException("Leave start must be before end"); workforce.activeProfile(employeeId); LeaveRequest leave = leaves.save(new LeaveRequest(employeeId, startAt, endAt, reason)); audit.record("REQUEST", "LeaveRequest", leave.getId(), "employeeId=" + employeeId); return leave; }
    @Transactional LeaveRequest approveLeave(Long id) { LeaveRequest leave = leaves.findById(id).orElseThrow(() -> new NotFoundException("Leave request not found")); leave.approve(); audit.record("APPROVE", "LeaveRequest", id, ""); return leave; }
    @Transactional ShiftSwapRequest requestSwap(Long shiftId, Long targetEmployeeId) { ShiftAssignment shift = shifts.findById(shiftId).orElseThrow(() -> new NotFoundException("Shift not found")); if (shift.getStatus() != ShiftStatus.PUBLISHED) throw new DomainConflictException("Only a published shift can be swapped"); workforce.activeProfile(targetEmployeeId); ShiftSwapRequest swap = swaps.save(new ShiftSwapRequest(shiftId, targetEmployeeId)); audit.record("REQUEST", "ShiftSwapRequest", swap.getId(), "shiftId=" + shiftId); return swap; }
    @Transactional ShiftSwapRequest approveSwap(Long id) {
        ShiftSwapRequest swap = swaps.findById(id).orElseThrow(() -> new NotFoundException("Shift swap request not found")); ShiftAssignment shift = shifts.findById(swap.getShiftId()).orElseThrow(() -> new NotFoundException("Shift not found"));
        EmployeeProfile target = workforce.lockActiveEmployee(swap.getTargetEmployeeId());
        if (!shift.getRequiredQualification().isBlank() && !shift.getRequiredQualification().equalsIgnoreCase(target.qualification())) throw new DomainConflictException("Target employee qualification does not meet shift requirement");
        if (!shifts.findOverlaps(target.id(), shift.getStartAt(), shift.getEndAt()).isEmpty()) throw new DomainConflictException("Target employee has an overlapping shift");
        shift.reassignTo(target.id(), target.departmentId()); swap.approve(); audit.record("APPROVE", "ShiftSwapRequest", id, "targetEmployeeId=" + target.id()); return swap;
    }
}
