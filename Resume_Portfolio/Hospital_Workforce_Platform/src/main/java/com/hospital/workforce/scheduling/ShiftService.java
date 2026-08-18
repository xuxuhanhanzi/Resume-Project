package com.hospital.workforce.scheduling;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import com.hospital.workforce.workforce.*;
import jakarta.transaction.Transactional;
import java.time.*;
import java.util.List;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;

@Service public class ShiftService {
    private final ShiftRepository shifts; private final IdempotencyRepository idempotency; private final LeaveRequestRepository leaves; private final WorkforceService workforce; private final AuditService audit; private final ApplicationEventPublisher events;
    ShiftService(ShiftRepository shifts, IdempotencyRepository idempotency, LeaveRequestRepository leaves, WorkforceService workforce, AuditService audit, ApplicationEventPublisher events) { this.shifts = shifts; this.idempotency = idempotency; this.leaves = leaves; this.workforce = workforce; this.audit = audit; this.events = events; }
    @Transactional public ShiftAssignment create(Long employeeId, Instant startAt, Instant endAt, String requiredQualification, String idempotencyKey) {
        if (idempotencyKey != null && !idempotencyKey.isBlank()) {
            var previous = idempotency.findByRequestKey(idempotencyKey);
            if (previous.isPresent()) return get(previous.get().getResourceId());
        }
        if (!startAt.isBefore(endAt)) throw new IllegalArgumentException("Shift start must be before end");
        EmployeeProfile employee = workforce.lockActiveEmployee(employeeId);
        if (leaves.existsApprovedOverlap(employeeId, startAt, endAt)) throw new DomainConflictException("Employee has approved leave during this shift");
        if (requiredQualification != null && !requiredQualification.isBlank() && !requiredQualification.equalsIgnoreCase(employee.qualification())) throw new DomainConflictException("Employee qualification does not meet shift requirement");
        if (!shifts.findOverlaps(employeeId, startAt, endAt).isEmpty()) throw new DomainConflictException("Employee already has an overlapping shift");
        ShiftAssignment shift = shifts.saveAndFlush(new ShiftAssignment(employeeId, employee.departmentId(), startAt, endAt, requiredQualification == null ? "" : requiredQualification));
        if (idempotencyKey != null && !idempotencyKey.isBlank()) idempotency.save(new IdempotencyRecord(idempotencyKey, shift.getId()));
        audit.record("CREATE", "ShiftAssignment", shift.getId(), "employeeId=" + employeeId); return shift;
    }
    @Transactional public ShiftAssignment publish(Long id) {
        ShiftAssignment shift = get(id); shift.publish(); audit.record("PUBLISH", "ShiftAssignment", id, "employeeId=" + shift.getEmployeeId()); events.publishEvent(new ShiftPublishedEvent(id, shift.getEmployeeId())); return shift;
    }
    public ShiftAssignment get(Long id) { return shifts.findById(id).orElseThrow(() -> new NotFoundException("Shift not found")); }
    public Page<ShiftAssignment> list(Long departmentId, Instant from, Instant to, Pageable pageable) { return shifts.findByDepartmentIdAndStartAtBetween(departmentId, from, to, pageable); }
    public long countPublishedNightShifts(Long employeeId, YearMonth period) {
        Instant from = period.atDay(1).atStartOfDay(ZoneOffset.UTC).toInstant(); Instant to = period.plusMonths(1).atDay(1).atStartOfDay(ZoneOffset.UTC).toInstant();
        return shifts.findByEmployeeIdAndStartAtBetweenAndStatus(employeeId, from, to, ShiftStatus.PUBLISHED).stream().filter(s -> LocalDateTime.ofInstant(s.getStartAt(), ZoneOffset.UTC).getHour() >= 18 || LocalDateTime.ofInstant(s.getStartAt(), ZoneOffset.UTC).getHour() < 6).count();
    }
}
