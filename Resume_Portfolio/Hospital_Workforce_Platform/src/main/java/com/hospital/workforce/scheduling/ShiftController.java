package com.hospital.workforce.scheduling;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.time.Instant;
import org.springframework.data.domain.*;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/shifts") class ShiftController {
    private final ShiftService service; ShiftController(ShiftService service) { this.service = service; }
    @PostMapping @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") ShiftAssignment create(@Valid @RequestBody CreateShiftRequest request, @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey) { return service.create(request.employeeId(), request.startAt(), request.endAt(), request.requiredQualification(), idempotencyKey); }
    @PostMapping("/{id}/publish") @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") ShiftAssignment publish(@PathVariable Long id) { return service.publish(id); }
    @GetMapping @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER','EMPLOYEE')") Page<ShiftAssignment> list(@RequestParam Long departmentId, @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant from, @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant to, Pageable pageable) { return service.list(departmentId, from, to, pageable); }
    record CreateShiftRequest(@NotNull Long employeeId, @NotNull Instant startAt, @NotNull Instant endAt, @Size(max = 80) String requiredQualification) { }
}
