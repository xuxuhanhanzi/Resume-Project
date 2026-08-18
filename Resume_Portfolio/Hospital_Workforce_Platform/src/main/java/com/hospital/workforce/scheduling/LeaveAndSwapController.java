package com.hospital.workforce.scheduling;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.time.Instant;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api") class LeaveAndSwapController {
    private final LeaveAndSwapService service; LeaveAndSwapController(LeaveAndSwapService service) { this.service = service; }
    @PostMapping("/leave-requests") @PreAuthorize("hasAnyRole('EMPLOYEE','HR_ADMIN')") LeaveRequest requestLeave(@Valid @RequestBody LeaveRequestBody body) { return service.requestLeave(body.employeeId(), body.startAt(), body.endAt(), body.reason()); }
    @PostMapping("/leave-requests/{id}/approve") @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") LeaveRequest approveLeave(@PathVariable Long id) { return service.approveLeave(id); }
    @PostMapping("/shift-swaps") @PreAuthorize("hasAnyRole('EMPLOYEE','DEPARTMENT_MANAGER')") ShiftSwapRequest requestSwap(@Valid @RequestBody SwapRequestBody body) { return service.requestSwap(body.shiftId(), body.targetEmployeeId()); }
    @PostMapping("/shift-swaps/{id}/approve") @PreAuthorize("hasRole('DEPARTMENT_MANAGER')") ShiftSwapRequest approveSwap(@PathVariable Long id) { return service.approveSwap(id); }
    record LeaveRequestBody(@NotNull Long employeeId, @NotNull Instant startAt, @NotNull Instant endAt, @NotBlank @Size(max = 500) String reason) { }
    record SwapRequestBody(@NotNull Long shiftId, @NotNull Long targetEmployeeId) { }
}
