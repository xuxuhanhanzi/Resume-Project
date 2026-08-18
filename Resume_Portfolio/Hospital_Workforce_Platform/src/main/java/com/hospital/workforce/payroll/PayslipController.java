package com.hospital.workforce.payroll;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.math.BigDecimal;
import java.time.YearMonth;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/payslips") class PayslipController {
    private final PayrollService service; PayslipController(PayrollService service) { this.service = service; }
    @PostMapping("/generate") @PreAuthorize("hasAnyRole('HR_ADMIN','FINANCE')") Payslip generate(@Valid @RequestBody GeneratePayslipRequest request) { return service.generate(request.employeeId(), request.parsedPeriod(), request.performanceBonus(), request.adjustment()); }
    @PostMapping("/{id}/submit") @PreAuthorize("hasRole('HR_ADMIN')") Payslip submit(@PathVariable Long id) { return service.submit(id); }
    @PostMapping("/{id}/approve") @PreAuthorize("hasRole('FINANCE')") Payslip approve(@PathVariable Long id) { return service.approve(id); }
    @PostMapping("/{id}/pay") @PreAuthorize("hasRole('FINANCE')") Payslip pay(@PathVariable Long id) { return service.markPaid(id); }
    @GetMapping("/{id}") @PreAuthorize("hasAnyRole('HR_ADMIN','FINANCE','EMPLOYEE')") Payslip get(@PathVariable Long id) { return service.get(id); }
    record GeneratePayslipRequest(@NotNull Long employeeId, @NotNull String period, @DecimalMin(value = "0.00", inclusive = false) BigDecimal performanceBonus, BigDecimal adjustment) {
        YearMonth parsedPeriod() { return YearMonth.parse(period); }
    }
}
