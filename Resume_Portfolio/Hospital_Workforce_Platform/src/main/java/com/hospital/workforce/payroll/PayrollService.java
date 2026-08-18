package com.hospital.workforce.payroll;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import com.hospital.workforce.scheduling.ShiftService;
import com.hospital.workforce.workforce.*;
import jakarta.transaction.Transactional;
import java.math.*;
import java.time.YearMonth;
import org.springframework.stereotype.Service;

@Service public class PayrollService {
    private static final BigDecimal NIGHT_SHIFT_RATE = new BigDecimal("150.00");
    private final PayslipRepository payslips; private final WorkforceService workforce; private final ShiftService shifts; private final AuditService audit;
    PayrollService(PayslipRepository payslips, WorkforceService workforce, ShiftService shifts, AuditService audit) { this.payslips = payslips; this.workforce = workforce; this.shifts = shifts; this.audit = audit; }
    @Transactional public Payslip generate(Long employeeId, YearMonth period, BigDecimal performanceBonus, BigDecimal adjustment) {
        String payPeriod = period.toString(); if (payslips.findByEmployeeIdAndPayPeriod(employeeId, payPeriod).isPresent()) throw new DomainConflictException("Payslip already exists for this employee and period");
        EmployeeProfile employee = workforce.activeProfile(employeeId); int nights = Math.toIntExact(shifts.countPublishedNightShifts(employeeId, period));
        BigDecimal nightAllowance = NIGHT_SHIFT_RATE.multiply(BigDecimal.valueOf(nights));
        Payslip payslip = payslips.save(new Payslip(employeeId, payPeriod, employee.baseSalary(), nights, nightAllowance, money(performanceBonus), money(adjustment)));
        audit.record("GENERATE", "Payslip", payslip.getId(), "employeeId=" + employeeId + ",period=" + payPeriod); return payslip;
    }
    @Transactional public Payslip submit(Long id) { Payslip p = get(id); p.submit(); audit.record("SUBMIT", "Payslip", id, ""); return p; }
    @Transactional public Payslip approve(Long id) { Payslip p = get(id); p.approve(); audit.record("APPROVE", "Payslip", id, ""); return p; }
    @Transactional public Payslip markPaid(Long id) { Payslip p = get(id); p.markPaid(); audit.record("PAY", "Payslip", id, ""); return p; }
    public Payslip get(Long id) { return payslips.findById(id).orElseThrow(() -> new NotFoundException("Payslip not found")); }
    private BigDecimal money(BigDecimal value) { return (value == null ? BigDecimal.ZERO : value).setScale(2, RoundingMode.HALF_UP); }
}
