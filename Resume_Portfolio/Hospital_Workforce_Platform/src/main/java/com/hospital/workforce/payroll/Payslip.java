package com.hospital.workforce.payroll;

import com.hospital.workforce.common.DomainConflictException;
import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity @Table(name = "payslip", uniqueConstraints = @UniqueConstraint(name = "uk_payslip_employee_period", columnNames = {"employee_id", "pay_period"})) public class Payslip {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long employeeId;
    @Column(name = "pay_period", nullable = false, length = 7) private String payPeriod;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal baseSalary;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal nightShiftAllowance;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal performanceBonus;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal adjustment;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal total;
    @Column(nullable = false) private int nightShiftCount;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private PayslipStatus status = PayslipStatus.DRAFT;
    @Column(nullable = false) private Instant generatedAt = Instant.now();
    @Version private Long version;
    protected Payslip() { }
    Payslip(Long employeeId, String payPeriod, BigDecimal baseSalary, int nightShiftCount, BigDecimal nightShiftAllowance, BigDecimal performanceBonus, BigDecimal adjustment) {
        this.employeeId = employeeId; this.payPeriod = payPeriod; this.baseSalary = baseSalary; this.nightShiftCount = nightShiftCount; this.nightShiftAllowance = nightShiftAllowance; this.performanceBonus = performanceBonus; this.adjustment = adjustment;
        this.total = baseSalary.add(nightShiftAllowance).add(performanceBonus).add(adjustment);
    }
    public Long getId() { return id; } public Long getEmployeeId() { return employeeId; } public String getPayPeriod() { return payPeriod; }
    public BigDecimal getBaseSalary() { return baseSalary; } public BigDecimal getNightShiftAllowance() { return nightShiftAllowance; }
    public BigDecimal getPerformanceBonus() { return performanceBonus; } public BigDecimal getAdjustment() { return adjustment; } public BigDecimal getTotal() { return total; }
    public int getNightShiftCount() { return nightShiftCount; } public PayslipStatus getStatus() { return status; }
    public void submit() { transition(PayslipStatus.DRAFT, PayslipStatus.SUBMITTED); }
    public void approve() { transition(PayslipStatus.SUBMITTED, PayslipStatus.APPROVED); }
    public void markPaid() { transition(PayslipStatus.APPROVED, PayslipStatus.PAID); }
    private void transition(PayslipStatus expected, PayslipStatus next) { if (status != expected) throw new DomainConflictException("Payslip must be " + expected + " before moving to " + next); status = next; }
}
