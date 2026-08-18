package com.hospital.workforce.payroll;

import static org.junit.jupiter.api.Assertions.*;
import java.math.BigDecimal;
import org.junit.jupiter.api.Test;

class PayslipTest {
    @Test void calculatesSnapshotAndEnforcesApprovalOrder() {
        Payslip payslip = new Payslip(1L, "2026-09", new BigDecimal("5000.00"), 2, new BigDecimal("300.00"), new BigDecimal("200.00"), new BigDecimal("-50.00"));
        assertEquals(new BigDecimal("5450.00"), payslip.getTotal());
        assertThrows(RuntimeException.class, payslip::approve);
        payslip.submit(); payslip.approve(); payslip.markPaid();
        assertEquals(PayslipStatus.PAID, payslip.getStatus());
    }
}
