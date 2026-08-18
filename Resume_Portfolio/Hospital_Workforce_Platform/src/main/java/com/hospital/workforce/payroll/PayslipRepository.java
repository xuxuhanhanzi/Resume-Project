package com.hospital.workforce.payroll;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
interface PayslipRepository extends JpaRepository<Payslip, Long> { Optional<Payslip> findByEmployeeIdAndPayPeriod(Long employeeId, String payPeriod); }
