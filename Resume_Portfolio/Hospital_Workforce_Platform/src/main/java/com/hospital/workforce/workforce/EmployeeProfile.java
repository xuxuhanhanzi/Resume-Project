package com.hospital.workforce.workforce;

import java.math.BigDecimal;
public record EmployeeProfile(Long id, Long departmentId, EmploymentStatus status, String qualification, BigDecimal baseSalary) { }
