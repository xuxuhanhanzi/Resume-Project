package com.hospital.workforce.workforce;

import jakarta.persistence.*;
import java.math.BigDecimal;

@Entity @Table(name = "employee") public class Employee {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false, unique = true, length = 32) private String employeeNo;
    @Column(nullable = false, length = 120) private String fullName;
    @Column(nullable = false) private Long departmentId;
    @Column(nullable = false, length = 80) private String jobTitle;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private EmploymentStatus status = EmploymentStatus.ACTIVE;
    @Column(length = 120) private String qualification;
    @Column(nullable = false, precision = 15, scale = 2) private BigDecimal baseSalary;
    @Column(length = 120) private String keycloakSubject;
    @Version private Long version;
    protected Employee() { }
    Employee(String employeeNo, String fullName, Long departmentId, String jobTitle, String qualification, BigDecimal baseSalary) {
        this.employeeNo = employeeNo; this.fullName = fullName; this.departmentId = departmentId; this.jobTitle = jobTitle; this.qualification = qualification; this.baseSalary = baseSalary;
    }
    public Long getId() { return id; } public String getEmployeeNo() { return employeeNo; } public String getFullName() { return fullName; }
    public Long getDepartmentId() { return departmentId; } public String getJobTitle() { return jobTitle; } public EmploymentStatus getStatus() { return status; }
    public String getQualification() { return qualification; } public BigDecimal getBaseSalary() { return baseSalary; }
    public void markOnLeave() { status = EmploymentStatus.ON_LEAVE; } public void activate() { status = EmploymentStatus.ACTIVE; } public void terminate() { status = EmploymentStatus.TERMINATED; }
}
