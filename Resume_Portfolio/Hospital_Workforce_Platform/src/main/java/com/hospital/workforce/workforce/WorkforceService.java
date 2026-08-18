package com.hospital.workforce.workforce;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import com.hospital.workforce.organization.DepartmentService;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

@Service public class WorkforceService {
    private final EmployeeRepository employees; private final DepartmentService departments; private final AuditService audit;
    WorkforceService(EmployeeRepository employees, DepartmentService departments, AuditService audit) { this.employees = employees; this.departments = departments; this.audit = audit; }
    @Transactional public Employee create(String employeeNo, String fullName, Long departmentId, String jobTitle, String qualification, BigDecimal baseSalary) {
        departments.get(departmentId);
        if (employees.findByEmployeeNo(employeeNo).isPresent()) throw new DomainConflictException("Employee number already exists");
        Employee employee = employees.save(new Employee(employeeNo, fullName, departmentId, jobTitle, qualification, baseSalary));
        audit.record("CREATE", "Employee", employee.getId(), "employeeNo=" + employeeNo); return employee;
    }
    public Employee get(Long id) { return employees.findById(id).orElseThrow(() -> new NotFoundException("Employee not found")); }
    public Page<Employee> list(Long departmentId, Pageable pageable) { return departmentId == null ? employees.findAll(pageable) : employees.findByDepartmentId(departmentId, pageable); }
    @Transactional public EmployeeProfile lockActiveEmployee(Long id) {
        Employee employee = employees.lockById(id).orElseThrow(() -> new NotFoundException("Employee not found"));
        if (employee.getStatus() != EmploymentStatus.ACTIVE) throw new DomainConflictException("Employee is not active and cannot be scheduled");
        return profile(employee);
    }
    public EmployeeProfile activeProfile(Long id) { Employee employee = get(id); if (employee.getStatus() != EmploymentStatus.ACTIVE) throw new DomainConflictException("Employee is not active"); return profile(employee); }
    private EmployeeProfile profile(Employee e) { return new EmployeeProfile(e.getId(), e.getDepartmentId(), e.getStatus(), e.getQualification(), e.getBaseSalary()); }
}
