package com.hospital.workforce.workforce;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.math.BigDecimal;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/employees") class EmployeeController {
    private final WorkforceService service; EmployeeController(WorkforceService service) { this.service = service; }
    @PostMapping @PreAuthorize("hasRole('HR_ADMIN')") Employee create(@Valid @RequestBody CreateEmployeeRequest request) {
        return service.create(request.employeeNo(), request.fullName(), request.departmentId(), request.jobTitle(), request.qualification(), request.baseSalary());
    }
    @GetMapping("/{id}") @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") Employee get(@PathVariable Long id) { return service.get(id); }
    @GetMapping @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") Page<Employee> list(@RequestParam(required = false) Long departmentId, Pageable pageable) { return service.list(departmentId, pageable); }
    record CreateEmployeeRequest(@NotBlank @Size(max = 32) String employeeNo, @NotBlank String fullName, @NotNull Long departmentId, @NotBlank String jobTitle, String qualification, @NotNull @DecimalMin("0.00") BigDecimal baseSalary) { }
}
