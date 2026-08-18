package com.hospital.workforce.organization;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/departments") class DepartmentController {
    private final DepartmentService service; DepartmentController(DepartmentService service) { this.service = service; }
    @PostMapping @PreAuthorize("hasRole('HR_ADMIN')") Department create(@Valid @RequestBody CreateDepartmentRequest request) { return service.create(request.code(), request.name()); }
    @GetMapping("/{id}") @PreAuthorize("hasAnyRole('HR_ADMIN','DEPARTMENT_MANAGER')") Department get(@PathVariable Long id) { return service.get(id); }
    record CreateDepartmentRequest(@NotBlank @Size(max = 32) String code, @NotBlank @Size(max = 120) String name) { }
}
