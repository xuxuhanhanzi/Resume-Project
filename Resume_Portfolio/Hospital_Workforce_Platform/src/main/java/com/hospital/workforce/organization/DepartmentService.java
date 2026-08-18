package com.hospital.workforce.organization;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import jakarta.transaction.Transactional;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
@Service public class DepartmentService {
    private final DepartmentRepository repository; private final AuditService audit;
    DepartmentService(DepartmentRepository repository, AuditService audit) { this.repository = repository; this.audit = audit; }
    @Transactional public Department create(String code, String name) {
        if (repository.findByCode(code).isPresent()) throw new DomainConflictException("Department code already exists");
        Department department = repository.save(new Department(code, name)); audit.record("CREATE", "Department", department.getId(), "code=" + code); return department;
    }
    @Cacheable(cacheNames = "departments", key = "#id") public Department get(Long id) { return repository.findById(id).orElseThrow(() -> new NotFoundException("Department not found")); }
}
