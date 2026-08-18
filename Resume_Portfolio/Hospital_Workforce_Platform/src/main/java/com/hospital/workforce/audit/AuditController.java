package com.hospital.workforce.audit;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/audit-logs") class AuditController {
    private final AuditService service; AuditController(AuditService service) { this.service = service; }
    @GetMapping @PreAuthorize("hasRole('AUDITOR')") Page<AuditLog> list(Pageable pageable) { return service.list(pageable); }
}
