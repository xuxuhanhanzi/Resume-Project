package com.hospital.workforce.audit;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
@Service public class AuditService {
    private final AuditLogRepository repository;
    AuditService(AuditLogRepository repository) { this.repository = repository; }
    public void record(String action, String type, Object id, String detail) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        repository.save(new AuditLog(auth == null ? "system" : auth.getName(), action, type, String.valueOf(id), detail));
    }
    public Page<AuditLog> list(Pageable pageable) { return repository.findAllByOrderByCreatedAtDesc(pageable); }
}
