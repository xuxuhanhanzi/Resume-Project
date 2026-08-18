package com.hospital.workforce.audit;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
interface AuditLogRepository extends JpaRepository<AuditLog, Long> { Page<AuditLog> findAllByOrderByCreatedAtDesc(Pageable pageable); }
