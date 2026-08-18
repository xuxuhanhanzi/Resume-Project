package com.hospital.workforce.scheduling;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
interface IdempotencyRepository extends JpaRepository<IdempotencyRecord, Long> { Optional<IdempotencyRecord> findByRequestKey(String requestKey); }
