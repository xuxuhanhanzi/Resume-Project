package com.hospital.workforce.organization;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
interface DepartmentRepository extends JpaRepository<Department, Long> { Optional<Department> findByCode(String code); }
