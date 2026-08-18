package com.hospital.workforce.workforce;

import java.util.Optional;
import jakarta.persistence.LockModeType;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.query.Param;
interface EmployeeRepository extends JpaRepository<Employee, Long> {
    Optional<Employee> findByEmployeeNo(String employeeNo);
    Page<Employee> findByDepartmentId(Long departmentId, Pageable pageable);
    @Lock(LockModeType.PESSIMISTIC_WRITE) @Query("select e from Employee e where e.id = :id") Optional<Employee> lockById(@Param("id") Long id);
}
