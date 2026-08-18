package com.hospital.workforce.scheduling;

import java.time.Instant;
import java.util.*;
import org.springframework.data.domain.*;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.query.Param;
interface ShiftRepository extends JpaRepository<ShiftAssignment, Long> {
    @Query("select s from ShiftAssignment s where s.employeeId = :employeeId and s.status <> com.hospital.workforce.scheduling.ShiftStatus.CANCELLED and s.startAt < :endAt and s.endAt > :startAt")
    List<ShiftAssignment> findOverlaps(@Param("employeeId") Long employeeId, @Param("startAt") Instant startAt, @Param("endAt") Instant endAt);
    Page<ShiftAssignment> findByDepartmentIdAndStartAtBetween(Long departmentId, Instant startAt, Instant endAt, Pageable pageable);
    List<ShiftAssignment> findByEmployeeIdAndStartAtBetweenAndStatus(Long employeeId, Instant startAt, Instant endAt, ShiftStatus status);
}
