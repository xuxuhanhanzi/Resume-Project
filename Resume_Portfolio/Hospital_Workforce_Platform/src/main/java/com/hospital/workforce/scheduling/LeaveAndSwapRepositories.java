package com.hospital.workforce.scheduling;

import java.time.Instant;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.query.Param;
interface LeaveRequestRepository extends JpaRepository<LeaveRequest, Long> {
    @Query("select count(l) > 0 from LeaveRequest l where l.employeeId = :employeeId and l.status = com.hospital.workforce.scheduling.LeaveStatus.APPROVED and l.startAt < :endAt and l.endAt > :startAt")
    boolean existsApprovedOverlap(@Param("employeeId") Long employeeId, @Param("startAt") Instant startAt, @Param("endAt") Instant endAt);
}
interface ShiftSwapRequestRepository extends JpaRepository<ShiftSwapRequest, Long> { }
