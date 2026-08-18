package com.hospital.workforce.training;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.*;
import com.hospital.workforce.workforce.WorkforceService;
import jakarta.transaction.Transactional;
import java.time.Instant;
import org.springframework.stereotype.Service;
@Service class TrainingService {
    private final TrainingCourseRepository courses; private final TrainingEnrollmentRepository enrollments; private final WorkforceService workforce; private final AuditService audit;
    TrainingService(TrainingCourseRepository courses, TrainingEnrollmentRepository enrollments, WorkforceService workforce, AuditService audit) { this.courses = courses; this.enrollments = enrollments; this.workforce = workforce; this.audit = audit; }
    @Transactional TrainingCourse createCourse(String title, Instant startAt, int capacity) { TrainingCourse c = courses.save(new TrainingCourse(title, startAt, capacity)); audit.record("CREATE", "TrainingCourse", c.getId(), "title=" + title); return c; }
    @Transactional TrainingEnrollment enroll(Long courseId, Long employeeId) { TrainingCourse c = courses.findById(courseId).orElseThrow(() -> new NotFoundException("Training course not found")); workforce.activeProfile(employeeId); if (enrollments.countByCourseIdAndStatus(courseId, EnrollmentStatus.ENROLLED) >= c.getCapacity()) throw new DomainConflictException("Training course is full"); TrainingEnrollment e = enrollments.save(new TrainingEnrollment(courseId, employeeId)); audit.record("ENROLL", "TrainingCourse", courseId, "employeeId=" + employeeId); return e; }
}
