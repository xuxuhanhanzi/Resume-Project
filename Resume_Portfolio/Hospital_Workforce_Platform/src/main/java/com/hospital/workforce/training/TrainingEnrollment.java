package com.hospital.workforce.training;

import jakarta.persistence.*;
@Entity @Table(name = "training_enrollment", uniqueConstraints = @UniqueConstraint(name = "uk_course_employee", columnNames = {"course_id", "employee_id"})) public class TrainingEnrollment {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long courseId;
    @Column(nullable = false) private Long employeeId;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private EnrollmentStatus status = EnrollmentStatus.ENROLLED;
    protected TrainingEnrollment() { } TrainingEnrollment(Long courseId, Long employeeId) { this.courseId = courseId; this.employeeId = employeeId; }
    public Long getId() { return id; } public Long getCourseId() { return courseId; } public Long getEmployeeId() { return employeeId; } public EnrollmentStatus getStatus() { return status; }
}
