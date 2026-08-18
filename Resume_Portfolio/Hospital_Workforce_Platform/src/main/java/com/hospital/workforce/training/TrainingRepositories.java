package com.hospital.workforce.training;

import org.springframework.data.jpa.repository.JpaRepository;
interface TrainingCourseRepository extends JpaRepository<TrainingCourse, Long> { }
interface TrainingEnrollmentRepository extends JpaRepository<TrainingEnrollment, Long> { long countByCourseIdAndStatus(Long courseId, EnrollmentStatus status); }
