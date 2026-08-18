package com.hospital.workforce.training;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.time.Instant;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api") class TrainingController {
    private final TrainingService service; TrainingController(TrainingService service) { this.service = service; }
    @PostMapping("/training-courses") @PreAuthorize("hasRole('HR_ADMIN')") TrainingCourse create(@Valid @RequestBody CourseRequest request) { return service.createCourse(request.title(), request.startAt(), request.capacity()); }
    @PostMapping("/training-courses/{id}/enrollments") @PreAuthorize("hasRole('HR_ADMIN')") TrainingEnrollment enroll(@PathVariable Long id, @Valid @RequestBody EnrollRequest request) { return service.enroll(id, request.employeeId()); }
    record CourseRequest(@NotBlank String title, @NotNull Instant startAt, @Min(1) int capacity) { }
    record EnrollRequest(@NotNull Long employeeId) { }
}
