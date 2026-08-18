package com.hospital.workforce.recruitment;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api") class RecruitmentController {
    private final RecruitmentService service; RecruitmentController(RecruitmentService service) { this.service = service; }
    @PostMapping("/job-requisitions") @PreAuthorize("hasRole('HR_ADMIN')") JobRequisition create(@Valid @RequestBody CreateRequisitionRequest request) { return service.createRequisition(request.departmentId(), request.title(), request.headcount()); }
    @PostMapping("/job-requisitions/{id}/candidates") @PreAuthorize("hasRole('HR_ADMIN')") Candidate apply(@PathVariable Long id, @Valid @RequestBody CandidateRequest request) { return service.apply(id, request.fullName(), request.email()); }
    record CreateRequisitionRequest(@NotNull Long departmentId, @NotBlank String title, @Min(1) int headcount) { }
    record CandidateRequest(@NotBlank String fullName, @Email @NotBlank String email) { }
}
