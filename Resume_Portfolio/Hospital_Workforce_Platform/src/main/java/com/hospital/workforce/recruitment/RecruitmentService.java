package com.hospital.workforce.recruitment;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.NotFoundException;
import com.hospital.workforce.organization.DepartmentService;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;
@Service class RecruitmentService {
    private final JobRequisitionRepository requisitions; private final CandidateRepository candidates; private final DepartmentService departments; private final AuditService audit;
    RecruitmentService(JobRequisitionRepository requisitions, CandidateRepository candidates, DepartmentService departments, AuditService audit) { this.requisitions = requisitions; this.candidates = candidates; this.departments = departments; this.audit = audit; }
    @Transactional JobRequisition createRequisition(Long departmentId, String title, int headcount) { departments.get(departmentId); JobRequisition r = requisitions.save(new JobRequisition(departmentId, title, headcount)); audit.record("CREATE", "JobRequisition", r.getId(), "title=" + title); return r; }
    @Transactional Candidate apply(Long requisitionId, String fullName, String email) { requisitions.findById(requisitionId).orElseThrow(() -> new NotFoundException("Job requisition not found")); Candidate c = candidates.save(new Candidate(requisitionId, fullName, email)); audit.record("CREATE", "Candidate", c.getId(), "requisitionId=" + requisitionId); return c; }
}
