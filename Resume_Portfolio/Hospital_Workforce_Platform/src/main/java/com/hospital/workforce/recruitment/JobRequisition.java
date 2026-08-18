package com.hospital.workforce.recruitment;

import jakarta.persistence.*;
@Entity @Table(name = "job_requisition") public class JobRequisition {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long departmentId;
    @Column(nullable = false, length = 120) private String title;
    @Column(nullable = false) private int headcount;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private RequisitionStatus status = RequisitionStatus.OPEN;
    protected JobRequisition() { } JobRequisition(Long departmentId, String title, int headcount) { this.departmentId = departmentId; this.title = title; this.headcount = headcount; }
    public Long getId() { return id; } public Long getDepartmentId() { return departmentId; } public String getTitle() { return title; } public int getHeadcount() { return headcount; } public RequisitionStatus getStatus() { return status; }
}
