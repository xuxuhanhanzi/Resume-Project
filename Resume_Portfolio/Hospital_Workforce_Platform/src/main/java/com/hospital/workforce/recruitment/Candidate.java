package com.hospital.workforce.recruitment;

import jakarta.persistence.*;
@Entity @Table(name = "candidate") public class Candidate {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false) private Long requisitionId;
    @Column(nullable = false, length = 120) private String fullName;
    @Column(nullable = false, length = 180) private String email;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24) private CandidateStatus status = CandidateStatus.APPLIED;
    protected Candidate() { } Candidate(Long requisitionId, String fullName, String email) { this.requisitionId = requisitionId; this.fullName = fullName; this.email = email; }
    public Long getId() { return id; } public Long getRequisitionId() { return requisitionId; } public String getFullName() { return fullName; } public String getEmail() { return email; } public CandidateStatus getStatus() { return status; }
}
