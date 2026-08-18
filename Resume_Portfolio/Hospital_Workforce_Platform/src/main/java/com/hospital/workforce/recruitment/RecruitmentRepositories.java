package com.hospital.workforce.recruitment;

import org.springframework.data.jpa.repository.JpaRepository;
interface JobRequisitionRepository extends JpaRepository<JobRequisition, Long> { }
interface CandidateRepository extends JpaRepository<Candidate, Long> { }
