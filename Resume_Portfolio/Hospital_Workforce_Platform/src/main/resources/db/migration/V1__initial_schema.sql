CREATE TABLE department (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(32) NOT NULL UNIQUE,
    name VARCHAR(120) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    version BIGINT
);
CREATE TABLE employee (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_no VARCHAR(32) NOT NULL UNIQUE,
    full_name VARCHAR(120) NOT NULL,
    department_id BIGINT NOT NULL,
    job_title VARCHAR(80) NOT NULL,
    status VARCHAR(24) NOT NULL,
    qualification VARCHAR(120),
    base_salary DECIMAL(15,2) NOT NULL,
    keycloak_subject VARCHAR(120),
    version BIGINT,
    CONSTRAINT fk_employee_department FOREIGN KEY (department_id) REFERENCES department(id),
    INDEX idx_employee_department (department_id)
);
CREATE TABLE shift_assignment (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_id BIGINT NOT NULL,
    department_id BIGINT NOT NULL,
    start_at TIMESTAMP(6) NOT NULL,
    end_at TIMESTAMP(6) NOT NULL,
    required_qualification VARCHAR(80) NOT NULL,
    status VARCHAR(24) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL,
    version BIGINT,
    CONSTRAINT fk_shift_employee FOREIGN KEY (employee_id) REFERENCES employee(id),
    CONSTRAINT fk_shift_department FOREIGN KEY (department_id) REFERENCES department(id),
    CONSTRAINT uk_shift_employee_start UNIQUE (employee_id, start_at),
    INDEX idx_shift_department_time (department_id, start_at),
    INDEX idx_shift_employee_time (employee_id, start_at)
);
CREATE TABLE idempotency_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_key VARCHAR(128) NOT NULL,
    resource_id BIGINT NOT NULL,
    created_at TIMESTAMP(6) NOT NULL,
    CONSTRAINT uk_idempotency_key UNIQUE (request_key)
);
CREATE TABLE payslip (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_id BIGINT NOT NULL,
    pay_period VARCHAR(7) NOT NULL,
    base_salary DECIMAL(15,2) NOT NULL,
    night_shift_allowance DECIMAL(15,2) NOT NULL,
    performance_bonus DECIMAL(15,2) NOT NULL,
    adjustment DECIMAL(15,2) NOT NULL,
    total DECIMAL(15,2) NOT NULL,
    night_shift_count INT NOT NULL,
    status VARCHAR(24) NOT NULL,
    generated_at TIMESTAMP(6) NOT NULL,
    version BIGINT,
    CONSTRAINT fk_payslip_employee FOREIGN KEY (employee_id) REFERENCES employee(id),
    CONSTRAINT uk_payslip_employee_period UNIQUE (employee_id, pay_period)
);
CREATE TABLE job_requisition (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    department_id BIGINT NOT NULL,
    title VARCHAR(120) NOT NULL,
    headcount INT NOT NULL,
    status VARCHAR(24) NOT NULL,
    CONSTRAINT fk_requisition_department FOREIGN KEY (department_id) REFERENCES department(id)
);
CREATE TABLE candidate (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    requisition_id BIGINT NOT NULL,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(180) NOT NULL,
    status VARCHAR(24) NOT NULL,
    CONSTRAINT fk_candidate_requisition FOREIGN KEY (requisition_id) REFERENCES job_requisition(id)
);
CREATE TABLE training_course (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(140) NOT NULL,
    start_at TIMESTAMP(6) NOT NULL,
    capacity INT NOT NULL
);
CREATE TABLE training_enrollment (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    course_id BIGINT NOT NULL,
    employee_id BIGINT NOT NULL,
    status VARCHAR(24) NOT NULL,
    CONSTRAINT fk_enrollment_course FOREIGN KEY (course_id) REFERENCES training_course(id),
    CONSTRAINT fk_enrollment_employee FOREIGN KEY (employee_id) REFERENCES employee(id),
    CONSTRAINT uk_course_employee UNIQUE (course_id, employee_id)
);
CREATE TABLE notification (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_id BIGINT NOT NULL,
    type VARCHAR(80) NOT NULL,
    message VARCHAR(1000) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL,
    read_flag BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT fk_notification_employee FOREIGN KEY (employee_id) REFERENCES employee(id)
);
CREATE TABLE leave_request (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_id BIGINT NOT NULL,
    start_at TIMESTAMP(6) NOT NULL,
    end_at TIMESTAMP(6) NOT NULL,
    reason VARCHAR(500) NOT NULL,
    status VARCHAR(24) NOT NULL,
    version BIGINT,
    CONSTRAINT fk_leave_employee FOREIGN KEY (employee_id) REFERENCES employee(id),
    INDEX idx_leave_employee_time (employee_id, start_at)
);
CREATE TABLE shift_swap_request (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    shift_id BIGINT NOT NULL,
    target_employee_id BIGINT NOT NULL,
    status VARCHAR(24) NOT NULL,
    CONSTRAINT fk_swap_shift FOREIGN KEY (shift_id) REFERENCES shift_assignment(id),
    CONSTRAINT fk_swap_target_employee FOREIGN KEY (target_employee_id) REFERENCES employee(id)
);
CREATE TABLE audit_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    actor VARCHAR(180) NOT NULL,
    action VARCHAR(80) NOT NULL,
    resource_type VARCHAR(80) NOT NULL,
    resource_id VARCHAR(120) NOT NULL,
    detail VARCHAR(2000),
    created_at TIMESTAMP(6) NOT NULL,
    INDEX idx_audit_created_at (created_at)
);
