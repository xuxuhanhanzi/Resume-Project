package com.hospital.workforce.organization;

import jakarta.persistence.*;
import java.io.Serializable;
@Entity @Table(name = "department") public class Department implements Serializable {
    private static final long serialVersionUID = 1L;
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false, unique = true, length = 32) private String code;
    @Column(nullable = false, length = 120) private String name;
    @Column(nullable = false) private boolean active = true;
    @Version private Long version;
    protected Department() { } Department(String code, String name) { this.code = code; this.name = name; }
    public Long getId() { return id; } public String getCode() { return code; } public String getName() { return name; } public boolean isActive() { return active; }
}
