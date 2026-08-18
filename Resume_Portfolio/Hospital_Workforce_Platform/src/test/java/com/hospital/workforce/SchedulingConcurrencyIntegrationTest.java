package com.hospital.workforce;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.jwt;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.hospital.workforce.audit.AuditService;
import com.hospital.workforce.common.DomainConflictException;
import com.hospital.workforce.organization.DepartmentService;
import com.hospital.workforce.scheduling.ShiftService;
import com.hospital.workforce.workforce.WorkforceService;
import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.MediaType;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.mysql.MySQLContainer;

@Testcontainers
@AutoConfigureMockMvc
@SpringBootTest(properties = {"spring.cache.type=simple", "spring.security.oauth2.resourceserver.jwt.jwk-set-uri=http://localhost:9999/keys"})
class SchedulingConcurrencyIntegrationTest {
    @Container static final MySQLContainer mysql = new MySQLContainer("mysql:8.4");
    @DynamicPropertySource static void database(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", mysql::getJdbcUrl); registry.add("spring.datasource.username", mysql::getUsername); registry.add("spring.datasource.password", mysql::getPassword);
    }
    @Autowired DepartmentService departments;
    @Autowired WorkforceService workforce;
    @Autowired ShiftService shifts;
    @Autowired AuditService audit;
    @Autowired MockMvc mockMvc;

    @Test void concurrentOverlappingShiftsAllowOnlyOneSuccess() throws Exception {
        Long departmentId = departments.create("ER-" + System.nanoTime(), "Emergency").getId();
        Long employeeId = workforce.create("E-" + System.nanoTime(), "Ava Chen", departmentId, "Nurse", "RN", new BigDecimal("5000.00")).getId();
        ExecutorService executor = Executors.newFixedThreadPool(2); CountDownLatch start = new CountDownLatch(1);
        try {
            Callable<Boolean> task = () -> { start.await(); try { shifts.create(employeeId, Instant.parse("2026-09-01T08:00:00Z"), Instant.parse("2026-09-01T16:00:00Z"), "RN", null); return true; } catch (DomainConflictException ex) { return false; } };
            Future<Boolean> first = executor.submit(task); Future<Boolean> second = executor.submit(task); start.countDown();
            assertEquals(1, (first.get() ? 1 : 0) + (second.get() ? 1 : 0));
        } finally {
            executor.shutdownNow();
        }
    }

    @Test void replayingIdempotencyKeyReturnsOriginalShift() {
        Long departmentId = departments.create("IDEMP-" + System.nanoTime(), "Idempotency").getId();
        Long employeeId = workforce.create("IDEMP-E-" + System.nanoTime(), "Ivy Chen", departmentId, "Nurse", "RN", new BigDecimal("5000.00")).getId();
        String key = "shift-" + System.nanoTime();
        var first = shifts.create(employeeId, Instant.parse("2026-10-01T08:00:00Z"), Instant.parse("2026-10-01T16:00:00Z"), "RN", key);
        var replay = shifts.create(employeeId, Instant.parse("2026-10-01T08:00:00Z"), Instant.parse("2026-10-01T16:00:00Z"), "RN", key);
        assertEquals(first.getId(), replay.getId());
    }

    @Test void auditTrailRecordsCreatedResources() {
        String code = "AUDIT-" + System.nanoTime();
        departments.create(code, "Audit Test");
        assertTrue(audit.list(PageRequest.of(0, 100)).stream()
                .anyMatch(log -> log.getResourceType().equals("Department") && log.getDetail().equals("code=" + code)));
    }

    @Test void cachedDepartmentSupportsRedisDefaultSerialization() {
        var department = departments.create("CACHE-" + System.nanoTime(), "Cache Test");
        assertDoesNotThrow(() -> {
            try (var bytes = new ByteArrayOutputStream(); var output = new ObjectOutputStream(bytes)) {
                output.writeObject(department);
            }
        });
    }

    @Test void departmentCreateEnforcesAuthenticationAndHrRole() throws Exception {
        String employeeBody = "{\"code\":\"SEC-EMP-" + System.nanoTime() + "\",\"name\":\"Forbidden\"}";
        String adminBody = "{\"code\":\"SEC-ADM-" + System.nanoTime() + "\",\"name\":\"Allowed\"}";

        mockMvc.perform(post("/api/departments").contentType(MediaType.APPLICATION_JSON).content(employeeBody))
                .andExpect(status().isUnauthorized());
        mockMvc.perform(post("/api/departments").with(jwt().authorities(new SimpleGrantedAuthority("ROLE_EMPLOYEE")))
                        .contentType(MediaType.APPLICATION_JSON).content(employeeBody))
                .andExpect(status().isForbidden());
        mockMvc.perform(post("/api/departments").with(jwt().authorities(new SimpleGrantedAuthority("ROLE_HR_ADMIN")))
                        .contentType(MediaType.APPLICATION_JSON).content(adminBody))
                .andExpect(status().isOk());
    }
}
