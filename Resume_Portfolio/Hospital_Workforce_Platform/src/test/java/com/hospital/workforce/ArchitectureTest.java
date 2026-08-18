package com.hospital.workforce;

import org.junit.jupiter.api.Test;
import org.springframework.modulith.core.ApplicationModules;

class ArchitectureTest {
    @Test void verifiesModuleBoundaries() { ApplicationModules.of(HospitalWorkforceApplication.class).verify(); }
}
