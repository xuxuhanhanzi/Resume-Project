package com.hospital.workforce;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.modulith.Modulithic;

@Modulithic
@EnableCaching
@SpringBootApplication
public class HospitalWorkforceApplication {
    public static void main(String[] args) {
        SpringApplication.run(HospitalWorkforceApplication.class, args);
    }
}
