package com.hospital.workforce.common;

import java.time.Instant;

public record ApiError(Instant timestamp, int status, String code, String message, String traceId) {
    static ApiError of(int status, String code, String message) {
        return new ApiError(Instant.now(), status, code, message, null);
    }
}
