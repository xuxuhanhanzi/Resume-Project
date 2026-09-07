#!/usr/bin/env python3
"""Cross-platform end-to-end acceptance test; never prints access tokens."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request(
    url: str,
    *,
    method: str = "GET",
    token: str = "",
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    actual_headers = dict(headers or {})
    if token:
        actual_headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        actual_headers["Content-Type"] = "application/json"
    operation = urllib.request.Request(
        url, data=data, method=method, headers=actual_headers
    )
    try:
        with urllib.request.urlopen(operation, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_healthy(url: str, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = request(url)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {url}")


def token(keycloak: str, username: str, password: str) -> str:
    payload = urllib.parse.urlencode(
        {
            "client_id": "hospital-api",
            "grant_type": "password",
            "username": username,
            "password": password,
        }
    ).encode()
    operation = urllib.request.Request(
        f"{keycloak}/realms/hospital/protocol/openid-connect/token",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(operation, timeout=15) as response:
        return json.load(response)["access_token"]


def expect(expected: int, response: tuple[int, bytes], label: str) -> bytes:
    if response[0] != expected:
        raise AssertionError(
            f"{label}: expected HTTP {expected}, got {response[0]}: {response[1][:300]!r}"
        )
    return response[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", default=os.getenv("HOSPITAL_API_URL", "http://localhost:8080")
    )
    parser.add_argument(
        "--keycloak",
        default=os.getenv("HOSPITAL_KEYCLOAK_URL", "http://localhost:8081"),
    )
    parser.add_argument(
        "--prometheus",
        default=os.getenv("HOSPITAL_PROMETHEUS_URL", "http://localhost:9090"),
    )
    parser.add_argument(
        "--grafana", default=os.getenv("HOSPITAL_GRAFANA_URL", "http://localhost:3000")
    )
    parser.add_argument(
        "--frontend",
        default=os.getenv("HOSPITAL_FRONTEND_URL", "http://localhost:5173"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    password = os.getenv("HOSPITAL_DEMO_PASSWORD", "Password1!")

    for endpoint in (
        f"{args.api}/actuator/health",
        f"{args.keycloak}/realms/hospital",
        f"{args.prometheus}/-/ready",
        f"{args.grafana}/api/health",
        f"{args.frontend}/",
    ):
        wait_healthy(endpoint)

    hr = token(args.keycloak, "hr.admin", password)
    employee = token(args.keycloak, "employee", password)
    auditor = token(args.keycloak, "auditor", password)
    stamp = int(time.time() * 1000)
    department_payload = {"code": f"AUTO-{stamp}", "name": "Automated Acceptance"}
    expect(
        401,
        request(f"{args.api}/api/departments", method="POST", body=department_payload),
        "anonymous create",
    )
    expect(
        403,
        request(
            f"{args.api}/api/departments",
            method="POST",
            token=employee,
            body=department_payload,
        ),
        "employee create",
    )
    department = json.loads(
        expect(
            200,
            request(
                f"{args.api}/api/departments",
                method="POST",
                token=hr,
                body=department_payload,
            ),
            "HR create",
        )
    )
    employee_body = {
        "employeeNo": f"AUTO-{stamp}",
        "fullName": "Automated Nurse",
        "departmentId": department["id"],
        "jobTitle": "Nurse",
        "qualification": "RN",
        "baseSalary": "5000.00",
    }
    created = json.loads(
        expect(
            200,
            request(
                f"{args.api}/api/employees", method="POST", token=hr, body=employee_body
            ),
            "employee",
        )
    )
    shift = {
        "employeeId": created["id"],
        "startAt": "2028-01-10T08:00:00Z",
        "endAt": "2028-01-10T16:00:00Z",
        "requiredQualification": "RN",
    }
    idem = {"Idempotency-Key": f"auto-{stamp}"}
    first = json.loads(
        expect(
            200,
            request(
                f"{args.api}/api/shifts",
                method="POST",
                token=hr,
                body=shift,
                headers=idem,
            ),
            "shift",
        )
    )
    replay = json.loads(
        expect(
            200,
            request(
                f"{args.api}/api/shifts",
                method="POST",
                token=hr,
                body=shift,
                headers=idem,
            ),
            "replay",
        )
    )
    if first["id"] != replay["id"]:
        raise AssertionError("idempotent replay returned another shift")
    expect(
        409,
        request(f"{args.api}/api/shifts", method="POST", token=hr, body=shift),
        "overlap",
    )
    audit = expect(
        200, request(f"{args.api}/api/audit-logs?size=100", token=auditor), "audit"
    ).decode()
    if (
        department_payload["code"] not in audit
        or employee_body["employeeNo"] not in audit
    ):
        raise AssertionError("audit trail is missing created records")

    targets = json.loads(
        expect(200, request(f"{args.prometheus}/api/v1/targets"), "prometheus targets")
    )
    backend = [
        item
        for item in targets["data"]["activeTargets"]
        if item.get("labels", {}).get("job") == "hospital-backend"
    ]
    if not backend or backend[0].get("health") != "up":
        raise AssertionError("Prometheus backend target is not up")
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed",
        "authorization_matrix": [401, 403, 200],
        "employee_id": created["id"],
        "idempotent_shift_id": first["id"],
        "overlap_status": 409,
        "audit_trail": "verified",
        "prometheus_target": "up",
        "access_tokens_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
