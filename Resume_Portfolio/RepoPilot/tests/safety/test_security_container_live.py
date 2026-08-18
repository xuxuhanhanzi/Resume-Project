from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from repopilot.runtime.runner import DockerSandboxConfig, DockerSandboxRunner

PROBE = r"""
import json
import os
import socket
from pathlib import Path

status = {}
for line in Path('/proc/self/status').read_text().splitlines():
    if ':' in line:
        key, value = line.split(':', 1)
        status[key] = value.strip()

network_connected = False
try:
    with socket.create_connection(('1.1.1.1', 53), timeout=0.5):
        network_connected = True
except OSError:
    pass

workspace_probe = Path('/workspace/container-write.txt')
workspace_probe.write_text('sandbox-write-ok\n', encoding='utf-8')
result = {
    'uid': os.getuid(),
    'gid': os.getgid(),
    'root_read_only': bool(os.statvfs('/').f_flag & os.ST_RDONLY),
    'no_new_privileges': status.get('NoNewPrivs'),
    'effective_capabilities': status.get('CapEff'),
    'network_connected': network_connected,
    'docker_socket_present': Path('/var/run/docker.sock').exists(),
    'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    'pids_max': Path('/sys/fs/cgroup/pids.max').read_text().strip(),
    'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    'workspace_write': workspace_probe.exists(),
}
print(json.dumps(result, sort_keys=True))
"""


@pytest.mark.safety
@pytest.mark.skipif(
    os.environ.get("REPOPILOT_RUN_DOCKER_SECURITY") != "1",
    reason="set REPOPILOT_RUN_DOCKER_SECURITY=1 for the live Docker security gate",
)
def test_live_container_enforces_security_profile(tmp_path: Path) -> None:
    image = os.environ.get("REPOPILOT_SANDBOX_IMAGE", "repopilot-sandbox:20260814")
    (tmp_path / "public.txt").write_text("public fixture\n", encoding="utf-8")
    result = asyncio.run(
        DockerSandboxRunner(DockerSandboxConfig(image)).run(
            ("python", "-c", PROBE), cwd=tmp_path, timeout_seconds=10
        )
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "cpu_max": "100000 100000",
        "docker_socket_present": False,
        "effective_capabilities": "0000000000000000",
        "gid": 65534,
        "memory_max": "1073741824",
        "network_connected": False,
        "no_new_privileges": "1",
        "pids_max": "64",
        "root_read_only": True,
        "uid": 65534,
        "workspace_write": True,
    }
