"""Build and verify the Stage 2 C++/CUDA ``silu_mul`` custom operator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast


def _inside_developer_environment(project_root: Path) -> int:
    python_scripts = str(Path(sys.executable).resolve().parent)
    path_entries = [python_scripts]
    vc_tools = os.environ.get("VCTOOLSINSTALLDIR")
    if vc_tools:
        path_entries.append(str(Path(vc_tools) / "bin" / "Hostx64" / "x64"))
    path_entries.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(path_entries)
    if shutil.which("cl.exe") is None:
        raise RuntimeError(
            "MSVC cl.exe is unavailable after VsDevCmd initialization; "
            "verify the Desktop development with C++ workload"
        )
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(project_root / "artifacts" / "torch_extensions")
    )
    import torch
    from torch.nn import functional as functional

    from forgellm.custom_ops.silu_mul import load_silu_mul_extension
    from forgellm.model.systems import benchmark_callable

    load_silu_mul_extension(with_cuda=torch.cuda.is_available(), verbose=False)
    results: dict[str, object] = {
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    for device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(device_name)
        gate = torch.randn(3, 5, device=device, dtype=torch.float64, requires_grad=True)
        value = torch.randn(3, 5, device=device, dtype=torch.float64, requires_grad=True)
        actual = torch.ops.forgellm_ops.silu_mul(gate, value)
        expected = functional.silu(gate) * value
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        gradcheck_passed = torch.autograd.gradcheck(torch.ops.forgellm_ops.silu_mul, (gate, value))
        opcheck_result = torch.library.opcheck(
            torch.ops.forgellm_ops.silu_mul, (gate.detach(), value.detach())
        )
        empty = torch.empty(0, 3, device=device)
        empty_output = torch.ops.forgellm_ops.silu_mul(empty, empty)
        if empty_output.shape != empty.shape:
            raise RuntimeError("silu_mul did not preserve the shape of an empty tensor")

        benchmark_gate = torch.randn(1_048_576, device=device)
        benchmark_value = torch.randn_like(benchmark_gate)

        def reference(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
            return functional.silu(gate_input) * value_input

        def extension(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
            return cast(
                torch.Tensor,
                torch.ops.forgellm_ops.silu_mul(gate_input, value_input),
            )

        with torch.inference_mode():
            reference_benchmark = benchmark_callable(
                reference,
                benchmark_gate,
                benchmark_value,
                warmup=5,
                repetitions=20,
            )
            extension_benchmark = benchmark_callable(
                extension,
                benchmark_gate,
                benchmark_value,
                warmup=5,
                repetitions=20,
            )
        results[device_name] = {
            "max_abs_error": float((actual - expected).abs().max()),
            "gradcheck": bool(gradcheck_passed),
            "opcheck": {name: str(result) for name, result in opcheck_result.items()},
            "empty_tensor": True,
            "benchmark": {
                "shape": [1_048_576],
                "dtype": "float32",
                "reference_ms": reference_benchmark.mean_milliseconds,
                "extension_ms": extension_benchmark.mean_milliseconds,
                "speedup": (
                    reference_benchmark.mean_milliseconds / extension_benchmark.mean_milliseconds
                ),
                "repetitions": extension_benchmark.repetitions,
            },
        }
    report = project_root / "artifacts" / "stage02" / "silu_mul_extension_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.resolve())
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))
    return 0


def _relaunch_with_msvc(project_root: Path) -> int:
    vsdev = Path(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    )
    if not vsdev.exists():
        print(f"Visual Studio developer command file not found: {vsdev}", file=sys.stderr)
        return 2
    script = Path(__file__).resolve()
    command = (
        f'call "{vsdev}" -arch=x64 '
        f'&& set "FORGELLM_VS_ENV=1" '
        f'&& "{sys.executable}" "{script}" --inside'
    )
    completed = subprocess.run(command, cwd=project_root, check=False, shell=True)
    return completed.returncode


def main() -> int:
    """Enter MSVC's environment once, then compile and verify the operator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.inside:
        return _inside_developer_environment(project_root)
    return _relaunch_with_msvc(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
