"""Python reference, build loader, FakeTensor, and Autograd for ``silu_mul``."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.cpp_extension import load

_DISPATCH_REGISTERED = False


def _prepare_extension_tool_path() -> None:
    """Expose venv Ninja and an installed MSVC compiler to PyTorch's JIT loader."""
    entries = [str(Path(sys.executable).resolve().parent)]
    if os.name == "nt" and shutil.which("cl.exe") is None:
        vc_tools = os.environ.get("VCTOOLSINSTALLDIR")
        if vc_tools:
            candidates = [Path(vc_tools)]
        else:
            visual_studio = Path(r"C:\Program Files\Microsoft Visual Studio\2022")
            candidates = [
                version
                for edition in ("Community", "Professional", "Enterprise", "BuildTools")
                for version in (visual_studio / edition / "VC" / "Tools" / "MSVC").glob("*")
            ]
        for candidate in sorted(candidates, reverse=True):
            compiler_bin = candidate / "bin" / "Hostx64" / "x64"
            if (compiler_bin / "cl.exe").exists():
                entries.append(str(compiler_bin))
                break
    entries.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(entries)


def silu_mul_reference(gate: Tensor, value: Tensor) -> Tensor:
    """Reference equation ``silu(gate) * value``."""
    if gate.shape != value.shape:
        raise ValueError("gate and value must have identical shapes")
    if gate.dtype != value.dtype or gate.device != value.device:
        raise ValueError("gate and value must share dtype and device")
    return F.silu(gate) * value


def _register_python_dispatch() -> None:
    global _DISPATCH_REGISTERED
    if _DISPATCH_REGISTERED:
        return

    @torch.library.register_fake("forgellm_ops::silu_mul")  # type: ignore[untyped-decorator]
    def _fake(gate: Tensor, value: Tensor) -> Tensor:
        torch._check(gate.shape == value.shape)  # type: ignore[no-untyped-call]
        torch._check(gate.dtype == value.dtype)  # type: ignore[no-untyped-call]
        torch._check(gate.device == value.device)  # type: ignore[no-untyped-call]
        return torch.empty_like(gate)

    def _setup_context(ctx: Any, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
        del output
        ctx.save_for_backward(*inputs)

    def _backward(context: Any, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        gate, value = context.saved_tensors
        sigmoid = torch.sigmoid(gate)
        silu_derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        return grad_output * value * silu_derivative, grad_output * F.silu(gate)

    torch.library.register_autograd(
        "forgellm_ops::silu_mul",
        _backward,
        setup_context=_setup_context,
    )
    _DISPATCH_REGISTERED = True


def load_silu_mul_extension(*, with_cuda: bool | None = None, verbose: bool = False) -> None:
    """Compile/load the C++ operator and register Python FakeTensor/Autograd rules."""
    _prepare_extension_tool_path()
    project_root = Path(__file__).resolve().parents[3]
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(project_root / "artifacts" / "torch_extensions")
    )
    source_root = project_root / "extensions" / "silu_mul"
    cpp_source = source_root / "silu_mul.cpp"
    cuda_source = source_root / "silu_mul_cuda.cu"
    if with_cuda is None:
        with_cuda = torch.cuda.is_available() and cuda_source.exists()
    if with_cuda and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    sources = [str(cpp_source)]
    if with_cuda:
        sources.append(str(cuda_source))
    windows = os.name == "nt"
    definition = "/DWITH_CUDA" if windows else "-DWITH_CUDA"
    cflags = ["/O2"] if windows else ["-O3"]
    if with_cuda:
        cflags.append(definition)
    load(
        name="forgellm_silu_mul_ext",
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=["-O3", "-DWITH_CUDA"] if with_cuda else None,
        with_cuda=with_cuda,
        is_python_module=False,
        verbose=verbose,
    )
    _register_python_dispatch()


def silu_mul(gate: Tensor, value: Tensor, *, use_extension: bool = False) -> Tensor:
    """Use the reference path or the loaded custom CPU/CUDA dispatcher path."""
    if not use_extension:
        return silu_mul_reference(gate, value)
    load_silu_mul_extension(with_cuda=gate.is_cuda)
    return cast(Tensor, torch.ops.forgellm_ops.silu_mul(gate, value))
