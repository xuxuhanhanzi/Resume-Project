"""Custom C++/CUDA operators used by the ForgeLLM learning project."""

from forgellm.custom_ops.silu_mul import (
    load_silu_mul_extension,
    silu_mul,
    silu_mul_reference,
)

__all__ = ["load_silu_mul_extension", "silu_mul", "silu_mul_reference"]
