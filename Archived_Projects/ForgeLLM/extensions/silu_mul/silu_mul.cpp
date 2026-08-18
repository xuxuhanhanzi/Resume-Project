#include <ATen/ATen.h>
#include <torch/library.h>

namespace forgellm_ops {

void check_inputs(const at::Tensor& gate, const at::Tensor& value) {
  TORCH_CHECK(gate.sizes() == value.sizes(), "gate and value must have identical shapes");
  TORCH_CHECK(gate.scalar_type() == value.scalar_type(), "gate and value must share dtype");
  TORCH_CHECK(gate.device() == value.device(), "gate and value must share device");
  TORCH_CHECK(gate.is_contiguous() && value.is_contiguous(), "gate and value must be contiguous");
  TORCH_CHECK(gate.is_floating_point(), "silu_mul supports floating-point tensors only");
}

at::Tensor silu_mul_cpu(const at::Tensor& gate, const at::Tensor& value) {
  check_inputs(gate, value);
  TORCH_CHECK(gate.device().is_cpu(), "CPU implementation received a non-CPU tensor");
  return at::silu(gate) * value;
}

#ifdef WITH_CUDA
at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& value);
#endif

}  // namespace forgellm_ops

TORCH_LIBRARY(forgellm_ops, library) {
  library.def("silu_mul(Tensor gate, Tensor value) -> Tensor");
}

TORCH_LIBRARY_IMPL(forgellm_ops, CPU, library) {
  library.impl("silu_mul", forgellm_ops::silu_mul_cpu);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(forgellm_ops, CUDA, library) {
  library.impl("silu_mul", forgellm_ops::silu_mul_cuda);
}
#endif
