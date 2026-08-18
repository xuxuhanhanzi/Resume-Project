#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace forgellm_ops {

void check_inputs(const at::Tensor& gate, const at::Tensor& value);

template <typename scalar_t>
__global__ void silu_mul_kernel(
    const scalar_t* gate,
    const scalar_t* value,
    scalar_t* output,
    int64_t element_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  using accumulator_t = at::acc_type<scalar_t, true>;
  const accumulator_t gate_value = static_cast<accumulator_t>(gate[index]);
  const accumulator_t silu = gate_value / (accumulator_t{1} + exp(-gate_value));
  output[index] = static_cast<scalar_t>(silu * static_cast<accumulator_t>(value[index]));
}

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& value) {
  check_inputs(gate, value);
  TORCH_CHECK(gate.is_cuda(), "CUDA implementation received a non-CUDA tensor");
  auto output = at::empty_like(gate);
  if (gate.numel() == 0) {
    return output;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((gate.numel() + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate.scalar_type(),
      "forgellm_silu_mul_cuda",
      [&] {
        silu_mul_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gate.numel());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace forgellm_ops
