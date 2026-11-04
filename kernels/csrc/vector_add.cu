#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
__global__ void add_kernel(const float* x, const float* y, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
torch::Tensor cuda_add(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == y.device(), "Devices must match");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32,
                "FP32 required");
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1 && x.sizes() == y.sizes(), "Equal vectors required");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "Contiguous inputs required");
    TORCH_CHECK(!x.requires_grad() && !y.requires_grad(), "Forward-only exercise");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    TORCH_CHECK(n <= 2147483647LL, "Exercise size limit exceeded");
    if (n > 0) {
        add_kernel<<<(n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
