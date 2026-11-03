"""Repairs and checks for the kernels already covered in the GPU lab.

python attention_finish_checks.py --cpu
TRITON_INTERPRET=1 python attention_finish_checks.py --interpreter  # Linux
python attention_finish_checks.py --gpu  # Linux, NVIDIA CUDA toolkit

Verified on Tesla T4 (CC 7.5), torch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0.
This specific configuration passed; it is not a general pre-Ampere support claim.

CPU mode needs torch only. Interpreter mode also needs triton and numpy.
GPU mode also needs triton, ninja, a CUDA-enabled torch, nvcc and a C++ compiler.
This is a teaching FP32 fused forward, not a tuned FlashAttention implementation.
The manual backward is PyTorch tensor code, not a fused GPU backward kernel.
GPU mode tests before benchmarking; any failed assertion stops execution.
"""
import argparse
import math
import os
import torch
import torch.nn.functional as F


def attention(q, k, v, causal=True):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        rows = torch.arange(q.shape[-2], device=q.device)
        cols = torch.arange(k.shape[-2], device=q.device)
        scores = scores.masked_fill(cols[None, :] > rows[:, None], -float('inf'))
    return scores.softmax(dim=-1) @ v


@torch.no_grad()
def manual_backward(q, k, v, upstream):
    scale = 1 / math.sqrt(q.shape[-1])
    rows = torch.arange(q.shape[-2], device=q.device)
    cols = torch.arange(k.shape[-2], device=q.device)
    allowed = cols[None, :] <= rows[:, None]
    scores = (q @ k.transpose(-2, -1)) * scale
    p = scores.masked_fill(~allowed, -float('inf')).softmax(dim=-1)
    dv = p.transpose(-2, -1) @ upstream
    dp = upstream @ v.transpose(-2, -1)
    correction = (dp * p).sum(dim=-1, keepdim=True)
    ds = (p * (dp - correction)).masked_fill(~allowed, 0.0)
    dq = (ds @ k) * scale
    dk = (ds.transpose(-2, -1) @ q) * scale
    return dq, dk, dv


def cpu_checks():
    torch.manual_seed(1)
    for n in (1, 7, 33):
        q = torch.randn(2, 3, n, 16, dtype=torch.float64, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn(2, 3, n, 8, dtype=torch.float64, requires_grad=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        upstream = torch.randn_like(ref)
        expected = torch.autograd.grad(ref, (q, k, v), grad_outputs=upstream)
        for ours, target in zip(manual_backward(q, k, v, upstream), expected):
            assert isinstance(ours, torch.Tensor), type(ours)
            assert torch.isfinite(ours).all().item()
            torch.testing.assert_close(ours, target, atol=1e-10, rtol=1e-8)
        print(f'PASS | manual backward | N={n}')

    torch.manual_seed(2)
    q = torch.randn(1, 2, 33, 32, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 2, 33, 16, dtype=torch.float64)
    ref = attention(q, k, v)
    print('\nPrecision: rounded inputs, FP32 arithmetic; reference is original FP64')
    print('dtype      input-rounding max    total max          relative L2')
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        stored = [t.to(dtype) for t in (q, k, v)]
        rounded_ref = attention(*(t.double() for t in stored))
        actual = attention(*(t.float() for t in stored)).double()
        assert torch.isfinite(actual).all().item()
        rounding = (rounded_ref - ref).abs().max().item()
        total = (actual - ref).abs().max().item()
        relative = ((actual-ref).norm() / ref.norm()).item()
        print(f'{str(dtype).split(".")[-1]:10} {rounding:18.6e} {total:18.6e} {relative:18.6e}')


def make_fused():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Q, K, V, O, M: tl.constexpr, N: tl.constexpr,
               D: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr,
               CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BD: tl.constexpr, BV: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        features = tl.arange(0, BD)
        vf = tl.arange(0, BV)
        q = tl.load(Q + rows[:, None]*D + features[None, :],
                    (rows[:, None] < M) & (features[None, :] < D), other=0).to(tl.float32)
        m = tl.full((BM,), -float('inf'), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BV), tl.float32)
        for tile in range(tl.cdiv(N, BN)):
            keys = tile * BN + tl.arange(0, BN)
            k = tl.load(K + keys[None, :]*D + features[:, None],
                        (keys[None, :] < N) & (features[:, None] < D), other=0).to(tl.float32)
            v = tl.load(V + keys[:, None]*DV + vf[None, :],
                        (keys[:, None] < N) & (vf[None, :] < DV), other=0).to(tl.float32)
            scores = tl.dot(q, k, input_precision='ieee') * SCALE
            allowed = keys[None, :] < N
            if CAUSAL:
                allowed = allowed & (keys[None, :] <= rows[:, None])
            scores = tl.where(allowed, scores, -float('inf'))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision='ieee')
            l = l * alpha + tl.sum(p, axis=1)
            m = new_m
        tl.store(O + rows[:, None]*DV + vf[None, :], acc / l[:, None],
                 (rows[:, None] < M) & (vf[None, :] < DV))

    def fused(q, k, v, causal=False):
        assert q.ndim == k.ndim == v.ndim == 2
        assert q.device == k.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32
        assert all(t.is_contiguous() and not t.requires_grad for t in (q, k, v))
        m, d = q.shape
        n, dv = v.shape
        assert k.shape == (n, d) and min(m, n, d, dv) > 0
        assert max(d, dv) <= 128
        assert not causal or m == n, 'This exercise supports square causal attention'
        out = torch.empty((m, dv), dtype=q.dtype, device=q.device)
        kernel[(triton.cdiv(m, 16),)](q, k, v, out, m, n, d, dv, d**-.5,
                                    causal, 16, 32, max(16, triton.next_power_of_2(d)),
                                    max(16, triton.next_power_of_2(dv)), num_warps=4)
        return out
    return fused


def fused_checks(fused, device):
    torch.manual_seed(3)
    for m, n, d, dv in ((1, 1, 16, 8), (7, 7, 16, 8), (17, 33, 32, 16), (31, 45, 29, 13)):
        q = torch.randn(m, d, device=device)
        k = torch.randn(n, d, device=device)
        v = torch.randn(n, dv, device=device)
        ref = attention(q.double(), k.double(), v.double(), False).float()
        torch.testing.assert_close(fused(q, k, v), ref, atol=1e-5, rtol=1e-4)
        print(f'PASS | fused forward {device} | {(m, n, d, dv)}')
    for value in (0., 100.):
        q = torch.full((7, 16), value, device=device)
        k = torch.full((33, 16), 100., device=device)
        v = torch.randn(33, 8, device=device)
        torch.testing.assert_close(fused(q, k, v), v.mean(0).expand(7, 8), atol=1e-5, rtol=1e-4)
        print(f'PASS | fused forward {device} | uniform scores, q={value}')
    for n in (1, 7, 33):
        q = torch.randn(n, 16, device=device)
        k = torch.randn_like(q)
        v = torch.randn(n, 8, device=device)
        ref = attention(q.double(), k.double(), v.double()).float()
        torch.testing.assert_close(fused(q, k, v, True), ref, atol=1e-5, rtol=1e-4)
        print(f'PASS | fused causal forward {device} | N={n}')


def build_cuda_add():
    from torch.utils.cpp_extension import load_inline
    source = r'''
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
'''
    return load_inline(name='gpu_lab_checked_add',
                       cpp_sources='torch::Tensor cuda_add(torch::Tensor x, torch::Tensor y);',
                       cuda_sources=source, functions=['cuda_add'],
                       extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False).cuda_add


def gpu_checks_and_measurements():
    assert os.getenv('TRITON_INTERPRET', '0') != '1', 'Unset TRITON_INTERPRET for real GPU execution'
    assert torch.cuda.is_available(), 'BLOCKED: a working NVIDIA GPU runtime is required'
    if torch.cuda.get_device_capability()[0] < 8:
        print('Compatibility test on pre-Ampere GPU; correctness must pass before timing')
    import triton
    import triton.testing
    print('GPU:', torch.cuda.get_device_name(), 'CC:', torch.cuda.get_device_capability())
    print('Torch:', torch.__version__, 'Triton:', triton.__version__, 'CUDA:', torch.version.cuda)
    torch.backends.cuda.matmul.allow_tf32 = False
    fused = make_fused()
    fused_checks(fused, 'cuda')
    cuda_add = build_cuda_add()
    for n in (0, 1, 4, 255, 256, 257, 1000, 1000003):
        x = torch.randn(n, device='cuda')
        y = torch.randn_like(x)
        torch.testing.assert_close(cuda_add(x, y), x+y, atol=0, rtol=0)
        print(f'PASS | CUDA add | N={n}')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.randn(1003, device='cuda')
        y = torch.randn_like(x)
        ours, ref = cuda_add(x, y), x+y
    stream.synchronize()
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)
    print('PASS | CUDA add | non-default stream')

    def measure(label, fn):
        # Warmup and compilation occur before timing; do_bench uses CUDA events.
        milliseconds = triton.testing.do_bench(fn, warmup=50, rep=200, return_mode='median')
        print(f'{label}: {milliseconds*1000:.3f} us')

    for n in (4, 1000, 1000003):
        x = torch.randn(n, device='cuda')
        y = torch.randn_like(x)
        measure(f'add N={n} PyTorch', lambda: x+y)
        measure(f'add N={n} CUDA', lambda: cuda_add(x, y))
    for n in (128, 512, 1024):
        q = torch.randn(n, 64, device='cuda')
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        calls = {
            'explicit PyTorch': lambda: attention(q, k, v, False),
            'fused teaching kernel': lambda: fused(q, k, v),
            'PyTorch SDPA default': lambda: F.scaled_dot_product_attention(
                q[None, None], k[None, None], v[None, None], dropout_p=0., is_causal=False)[0, 0],
        }
        ref = attention(q.double(), k.double(), v.double(), False).float()
        for label, fn in calls.items():
            torch.testing.assert_close(fn(), ref, atol=1e-5, rtol=1e-4)
            measure(f'attention FP32 N={n} {label}', fn)
    from torch.profiler import profile, ProfilerActivity, record_function
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        for _ in range(5):
            for label, fn in calls.items():
                with record_function(label):
                    fn()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=20))
    prof.export_chrome_trace('attention_gpu_trace.json')
    print('Saved attention_gpu_trace.json; timings above exclude profiler overhead.')
    print('For hardware counters, run Nsight Compute on the attention kernel separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cpu', action='store_true')
    modes.add_argument('--interpreter', action='store_true')
    modes.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    if args.cpu:
        cpu_checks()
    elif args.interpreter:
        assert os.getenv('TRITON_INTERPRET') == '1', 'Set TRITON_INTERPRET=1 before starting Python'
        print('CPU interpreter: checks logic only, not GPU compilation, races or speed.')
        fused_checks(make_fused(), 'cpu')
    else:
        gpu_checks_and_measurements()
