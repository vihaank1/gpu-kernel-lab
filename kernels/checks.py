import os
import torch
import torch.nn.functional as F
from .reference import attention, manual_backward
from .triton_attention import make_fused
from .cuda_add import build_cuda_add


def cpu_checks():
    torch.manual_seed(1)
    for n in (1, 7, 31, 32, 33):
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


def gpu_checks():
    assert os.getenv('TRITON_INTERPRET', '0') != '1', 'Unset TRITON_INTERPRET for real GPU execution'
    assert torch.cuda.is_available(), 'BLOCKED: a working NVIDIA GPU runtime is required'
    if torch.cuda.get_device_capability()[0] < 8:
        print('Compatibility test on pre-Ampere GPU; correctness must pass before timing')
    import triton
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

    return fused, cuda_add
