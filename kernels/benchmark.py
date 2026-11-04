"""Matched-input, single-stream CUDA measurements; compilation is outside timing."""
import json
import random
import time
from statistics import median, quantiles

import torch
import torch.nn.functional as F

from .reference import attention


def quartiles(xs):
    q1, _, q3 = quantiles(xs, method="inclusive")
    return median(xs), q1, q3


def sample(fn, iterations):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / iterations


def peak_bytes(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def compare(kind, n, calls, ref, args, rng):
    samples, peaks = {}, {}
    for name, fn in calls.items():
        value = fn()
        tol = dict(atol=0, rtol=0) if kind == "add" else dict(atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(value, ref, **tol)
        del value
        for _ in range(args.warmup):
            fn()
        peaks[name] = peak_bytes(fn)
        samples[name] = []
    order = list(calls)
    for _ in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            samples[name].append(sample(calls[name], args.iterations))
    rows = []
    for name, xs in samples.items():
        mid, lo, hi = quartiles(xs)
        print(f"{kind} N={n} {name}: {mid:.3f} us [Q1 {lo:.3f}, Q3 {hi:.3f}], "
              f"peak +{peaks[name]/2**20:.3f} MiB", flush=True)
        rows.append(dict(kind=kind, n=n, method=name, samples_us=xs, median_us=mid,
                         q1_us=lo, q3_us=hi, peak_allocated_bytes=peaks[name]))
    if kind == "add":
        ratios = [a/b for a, b in zip(samples["torch"], samples["cuda"])]
        mid, lo, hi = quartiles(ratios)
        print(f"  paired torch/cuda ratio: {mid:.3f}x [Q1 {lo:.3f}, Q3 {hi:.3f}]")
    return rows


@torch.no_grad()
def run(fused, cuda_add, args):
    import triton
    torch.manual_seed(11)
    rng = random.Random(11)
    start = time.monotonic()
    rows = []
    for n in (4, 1000, 1000003):
        x = torch.randn(n, device="cuda")
        y = torch.randn_like(x)
        rows.extend(compare("add", n, {"torch": lambda: x+y, "cuda": lambda: cuda_add(x,y)},
                            x+y, args, rng))
    for n in (128, 512, 1024, 2048, 4096, 8192):
        q = torch.randn(n, 64, device="cuda")
        k, v = torch.randn_like(q), torch.randn_like(q)
        # All rows checked; a 128 x N FP64 score tile avoids an N x N oracle.
        kd, vd = k.double(), v.double()
        ref = torch.cat([attention(chunk.double(), kd, vd, False).float()
                         for chunk in q.split(128)])
        del kd, vd
        calls = {"explicit": lambda: attention(q,k,v,False), "triton": lambda: fused(q,k,v),
                 "sdpa_default": lambda: F.scaled_dot_product_attention(
                     q[None,None], k[None,None], v[None,None], dropout_p=0.)[0,0]}
        rows.extend(compare("attention", n, calls, ref, args, rng))
    report = dict(session=args.session, gpu=torch.cuda.get_device_name(),
                  cc=torch.cuda.get_device_capability(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, triton=triton.__version__, seed=11,
                  warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
                  benchmark_seconds=time.monotonic()-start, rows=rows)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output} ({report['benchmark_seconds']:.1f} benchmark seconds)")
