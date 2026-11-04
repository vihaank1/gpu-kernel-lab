import argparse
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest="mode", required=True)
    for mode in ("cpu", "interpreter", "gpu"):
        modes.add_parser(mode)
    b = modes.add_parser("benchmark")
    b.add_argument("--session", required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--iterations", type=int, default=10)
    b.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    if a.mode == "benchmark":
        if min(a.warmup, a.iterations) < 1 or a.repeats < 2:
            p.error("need positive warmup and iterations and at least two repeats")
        if a.output.exists() or not a.output.parent.is_dir():
            p.error("choose a new output file in an existing directory")
    from .checks import cpu_checks, fused_checks, gpu_checks, make_fused
    start = time.monotonic()
    if a.mode == "cpu":
        cpu_checks()
    elif a.mode == "interpreter":
        if os.getenv("TRITON_INTERPRET") != "1":
            p.error("start Python with TRITON_INTERPRET=1")
        fused_checks(make_fused(), "cpu")
    else:
        fused, cuda_add = gpu_checks()
        if a.mode == "benchmark":
            from .benchmark import run
            run(fused, cuda_add, a)
    print(f"Elapsed: {time.monotonic()-start:.1f} s")


if __name__ == "__main__":
    main()
