# GPU Kernel Lab

PyTorch references, a Triton online-softmax attention kernel, and a CUDA C++ vector-add kernel, with correctness checks and measured Tesla T4 results. Built as the GPU kernel engineering phase of an AI systems learning roadmap.

The focus is understanding memory access, masking, tiling, numerical stability, fusion, and performance measurement. The attention kernel is a teaching implementation of the FlashAttention-style online-softmax idea, not a production FlashAttention replacement.

## Implemented and checked

- Causal attention backward in PyTorch tensor operations, compared with autograd in FP64 for sequence lengths 1, 7 and 33.
- FP32, FP16 and BF16 input-rounding error analysis against an FP64 reference.
- Triton fused attention forward, including irregular dimensions, uniform/large scores and square causal masking: nine GPU checks passed.
- CUDA C++ vector addition, including zero length, block boundaries, a large vector and a non-default stream: nine checks passed.
- Matched-input benchmarks and PyTorch CPU/CUDA profiling. Correctness assertions run before timing.

## Run

Use Python in a virtual environment. CPU checks need PyTorch and can run on macOS:

```bash
python -m pip install -r requirements-cpu.txt
python attention_finish_checks.py --cpu
```

For GPU checks, use Linux with CUDA-enabled PyTorch, an NVIDIA GPU, a compatible CUDA toolkit (`nvcc`) and a C++ compiler. The recorded environment was Python 3.13, PyTorch **2.11.0+cu130**, Triton **3.6.0**, CUDA **13.0**, and a Tesla **T4 (CC 7.5)**. A passing run on this configuration is not a general compatibility claim for all Triton versions or older GPUs.

In that configured GPU environment:

```bash
python -m pip install numpy ninja triton==3.6.0
MAX_JOBS=1 python attention_finish_checks.py --gpu
```

Optional Linux CPU interpreter checks, with Triton and NumPy installed:

```bash
TRITON_INTERPRET=1 python attention_finish_checks.py --interpreter
```

Interpreter execution checks logic; it does not validate GPU execution or performance. Unset `TRITON_INTERPRET` before using `--gpu`. Apple MPS does not execute these CUDA kernels.

The GPU run exports `attention_gpu_trace.json`. The historical trace was not retained, but its printed profiler table is included in the recorded results.

## Measured results

Tesla T4, November 2, 2026. Times are medians in microseconds after warmup, outside the profiler. These are one session's measurements, not cross-device guarantees.

### FP32 attention

Identical contiguous inputs, single head, `D=DV=64`, no dropout, noncausal timing. Causal correctness is tested separately.

| Sequence length | Explicit PyTorch | Teaching fused | Default PyTorch SDPA |
|---:|---:|---:|---:|
| 128 | 22.560 | 23.200 | 34.224 |
| 512 | 42.912 | 116.736 | 108.224 |
| 1,024 | 141.280 | 428.032 | 190.656 |

At length 1,024, the teaching kernel was **3.03x slower** than explicit PyTorch. Avoiding full score/probability intermediates did not offset the cost of small IEEE FP32 dot products and repeated online-softmax reductions/rescaling. This is an explanation based on the implementation and timings; occupancy, stalls and DRAM traffic were not measured directly.

The profile identifies the SDPA backend as a CUTLASS memory-efficient SM75 kernel. These results are not a forced FlashAttention-2 comparison. Profiler parent/child events overlap and its timings include instrumentation effects.

### FP32 vector addition

| Elements | PyTorch | CUDA C++ | Speedup |
|---:|---:|---:|---:|
| 4 | 6.784 | 4.304 | 1.58x |
| 1,000 | 4.640 | 4.288 | 1.08x |
| 1,000,003 | 55.296 | 52.224 | 1.06x |

Each element needs two FP32 reads and one write for one addition: 12 logical bytes per FLOP. The large-vector timing implies about 230 GB/s effective throughput, not a direct measurement of DRAM bandwidth. Fixed launch/dispatch costs matter most for tiny inputs.

## Files and scope

- `attention_finish_checks.py`: references, kernels, checks, timing and profiling. The CUDA C++ kernel is embedded in `build_cuda_add()` and compiled with PyTorch's extension loader.
- `gpu_kernel_validation_results.ipynb`: runnable notebook with captured output and architectural interpretation.
- `results/t4_validation.txt`: captured correctness, precision, benchmark and profiler output.

The fused kernel accepts contiguous FP32 2D inputs and supports square causal attention. Manual backward uses PyTorch operations; there is no custom fused GPU backward. The precision table rounds inputs to each dtype, then computes in FP32; it does not measure native FP16/BF16 Tensor Core arithmetic. Tests cover sampled shapes rather than all possible inputs.

Earlier roadmap exercises are not included in this snapshot. Further work includes tile tuning, native low-precision kernels, fused backward, hardware-counter profiling and integration into a full model.
