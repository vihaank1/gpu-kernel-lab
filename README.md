# GPU Kernel Lab

A Triton kernel computes exact attention a tile at a time using online softmax. This repository compares it with explicit PyTorch attention and PyTorch SDPA, and includes a separate CUDA vector-add kernel.

**TL;DR:** The FP32 kernel passed all nine forward checks on a Tesla T4. At N=512 and 1,024 it took **2.72× and 3.03× as long** as explicit PyTorch, respectively; default SDPA was also faster at both lengths.  
The kernel avoids storing a full score matrix. **Peak-memory savings have not yet been measured** in the published run; the benchmark now tests that claim through N=8,192.

## Limitations

- The published timings are from **one T4 session** through N=1,024. It did not retain timing samples or measure peak allocations. There is no published timing, spread, or memory result for N=2,048–8,192. The retained profiler table identifies kernels; no trace file or Nsight hardware counters are available.
- The Triton forward accepts contiguous 2D FP32 inputs with head/value widths up to 128. The causal path requires square attention. The backward is PyTorch tensor code, not a fused backward. The precision experiment rounds inputs, then computes in FP32; it does not benchmark FP16/BF16 Tensor Cores.
- On this T4 (CC 7.5), [PyTorch 2.11's CUDA FlashAttention backend requires SM80 or newer](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp). Default SDPA used a CUTLASS memory-efficient kernel in the recorded N=1,024 profile. These are not FA2 timings; automatic SDPA selection can differ on other hardware, dtypes or shapes.
- **Slowdown hypothesis:** one program owns 16 query rows and loops over 32-key tiles. That reuses the queries, but repeats reductions, rescaling and IEEE FP32 dot products at every tile. The timing alone cannot identify which cost dominates. Occupancy, instruction mix, stalls and DRAM traffic were not measured.

## Run

Run from the repository root, in a fresh Python environment. For CPU checks on Linux or macOS:

```bash
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=1 python -m kernels cpu
```

Expected: five backward PASS lines (N=1, 7, 31, 32, 33), followed by the precision table. On Linux with PyTorch 2.11.0+cpu the command took **3.1 s including startup**. CI runs the same checks on the Linux CPU wheel.

For the recorded GPU environment use Linux x86-64, Python 3.13, an NVIDIA GPU and driver compatible with CUDA 13.0, the CUDA 13.0 toolkit (including nvcc), and a compatible C++ compiler. The torch wheel alone does not install the compiler.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-gpu.txt
python -m pip check
MAX_JOBS=1 python -m kernels gpu
MAX_JOBS=1 python -m kernels benchmark --session t4-01 --output results/t4-01.json
```

The [Linux cu130 torch wheel metadata](https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl.metadata) pins **triton==3.6.0**, as does [PyTorch's release pin](https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/docker/triton_version.txt). MAX_JOBS=1 makes Ninja compile one translation unit at a time to limit host RAM usage; it does not change benchmark parallelism. Allow minutes for a cold CUDA extension build; later runs can reuse the build cache. GPU commands print actual elapsed seconds. The expanded GPU sweep has no verified wall-clock estimate yet.

Expected GPU output: nine fused-forward PASS lines, then nine CUDA-add PASS lines. The benchmark repeats those checks, validates every timed shape, and prints lines of the form `attention N=8192 triton: … us [Q1 …, Q3 …], peak +… MiB`. It writes JSON only on success and refuses to overwrite an existing run.

Repeat the benchmark in **three independent sessions** on the same hardware/software, changing both session label and filename (t4-01, t4-02, t4-03). Bring the reports together and run:

```bash
python -m kernels.sessions results/t4-01.json results/t4-02.json results/t4-03.json
```

Optional Linux logic check without a GPU: install Triton 3.6.0 and NumPy alongside CPU torch, then run `TRITON_INTERPRET=1 python -m kernels interpreter`. Expect nine PASS lines (~2.4 s including startup here). Unset TRITON_INTERPRET before CUDA runs.

## Results and measurement details

[Recorded T4 output and profiler table](results/t4_validation.txt) · [Benchmark source](kernels/benchmark.py). All attention timings use the same random FP32 Q, K and V, one head, D=DV=64, noncausal, no dropout, with TF32 disabled. Times below are microseconds.

| N | Explicit PyTorch | Triton online softmax | SDPA (default) |
|---:|---:|---:|---:|
| 128 | 22.560 | 23.200 | 34.224 |
| 512 | 42.912 | 116.736 | 108.224 |
| 1,024 | 141.280 | 428.032 | 190.656 |

| Vector elements | PyTorch (µs) | CUDA (µs) | Observed time ratio (PyTorch/CUDA) | Spread |
|---:|---:|---:|---:|:---|
| 4 | 6.784 | 4.304 | 1.58× | Not recorded |
| 1,000 | 4.640 | 4.288 | 1.08× | Not recorded |
| 1,000,003 | 55.296 | 52.224 | 1.06× | Not recorded |

Four additions are too little work to distinguish arithmetic throughput: the 1.58× ratio mostly reflects launch/dispatch costs. Without spread the 1.06–1.08× ratios do not establish a repeatable advantage. The last CUDA row implies **229.8 GB/s effective bandwidth**: two FP32 reads and one write per element, divided by 52.224 µs. That is **71.8% of the [T4's 320 GB/s peak](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/T4%20Product%20Brief.pdf)**. Logical bytes are not a hardware measurement of DRAM traffic.

| Protocol | Historical table | New benchmark |
|---|---|---|
| Timer | Triton do_bench, CUDA events, outside profiler | CUDA event pair on current stream; wait for end event |
| Warmup and timing | 50 ms warmup and 200 ms repetition **budgets**; auto-selected iteration counts were not saved | 5 warmup calls per method/shape; 10 calls per timed batch; 7 batches, shuffled method order each round |
| Report | Median from one session; no retained dispersion | Raw batch means in µs/call, median and Q1/Q3; paired vector time ratios. Across sessions: median and range of session summaries |
| Attention lengths | 128, 512, 1,024 | 128, 512, 1,024, 2,048, 4,096, 8,192 |
| Memory | No peak measurement | Extra peak **allocated** bytes above the live baseline, one warmed call per method/shape, synchronized; includes the output, excludes existing inputs and oracle |

The new timer measures repeated calls on warm inputs and includes stream gaps if the host cannot keep up. Compare methods **within a protocol**. Allocator peaks cover PyTorch-tracked allocations, not all device memory or memory traffic. At N=8,192 one explicit FP32 score matrix alone occupies 256 MiB. The experiment compares the observed growth for explicit attention, Triton and default SDPA; SDPA may itself avoid quadratic storage. The FP64 correctness oracle processes 128 query rows at a time and runs outside the measurement.

## Correctness checks

[Kernels and checks](kernels/checks.py). Each row below is one forward case; the four dimensions are (query length, key length, query/key width, value width). All nine use FP32 output, **atol=1e-5, rtol=1e-4**.

| # | Inputs | Reference or property |
|---:|---|---|
| 1 | (1, 1, 16, 8) | Explicit FP64 attention cast to FP32 |
| 2 | (7, 7, 16, 8) | Same; partial tiles |
| 3 | (17, 33, 32, 16) | Same; both sequence dimensions cross a tile |
| 4 | (31, 45, 29, 13) | Same; masked feature and value lanes |
| 5 | (7, 33, 16, 8), Q=0, K=100 | Every output row equals the mean of V |
| 6 | Same shape, Q=K=100 | Same mean despite large logits |
| 7 | Square causal N=1, D=16, DV=8 | Masked explicit FP64 attention cast to FP32 |
| 8 | Square causal N=7, same widths | Same; no future keys |
| 9 | Square causal N=33, same widths | Same; mask crosses the 32-key boundary |

**Backward:** compare all of dQ, dK and dV with causal FP64 SDPA autograd at N=1, 7, 31, 32 and 33, with atol=1e-10, rtol=1e-8 and a finiteness check. The final three straddle the forward key-tile boundary; backward itself is untiled. **CUDA add:** exact equality (atol=rtol=0) at N=0, 1, 4, 255, 256, 257, 1,000 and 1,000,003, plus N=1,003 on a non-default stream.

## Implementation and references

[PyTorch reference and backward](kernels/reference.py) · [Triton forward](kernels/triton_attention.py) · [CUDA vector add](kernels/csrc/vector_add.cu) · [GPU validation notebook](gpu_kernel_validation_results.ipynb) · [MIT license](LICENSE).

The forward keeps a running maximum, normalization sum and weighted value sum for each query row. When a new tile changes the maximum, it rescales the previous sums before adding the new contribution. This computes the full softmax-weighted result without allocating the full score matrix. See Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018), and Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022).
