from pathlib import Path

def build_cuda_add():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / "csrc"
    return load(name="gpu_lab_vector_add", sources=[str(root / "bindings.cpp"), str(root / "vector_add.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"]).cuda_add
