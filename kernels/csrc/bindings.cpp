#include <torch/extension.h>
torch::Tensor cuda_add(torch::Tensor x, torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("cuda_add", &cuda_add);
}
