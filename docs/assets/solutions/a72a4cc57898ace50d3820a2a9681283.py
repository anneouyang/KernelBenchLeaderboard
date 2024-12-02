import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU activation
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)