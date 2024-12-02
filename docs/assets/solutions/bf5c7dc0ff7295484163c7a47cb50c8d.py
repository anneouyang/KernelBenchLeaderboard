import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = abs(diff);
        if (abs_diff < 1.0f) {
            out[idx] = 0.5f * diff * diff;
        } else {
            out[idx] = abs_diff - 0.5f;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

smooth_l1_loss_cpp_source = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss = load_inline(
    name='smooth_l1_loss',
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=['smooth_l1_loss_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        loss = self.smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)
        return loss.mean()