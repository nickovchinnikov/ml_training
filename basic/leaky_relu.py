import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor):
    torch.where(input > 0, input, input * 0.01, out=output)
