import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor):
    # max trick to prevent overflow
    x = input - input.max()
    x_exp = torch.exp(x)
    output[:] = x_exp / x_exp.sum()

