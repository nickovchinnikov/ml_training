import torch


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
):
    window = input.unfold(0, kernel.shape[0], step=1)
    torch.matmul(window, kernel, out=output)

