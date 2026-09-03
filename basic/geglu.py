import math

import torch

SQRT2 = math.sqrt(2)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor):
    a, b = input.chunk(2)

    output[:] = a * (0.5 * b * (1 + torch.erf(b / SQRT2)))
