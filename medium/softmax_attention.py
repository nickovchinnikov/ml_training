import math

import torch


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor
):
    d = Q.shape[1]

    output[:] = torch.softmax(((Q @ K.T) / math.sqrt(d)), dim=-1) @ V
