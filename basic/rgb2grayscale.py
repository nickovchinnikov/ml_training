import torch

W = None

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor):
    global W
    if W is None:
        W = torch.tensor([0.299, 0.587, 0.114], device=input.device)
    output[:] = input.view(-1, 3) @ W
    
