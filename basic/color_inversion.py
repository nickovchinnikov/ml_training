import torch


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    image[::4] = 255 - image[::4]
    image[1::4] = 255 - image[1::4]
    image[2::4] = 255 - image[2::4]
