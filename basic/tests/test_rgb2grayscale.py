import pytest
import torch

from basic.rgb2grayscale import solve


def reference(x: torch.Tensor) -> torch.Tensor:
    rgb = x.reshape(-1, 3)
    return (
        rgb[:, 0] * 0.299
        + rgb[:, 1] * 0.587
        + rgb[:, 2] * 0.114
    )


@pytest.mark.parametrize(
    "width,height,input_values,expected",
    [
        (
            2,
            2,
            [
                255.0, 0.0, 0.0,
                0.0, 255.0, 0.0,
                0.0, 0.0, 255.0,
                128.0, 128.0, 128.0,
            ],
            [76.245, 149.685, 29.07, 128.0],
        ),
        (
            1,
            1,
            [100.0, 150.0, 200.0],
            [140.75],
        ),
        (
            1,
            1,
            [0.0, 0.0, 0.0],
            [0.0],
        ),
        (
            1,
            1,
            [255.0, 255.0, 255.0],
            [255.0],
        ),
        (
            3,
            1,
            [
                255.0, 0.0, 0.0,
                0.0, 255.0, 0.0,
                0.0, 0.0, 255.0,
            ],
            [76.245, 149.685, 29.07],
        ),
    ],
)
def test_examples(width, height, input_values, expected):
    inp = torch.tensor(input_values, dtype=torch.float32, device="cuda")
    out = torch.empty(width * height, dtype=torch.float32, device="cuda")

    solve(inp, out)

    expected = torch.tensor(expected, dtype=torch.float32, device="cuda")
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("width,height", [(2, 2), (8, 4), (31, 17), (64, 64)])
def test_random(width, height):
    torch.manual_seed(width * 1000 + height)

    inp = torch.rand(width * height * 3, device="cuda") * 255
    out = torch.empty(width * height, device="cuda")

    solve(inp, out)

    expected = reference(inp)
    assert torch.allclose(out, expected, atol=1e-5)


def test_large_random():
    torch.manual_seed(42)

    width, height = 512, 512
    inp = torch.rand(width * height * 3, device="cuda") * 255
    out = torch.empty(width * height, device="cuda")

    solve(inp, out)

    expected = reference(inp)
    assert torch.allclose(out, expected, atol=1e-5)