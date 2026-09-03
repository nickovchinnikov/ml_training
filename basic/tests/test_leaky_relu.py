import pytest
import torch

from basic.leaky_relu import solve


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=0.01)


@pytest.mark.parametrize(
    "data",
    [
        # Examples from the statement
        [1.0, -2.0, 3.0, -4.0],
        [-1.5, 0.0, 2.5, -3.0],
        # Single element
        [5.0],
        [-5.0],
        [0.0],
        # Same sign
        [0.5, 1.0, 2.0, 1000.0],
        [-0.5, -1.0, -2.0, -1000.0],
        # Around zero
        [-1e-6, 0.0, 1e-6],
        # Mixed signs
        [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0],
        # Fractional values
        [-123.456, -0.25, 0.25, 123.456],
        # Boundary values
        [-1000.0, 1000.0],
        # Repeated values
        [-2.0, -2.0, 2.0, 2.0, 0.0],
        # Miscellaneous
        [7.3, -8.4, 0.0, -0.001, 0.001, 999.99, -999.99],
    ],
)
def test_leaky_relu(data):
    x = torch.tensor(data, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    solve(x, out)

    expected = reference(x)

    torch.testing.assert_close(out, expected)

