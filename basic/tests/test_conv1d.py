import pytest
import torch

from basic.conv1d import solve


@pytest.mark.parametrize(
    "input_data,kernel_data,expected",
    [
        # Example 1
        (
            [1, 2, 3, 4, 5],
            [1, 0, -1],
            [-2, -2, -2],
        ),
        # Example 2
        (
            [2, 4, 6, 8],
            [0.5, 0.2],
            [1.8, 3.2, 4.6],
        ),
        # Kernel of size 1 (identity scaling)
        (
            [1, 2, 3],
            [2],
            [2, 4, 6],
        ),
        # Kernel equals input size
        (
            [1, 2, 3],
            [1, 1, 1],
            [6],
        ),
        # Negative values
        (
            [-1, -2, -3, -4],
            [1, -1],
            [1, 1, 1],
        ),
        # Floating point values
        (
            [1.5, 2.5, 3.5],
            [0.5, 0.5],
            [2.0, 3.0],
        ),
    ],
)
def test_solve(input_data, kernel_data, expected):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    kernel_tensor = torch.tensor(kernel_data, dtype=torch.float32)

    output = torch.empty(
        len(input_data) - len(kernel_data) + 1,
        dtype=torch.float32,
    )

    solve(input_tensor, kernel_tensor, output)

    expected = torch.tensor(expected, dtype=torch.float32)

    assert torch.allclose(output, expected)

