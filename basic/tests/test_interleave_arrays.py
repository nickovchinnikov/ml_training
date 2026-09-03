import pytest
import torch

from basic.interleave_arrays import solve


def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    out = torch.empty(A.numel() * 2, dtype=A.dtype, device=A.device)
    out[0::2] = A
    out[1::2] = B
    return out


@pytest.mark.parametrize(
    "A, B",
    [
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),          # Example 1
        ([10.0, 20.0], [30.0, 40.0]),               # Example 2
        ([1.5], [2.5]),                             # N = 1
        ([-1.0, -2.0], [3.0, 4.0]),                 # Negative values
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),         # Zeros
    ],
)
def test_examples(A, B):
    A = torch.tensor(A, dtype=torch.float32, device="cuda")
    B = torch.tensor(B, dtype=torch.float32, device="cuda")

    output = torch.empty(A.numel() * 2, dtype=torch.float32, device="cuda")
    solve(A, B, output, A.numel())

    expected = reference(A, B)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("N", [7, 32, 1000, 100_000])
def test_random(N):
    torch.manual_seed(0)

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)

    output = torch.empty(2 * N, device="cuda", dtype=torch.float32)
    solve(A, B, output, N)

    expected = reference(A, B)
    torch.testing.assert_close(output, expected)


def test_output_layout():
    A = torch.arange(8, device="cuda", dtype=torch.float32)
    B = torch.arange(100, 108, device="cuda", dtype=torch.float32)

    output = torch.empty(16, device="cuda", dtype=torch.float32)
    solve(A, B, output, 8)

    torch.testing.assert_close(output[0::2], A)
    torch.testing.assert_close(output[1::2], B)