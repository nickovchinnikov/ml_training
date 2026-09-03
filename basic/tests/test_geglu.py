import pytest
import torch
import torch.nn.functional as F

from basic.geglu import solve


def reference(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2)
    return a * F.gelu(b)


@pytest.mark.parametrize("n", [2, 4, 16, 128, 1024])
def test_random(n):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty(n // 2, device="cuda", dtype=torch.float32)

    solve(x, out)

    expected = reference(x)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_example_1():
    x = torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)

    solve(x, out)

    expected = reference(x)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_example_2():
    x = torch.tensor([2.0, -1.0, 1.0, 0.5], device="cuda", dtype=torch.float32)
    out = torch.empty(2, device="cuda", dtype=torch.float32)

    solve(x, out)

    expected = reference(x)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_zero_input():
    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    out = torch.empty(32, device="cuda", dtype=torch.float32)

    solve(x, out)

    expected = reference(x)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_extreme_values():
    x = torch.tensor(
        [100.0, -100.0, 100.0, -100.0],
        device="cuda",
        dtype=torch.float32,
    )
    out = torch.empty(2, device="cuda", dtype=torch.float32)

    solve(x, out)

    expected = reference(x)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)