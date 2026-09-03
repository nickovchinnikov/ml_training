import pytest
import torch

from basic.softmax import solve


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100, 1000, 100000])
def test_random(n):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    solve(x, out)

    expected = torch.softmax(x, dim=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_large_positive():
    x = torch.tensor(
        [1000.0, 1001.0, 1002.0],
        device="cuda",
        dtype=torch.float32,
    )

    out = torch.empty_like(x)
    solve(x, out)

    expected = torch.softmax(x, dim=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_large_negative():
    x = torch.tensor(
        [-1002.0, -1001.0, -1000.0],
        device="cuda",
        dtype=torch.float32,
    )

    out = torch.empty_like(x)
    solve(x, out)

    expected = torch.softmax(x, dim=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_uniform():
    x = torch.full(
        (1024,),
        7.5,
        device="cuda",
        dtype=torch.float32,
    )

    out = torch.empty_like(x)
    solve(x, out)

    expected = torch.softmax(x, dim=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_single_element():
    x = torch.tensor([42.0], device="cuda", dtype=torch.float32)

    out = torch.empty_like(x)
    solve(x, out)

    expected = torch.softmax(x, dim=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

