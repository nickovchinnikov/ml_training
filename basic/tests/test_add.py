import torch

from basic.add import solve


def test_vector_addition():
    for n in [1, 10, 1000, 100000]:
        A = torch.randn(n, device="cuda")
        B = torch.randn(n, device="cuda")
        C = torch.empty_like(A)

        solve(A, B, C)

        assert torch.equal(C, A + B)


def test_examples():
    A = torch.tensor([1., 2., 3., 4.], device="cuda")
    B = torch.tensor([5., 6., 7., 8.], device="cuda")
    C = torch.empty_like(A)

    solve(A, B, C)

    assert torch.equal(C, torch.tensor([6., 8., 10., 12.], device="cuda"))


def test_random():
    for n in [1, 32, 1000, 100000]:
        A = torch.randn(n, device="cuda")
        B = torch.randn(n, device="cuda")
        C = torch.empty_like(A)

        solve(A, B, C)

        assert torch.equal(C, A + B)

