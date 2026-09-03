import torch
import torch.nn.functional as F

from medium.softmax_attention import solve


def reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    return F.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(0),
        K.unsqueeze(0).unsqueeze(0),
        V.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
    ).squeeze(0).squeeze(0)


def test_example1():
    device = "cuda"

    Q = torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
    ], device=device)

    K = torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
    ], device=device)

    V = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
    ], device=device)

    output = torch.empty_like(Q)

    solve(Q, K, V, output)

    expected = reference(Q, K, V)

    torch.testing.assert_close(output, expected)


def test_example2():
    device = "cuda"

    Q = torch.tensor([[1., 2.]], device=device)

    K = torch.tensor([
        [1., 0.],
        [0., 1.],
    ], device=device)

    V = torch.tensor([
        [3., 4.],
        [5., 6.],
    ], device=device)

    output = torch.empty_like(Q)

    solve(Q, K, V, output)

    expected = reference(Q, K, V)

    torch.testing.assert_close(output, expected)


def test_random():
    torch.manual_seed(0)

    for M, N, d in [
        (1, 1, 1),
        (1, 2, 8),
        (4, 8, 16),
        (32, 64, 32),
        (128, 256, 64),
    ]:
        Q = torch.randn(M, d, device="cuda")
        K = torch.randn(N, d, device="cuda")
        V = torch.randn(N, d, device="cuda")

        output = torch.empty(M, d, device="cuda")

        solve(Q, K, V, output)

        expected = reference(Q, K, V)

        torch.testing.assert_close(
            output,
            expected,
            atol=1e-5,
            rtol=1e-5,
        )