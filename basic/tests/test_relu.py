import torch

from basic.relu import solve


def run_test(inp):
    inp = torch.tensor(inp, dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)

    solve(inp, out)

    expected = torch.relu(inp)

    assert torch.equal(out, expected), (
        f"\ninput={inp.cpu()}"
        f"\nexpected={expected.cpu()}"
        f"\nactual={out.cpu()}"
    )


def test_examples():
    run_test([-2.0, -1.0, 0.0, 1.0, 2.0])
    run_test([-3.5, 0.0, 4.2])


def test_all_negative():
    run_test([-1.0, -5.0, -100.0, -1e-6])


def test_all_positive():
    run_test([1.0, 2.5, 100.0, 1e-6])


def test_all_zero():
    run_test([0.0] * 16)


def test_mixed():
    run_test([
        -10.0,
        0.0,
        3.5,
        -4.2,
        7.1,
        -1e-8,
        1e-8,
        1000.0,
        -1000.0,
    ])


def test_single_element():
    run_test([-5.0])
    run_test([0.0])
    run_test([8.0])


def test_large():
    x = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    solve(x, out)

    assert torch.equal(out, torch.relu(x))

