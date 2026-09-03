import pytest
import torch

from basic.rainbow_table import fnv1a_hash, solve


def reference(input_tensor: torch.Tensor, rounds: int) -> torch.Tensor:
    x = input_tensor.clone()
    for _ in range(rounds):
        x = fnv1a_hash(x)
    return x


@pytest.mark.parametrize(
    "numbers, rounds",
    [
        ([123, 456, 789], 2),                      # Example 1
        ([0, 1, 2147483647], 3),                   # Example 2
        ([42], 1),                                # Single element
        ([42], 10),                               # Many rounds
        ([0], 100),                               # Max rounds
        ([0, 0, 0, 0], 5),                        # All zeros
        ([1, 2, 3, 4, 5], 1),                     # One round
        ([1, 2, 3, 4, 5], 10),                    # Multiple rounds
        ([2147483647] * 8, 4),                    # Max int values
        (list(range(256)), 3),                    # Sequential values
    ],
)
def test_rainbow_table(numbers, rounds):
    device = "cuda"

    inp = torch.tensor(numbers, dtype=torch.int32, device=device)
    out = torch.empty_like(inp)

    solve(inp, out, len(numbers), rounds)

    expected = reference(inp, rounds)

    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("rounds", [1, 2, 5, 10, 100])
def test_random(rounds):
    device = "cuda"

    torch.manual_seed(42)

    inp = torch.randint(
        0,
        2**31,
        (10000,),
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty_like(inp)

    solve(inp, out, inp.numel(), rounds)

    expected = reference(inp, rounds)

    torch.testing.assert_close(out, expected)


def test_output_written():
    device = "cuda"

    inp = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    out = torch.full_like(inp, -1)

    solve(inp, out, 3, 2)

    assert not torch.equal(out, torch.full_like(inp, -1))

