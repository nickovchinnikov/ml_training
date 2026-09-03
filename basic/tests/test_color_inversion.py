import torch

from basic.color_inversion import solve


def invert_reference(image):
    expected = image.clone()
    expected[0::4] = 255 - expected[0::4]  # R
    expected[1::4] = 255 - expected[1::4]  # G
    expected[2::4] = 255 - expected[2::4]  # B
    # Alpha не меняется
    return expected


def test_example_1():
    image = torch.tensor(
        [255, 0, 128, 255,
         0, 255, 0, 255],
        device="cuda",
        dtype=torch.uint8,
    )

    solve(image, 1, 2)

    expected = torch.tensor(
        [0, 255, 127, 255,
         255, 0, 255, 255],
        device="cuda",
        dtype=torch.uint8,
    )

    assert torch.equal(image, expected)


def test_example_2():
    image = torch.tensor(
        [10, 20, 30, 255,
         100, 150, 200, 255],
        device="cuda",
        dtype=torch.uint8,
    )

    solve(image, 2, 1)

    expected = torch.tensor(
        [245, 235, 225, 255,
         155, 105, 55, 255],
        device="cuda",
        dtype=torch.uint8,
    )

    assert torch.equal(image, expected)


def test_random():
    torch.manual_seed(42)

    for width, height in [
        (1, 1),
        (2, 3),
        (16, 16),
        (127, 31),
        (512, 512),
    ]:
        image = torch.randint(
            0,
            256,
            (width * height * 4,),
            device="cuda",
            dtype=torch.uint8,
        )

        expected = invert_reference(image)

        solve(image, width, height)

        assert torch.equal(image, expected)