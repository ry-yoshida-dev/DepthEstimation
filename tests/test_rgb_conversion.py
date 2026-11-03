import numpy as np
import pytest
from PIL import Image

from depth_estimation.rgb_conversion import RgbConversion


def test_rgb_image_is_returned_as_is() -> None:
    image: Image.Image = Image.new("RGB", (2, 2))
    assert RgbConversion.convert(image) is image


def test_grayscale_is_converted() -> None:
    assert RgbConversion.convert(Image.new("L", (2, 2), 7)).getpixel((0, 0)) == (7, 7, 7)


def test_transparency_is_composited_over_white() -> None:
    converted: Image.Image = RgbConversion.convert(Image.new("RGBA", (2, 2), (10, 20, 30, 0)))
    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (255, 255, 255)


def test_palette_transparency_is_composited_over_white() -> None:
    image: Image.Image = Image.new("P", (2, 2), 0)
    image.putpalette([1, 2, 3])
    image.info["transparency"] = 0
    assert RgbConversion.convert(image).getpixel((0, 0)) == (255, 255, 255)


def test_high_bit_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="high-bit-depth"):
        RgbConversion.convert(Image.fromarray(np.zeros((2, 2), dtype=np.uint16)))
