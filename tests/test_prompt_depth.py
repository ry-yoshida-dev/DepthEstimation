import numpy as np
import pytest
from PIL import Image

from depth_estimation import PromptDepth
from depth_estimation.array_types import FloatArray


def test_from_image_scales_16_bit_millimeters_to_meters() -> None:
    image: Image.Image = Image.fromarray(np.array([[1000, 2500]], dtype=np.uint16))
    prompt_depth: PromptDepth = PromptDepth.from_image(image, meters_per_unit=0.001)
    np.testing.assert_allclose(prompt_depth.meters, [[1.0, 2.5]])
    assert prompt_depth.meters.dtype == np.float32


@pytest.mark.parametrize(
    ("meters", "message"),
    [
        (np.zeros((2,), dtype=np.float32), "2-D"),
        (np.zeros((0, 2), dtype=np.float32), "2-D"),
        (np.array([[1.0, np.nan]], dtype=np.float32), "finite"),
        (np.array([[-1.0, 1.0]], dtype=np.float32), "negative"),
        (np.ones((2, 2), dtype=np.float32), "constant"),
    ],
)
def test_invalid_meters_raise(meters: FloatArray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PromptDepth(meters)


def test_from_image_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="one channel"):
        PromptDepth.from_image(Image.new("RGB", (2, 2)), meters_per_unit=0.001)
    with pytest.raises(ValueError, match="meters_per_unit"):
        PromptDepth.from_image(Image.new("L", (2, 2)), meters_per_unit=0.0)
