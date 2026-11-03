import numpy as np
import pytest
from PIL import Image

from depth_estimation import DepthKind, DepthMap
from depth_estimation.array_types import ByteArray, FloatArray
from depth_estimation.result import TurboColormap


def array(*rows: list[float]) -> FloatArray:
    return np.array(rows, dtype=np.float32)


def test_size_and_shape_validation() -> None:
    depth_map: DepthMap = DepthMap(values=np.zeros((2, 3), dtype=np.float32), kind=DepthKind.METRIC)
    assert (depth_map.height, depth_map.width) == (2, 3)
    with pytest.raises(ValueError, match="2-D"):
        DepthMap(values=np.zeros((3,), dtype=np.float32), kind=DepthKind.METRIC)


def test_metric_nearness_inverts_depth() -> None:
    nearness: FloatArray = DepthMap(values=array([1.0, 2.0, 4.0]), kind=DepthKind.METRIC).nearness()
    np.testing.assert_allclose(nearness, [[1.0, 1.0 / 3.0, 0.0]], atol=1e-6)


def test_relative_nearness_keeps_direction() -> None:
    nearness: FloatArray = DepthMap(values=array([10.0, 20.0, 30.0]), kind=DepthKind.RELATIVE_INVERSE).nearness()
    np.testing.assert_allclose(nearness, [[0.0, 0.5, 1.0]])


def test_metric_nearness_gives_invalid_depth_zero() -> None:
    nearness: FloatArray = DepthMap(values=array([0.0, 1.0, 2.0, np.nan]), kind=DepthKind.METRIC).nearness()
    np.testing.assert_allclose(nearness, [[0.0, 1.0, 0.0, 0.0]])


@pytest.mark.parametrize("values", [array([3.0, 3.0]), array([0.0, 0.0])])
def test_flat_or_invalid_maps_have_zero_nearness(values: FloatArray) -> None:
    assert not DepthMap(values=values, kind=DepthKind.METRIC).nearness().any()


def test_rendered_images_match_the_map() -> None:
    depth_map: DepthMap = DepthMap(values=array([1.0, 2.0], [4.0, 8.0]), kind=DepthKind.METRIC)
    grayscale: Image.Image = depth_map.to_grayscale_image()
    assert (grayscale.mode, grayscale.size) == ("L", (2, 2))
    assert grayscale.getpixel((0, 0)) == 255
    assert grayscale.getpixel((1, 1)) == 0
    colored: Image.Image = depth_map.to_colored_image()
    assert (colored.mode, colored.size) == ("RGB", (2, 2))


def test_turbo_runs_from_blue_through_green_to_red() -> None:
    colors: ByteArray = TurboColormap.apply(np.array([0.1, 0.5, 0.9, -1.0, 2.0], dtype=np.float32))
    assert colors.shape == (5, 3)
    assert colors.dtype == np.uint8
    assert colors[0].argmax() == 2
    assert colors[1].argmax() == 1
    assert colors[2].argmax() == 0
    np.testing.assert_array_equal(colors[3], TurboColormap.apply(np.zeros(1, dtype=np.float32))[0])
