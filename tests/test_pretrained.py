import numpy as np
import pytest
from PIL import Image

from depth_estimation import DepthBackend, DepthKind, DepthMap, PresetCatalog

pytestmark = pytest.mark.pretrained


def gradient_image() -> Image.Image:
    gradient: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.tile(
        np.linspace(240, 30, 240, dtype=np.uint8)[:, None], (1, 320)
    )
    return Image.fromarray(gradient).convert("RGB")


@pytest.mark.parametrize(
    ("backend", "name", "kind"),
    [
        (DepthBackend.DEPTH_ANYTHING, "v2_small", DepthKind.RELATIVE_INVERSE),
        (DepthBackend.DEPTH_ANYTHING, "v2_metric_indoor_small", DepthKind.METRIC),
        (DepthBackend.DPT, "swinv2_tiny_256", DepthKind.RELATIVE_INVERSE),
    ],
)
def test_real_model_returns_depth_at_image_size(backend: DepthBackend, name: str, kind: DepthKind) -> None:
    depth_map: DepthMap = PresetCatalog.load(backend, name).build().estimate(gradient_image())
    assert depth_map.values.shape == (240, 320)
    assert depth_map.kind is kind
    assert np.isfinite(depth_map.values).all()
    assert depth_map.values.std() > 0.0
