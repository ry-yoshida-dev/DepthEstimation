from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from depth_estimation import DepthBackend, DepthEstimator, DepthKind, DepthMap, EstimatorSettings, PromptDepth


class StubEstimator(DepthEstimator):
    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset(DepthBackend)

    def __init__(self, settings: EstimatorSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []
        self.received_prompts: list[PromptDepth | None] = []
        self.received_modes: list[str] = []

    def _estimate_mini_batch(
        self, images: Sequence[Image.Image], prompt_depths: Sequence[PromptDepth | None]
    ) -> list[DepthMap]:
        self.mini_batch_sizes.append(len(images))
        self.received_prompts.extend(prompt_depths)
        self.received_modes.extend(image.mode for image in images)
        return [
            DepthMap(values=np.full((image.height, image.width), image.width, np.float32), kind=DepthKind.METRIC)
            for image in images
        ]


def build_estimator(backend: DepthBackend = DepthBackend.DPT, batch_size: int = 2) -> StubEstimator:
    return StubEstimator(
        EstimatorSettings(backend=backend, weights_path="stub", depth_kind=DepthKind.METRIC, batch_size=batch_size)
    )


def build_prompt() -> PromptDepth:
    return PromptDepth(np.array([[1.0, 2.0]], dtype=np.float32))


def test_estimate_images_batches_and_keeps_order() -> None:
    estimator: StubEstimator = build_estimator()
    depth_maps: list[DepthMap] = estimator.estimate_images([Image.new("RGB", (width, 2)) for width in (1, 2, 3, 4, 5)])
    assert estimator.mini_batch_sizes == [2, 2, 1]
    assert [depth_map.width for depth_map in depth_maps] == [1, 2, 3, 4, 5]


def test_images_reach_the_model_as_rgb() -> None:
    estimator: StubEstimator = build_estimator()
    estimator.estimate(Image.new("L", (2, 2)))
    assert estimator.received_modes == ["RGB"]


def test_stream_is_read_one_mini_batch_at_a_time() -> None:
    read_count: int = 0

    def stream() -> Iterator[Image.Image]:
        nonlocal read_count
        for _ in range(5):
            read_count += 1
            yield Image.new("RGB", (2, 2))

    depth_maps: Iterator[DepthMap] = build_estimator().iter_estimate_images(stream())
    assert read_count == 0
    next(depth_maps)
    assert read_count == 2


def test_empty_images_raise() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_estimator().estimate_images([])


def test_prompt_depth_is_rejected_when_not_accepted() -> None:
    estimator: StubEstimator = build_estimator()
    with pytest.raises(ValueError, match="does not accept"):
        estimator.estimate(Image.new("RGB", (2, 2)), build_prompt())
    assert estimator.mini_batch_sizes == []


def test_prompt_depth_is_required_by_prompt_depth_anything() -> None:
    estimator: StubEstimator = build_estimator(DepthBackend.PROMPT_DEPTH_ANYTHING)
    with pytest.raises(ValueError, match="needs a prompt depth"):
        estimator.iter_estimate_images([Image.new("RGB", (2, 2))])
    prompt: PromptDepth = build_prompt()
    estimator.estimate(Image.new("RGB", (2, 2)), prompt)
    assert estimator.received_prompts == [prompt]


def test_prompt_depth_count_must_match() -> None:
    estimator: StubEstimator = build_estimator(DepthBackend.PROMPT_DEPTH_ANYTHING)
    images: list[Image.Image] = [Image.new("RGB", (2, 2)) for _ in range(2)]
    with pytest.raises(ValueError, match="one prompt depth per image"):
        estimator.estimate_images(images, [build_prompt()])
    with pytest.raises(ValueError, match="shorter"):
        list(estimator.iter_estimate_images(images, [build_prompt()]))


def test_settings_of_another_backend_raise() -> None:
    class DptOnly(StubEstimator):
        BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset({DepthBackend.DPT})

    with pytest.raises(ValueError, match="needs settings"):
        DptOnly(EstimatorSettings(backend=DepthBackend.GLPN, weights_path="stub", depth_kind=DepthKind.METRIC))
