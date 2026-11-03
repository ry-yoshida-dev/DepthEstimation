from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    DepthProImageProcessor,
    DPTImageProcessor,
    PromptDepthAnythingImageProcessor,
    ZoeDepthImageProcessor,
)
from transformers.modeling_outputs import DepthEstimatorOutput

from depth_estimation import DepthBackend, DepthKind, DepthMap, Device, EstimatorSettings, Precision, PromptDepth
from depth_estimation.backends.checkpoint_config import CheckpointConfig
from depth_estimation.backends.core import TransformersDepthEstimator
from depth_estimation.backends.depth_pro import DepthProEstimator
from depth_estimation.backends.depth_processor import DepthProcessor
from depth_estimation.backends.prompt_depth_anything import PromptDepthAnythingEstimator
from depth_estimation.backends.zoedepth import ZoeDepthEstimator
from depth_estimation.runtime import TorchRuntime


class BrightnessDepthModel(torch.nn.Module):
    """Outputs each image's mean brightness as its depth, recording the shape of every forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_shapes: list[torch.Size] = []
        self.received_inputs: list[dict[str, torch.Tensor]] = []
        self.field_of_view: float | None = None

    def forward(self, pixel_values: torch.Tensor, **inputs: torch.Tensor) -> DepthEstimatorOutput:
        self.batch_shapes.append(pixel_values.shape)
        self.received_inputs.append(inputs)
        batch_size, _, height, width = pixel_values.shape
        brightness: torch.Tensor = pixel_values.mean(dim=(1, 2, 3)).view(batch_size, 1, 1) + 10.0
        output: DepthEstimatorOutput = DepthEstimatorOutput(predicted_depth=brightness.expand(-1, height, width))
        if self.field_of_view is not None:
            output["field_of_view"] = torch.full((batch_size,), self.field_of_view, dtype=pixel_values.dtype)
        return output


def build_estimator[EstimatorT: TransformersDepthEstimator](
    estimator_type: type[EstimatorT],
    backend: DepthBackend,
    processor: object,
    precision: Precision = Precision.FLOAT32,
) -> tuple[EstimatorT, BrightnessDepthModel]:
    settings: EstimatorSettings = EstimatorSettings(
        backend=backend, weights_path="stub", depth_kind=DepthKind.METRIC, device=Device.CPU, precision=precision
    )
    estimator: EstimatorT = object.__new__(estimator_type)
    model: BrightnessDepthModel = BrightnessDepthModel()
    estimator.settings = settings
    estimator._runtime = TorchRuntime(settings)
    estimator._processor = cast(DepthProcessor, processor)
    estimator._model = model
    return estimator, model


def aspect_preserving_processor() -> DPTImageProcessor:
    return DPTImageProcessor(size={"height": 28, "width": 28}, keep_aspect_ratio=True, ensure_multiple_of=14)


def test_images_of_one_input_shape_share_a_forward_pass_and_order_is_kept() -> None:
    estimator, model = build_estimator(TransformersDepthEstimator, DepthBackend.DPT, aspect_preserving_processor())
    images: list[Image.Image] = [
        Image.new("RGB", (40, 40), (10, 10, 10)),
        Image.new("RGB", (80, 40), (90, 90, 90)),
        Image.new("RGB", (40, 40), (200, 200, 200)),
    ]
    depth_maps: list[DepthMap] = estimator._estimate_mini_batch(images, [None, None, None])
    assert sorted(shape[0] for shape in model.batch_shapes) == [1, 2]
    assert [depth_map.values.shape for depth_map in depth_maps] == [(40, 40), (40, 80), (40, 40)]
    means: list[float] = [float(depth_map.values.mean()) for depth_map in depth_maps]
    assert means[0] < means[1] < means[2]
    assert all(depth_map.values.dtype == np.float32 for depth_map in depth_maps)


def test_half_precision_output_becomes_float32() -> None:
    estimator, _ = build_estimator(
        TransformersDepthEstimator, DepthBackend.DPT, aspect_preserving_processor(), Precision.BFLOAT16
    )
    depth_map: DepthMap = estimator._estimate_mini_batch([Image.new("RGB", (30, 20))], [None])[0]
    assert depth_map.values.dtype == np.float32
    assert depth_map.values.shape == (20, 30)


def test_zoedepth_removes_its_padding_and_returns_the_image_size() -> None:
    estimator, _ = build_estimator(ZoeDepthEstimator, DepthBackend.ZOEDEPTH, ZoeDepthImageProcessor())
    depth_map: DepthMap = estimator._estimate_mini_batch([Image.new("RGB", (64, 48), (50, 50, 50))], [None])[0]
    assert depth_map.values.shape == (48, 64)


def test_depth_pro_scales_by_focal_length_and_reports_camera() -> None:
    estimator, model = build_estimator(DepthProEstimator, DepthBackend.DEPTH_PRO, DepthProImageProcessor())
    model.field_of_view = 90.0
    depth_map: DepthMap = estimator._estimate_mini_batch([Image.new("RGB", (64, 48))], [None])[0]
    assert depth_map.values.shape == (48, 64)
    assert depth_map.field_of_view_degrees == pytest.approx(90.0)
    assert depth_map.focal_length_pixels == pytest.approx(32.0)


def test_prompt_depth_reaches_the_model_in_meters() -> None:
    estimator, model = build_estimator(
        PromptDepthAnythingEstimator, DepthBackend.PROMPT_DEPTH_ANYTHING, PromptDepthAnythingImageProcessor()
    )
    meters: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.array([[1.5, 3.0]], dtype=np.float32)
    estimator._estimate_mini_batch([Image.new("RGB", (28, 28))], [PromptDepth(meters)])
    prompt_tensor: torch.Tensor = model.received_inputs[0]["prompt_depth"]
    assert prompt_tensor.shape == (1, 1, 1, 2)
    assert prompt_tensor.flatten().tolist() == pytest.approx([1.5, 3.0])


def test_images_with_differently_sized_prompts_do_not_share_a_forward_pass() -> None:
    estimator, model = build_estimator(
        PromptDepthAnythingEstimator, DepthBackend.PROMPT_DEPTH_ANYTHING, PromptDepthAnythingImageProcessor()
    )
    prompts: list[PromptDepth | None] = [
        PromptDepth(np.array([[1.0, 2.0]], dtype=np.float32)),
        PromptDepth(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)),
    ]
    estimator._estimate_mini_batch([Image.new("RGB", (28, 28)), Image.new("RGB", (28, 28))], prompts)
    assert len(model.batch_shapes) == 2


def tiny_settings(directory: Path, backend: DepthBackend, depth_kind: DepthKind) -> EstimatorSettings:
    return EstimatorSettings(backend=backend, weights_path=str(directory), depth_kind=depth_kind, device=Device.CPU)


def test_real_checkpoint_loads_and_estimates_images_of_different_sizes(tiny_depth_anything_directory: Path) -> None:
    settings: EstimatorSettings = tiny_settings(
        tiny_depth_anything_directory, DepthBackend.DEPTH_ANYTHING, DepthKind.RELATIVE_INVERSE
    )
    depth_maps: list[DepthMap] = settings.build().estimate_images(
        [Image.new("RGB", (50, 30)), Image.new("L", (20, 60))]
    )
    assert [depth_map.values.shape for depth_map in depth_maps] == [(30, 50), (60, 20)]
    assert all(np.isfinite(depth_map.values).all() for depth_map in depth_maps)
    assert all(depth_map.kind is DepthKind.RELATIVE_INVERSE for depth_map in depth_maps)


def test_checkpoint_of_another_backend_is_rejected(tiny_depth_anything_directory: Path) -> None:
    settings: EstimatorSettings = tiny_settings(tiny_depth_anything_directory, DepthBackend.DPT, DepthKind.METRIC)
    with pytest.raises(ValueError, match="model_type"):
        settings.build()


def test_depth_anything_checkpoint_of_another_kind_is_rejected(tiny_depth_anything_directory: Path) -> None:
    settings: EstimatorSettings = tiny_settings(
        tiny_depth_anything_directory, DepthBackend.DEPTH_ANYTHING, DepthKind.METRIC
    )
    with pytest.raises(ValueError, match="outputs relative_inverse"):
        settings.build()


@pytest.mark.parametrize(
    ("depth_estimation_type", "depth_kind", "is_valid"),
    [
        (None, DepthKind.RELATIVE_INVERSE, True),
        ("relative", DepthKind.RELATIVE_INVERSE, True),
        ("metric", DepthKind.METRIC, True),
        ("metric", DepthKind.RELATIVE_INVERSE, False),
        ("unknown", DepthKind.METRIC, False),
    ],
)
def test_depth_anything_kinds(depth_estimation_type: str | None, depth_kind: DepthKind, is_valid: bool) -> None:
    config: CheckpointConfig = CheckpointConfig("depth_anything", depth_estimation_type)
    settings: EstimatorSettings = EstimatorSettings(
        backend=DepthBackend.DEPTH_ANYTHING, weights_path="stub", depth_kind=depth_kind
    )
    if is_valid:
        config.validate(settings)
    else:
        with pytest.raises(ValueError, match="outputs"):
            config.validate(settings)


def test_pickle_only_checkpoint_is_refused(tiny_depth_anything_directory: Path, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text((tiny_depth_anything_directory / "config.json").read_text(), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(
        (tiny_depth_anything_directory / "preprocessor_config.json").read_text(), encoding="utf-8"
    )
    (tmp_path / "pytorch_model.bin").write_bytes(b"must never be unpickled")
    settings: EstimatorSettings = tiny_settings(tmp_path, DepthBackend.DEPTH_ANYTHING, DepthKind.RELATIVE_INVERSE)
    with pytest.raises(OSError, match="safetensors"):
        settings.build()
