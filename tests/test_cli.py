import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from depth_estimation import DepthBackend, DepthEstimator, DepthKind, DepthMap, EstimatorSettings, PromptDepth
from depth_estimation.cli import EstimateCommand


class StubEstimator(DepthEstimator):
    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset(DepthBackend)
    received_settings: ClassVar[list[EstimatorSettings]] = []

    def _estimate_mini_batch(
        self, images: Sequence[Image.Image], prompt_depths: Sequence[PromptDepth | None]
    ) -> list[DepthMap]:
        return [
            DepthMap(
                values=np.arange(image.height * image.width, dtype=np.float32).reshape(image.height, image.width) + 1.0,
                kind=self.settings.depth_kind,
            )
            for image in images
        ]


@pytest.fixture(autouse=True)
def stub_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(settings: EstimatorSettings) -> DepthEstimator:
        StubEstimator.received_settings.append(settings)
        return StubEstimator(settings)

    StubEstimator.received_settings.clear()
    monkeypatch.setattr(EstimatorSettings, "build", build)


def write_images(directory: Path, names: Iterable[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (4, 2)).save(directory / name)


def run(*arguments: str | Path) -> tuple[str, str]:
    output: io.StringIO = io.StringIO()
    warnings: io.StringIO = io.StringIO()
    EstimateCommand.run([str(argument) for argument in arguments], output, warnings)
    return output.getvalue(), warnings.getvalue()


def test_list_presets() -> None:
    output, _ = run("--list-presets")
    assert "depth_anything/v2_small" in output.splitlines()
    assert "depth_pro/base" in output.splitlines()


def test_writes_values_and_image_of_every_file(tmp_path: Path) -> None:
    write_images(tmp_path / "in", ["a.png", "b.jpg"])
    output, warnings = run(
        tmp_path / "in", "--preset", "depth_anything/v2_small", "--output-directory", tmp_path / "out"
    )
    assert warnings == ""
    assert len(output.splitlines()) == 2
    assert "relative_inverse [1.000, 8.000]" in output.splitlines()[0]
    assert np.load(tmp_path / "out" / "a.npy").shape == (2, 4)
    with Image.open(tmp_path / "out" / "b.png") as image:
        assert (image.mode, image.size) == ("RGB", (4, 2))


def test_corrupt_and_high_bit_depth_files_are_skipped(tmp_path: Path) -> None:
    write_images(tmp_path, ["good.png"])
    (tmp_path / "broken.png").write_bytes(b"not an image")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(tmp_path / "deep.png")
    output, warnings = run(tmp_path, "--preset", "dpt/vit_large", "--output-directory", tmp_path / "out")
    assert "good.png" in output
    assert len(output.splitlines()) == 1
    assert "skipped 2 of 3" in warnings
    assert (tmp_path / "out" / "good.npy").exists()
    assert not (tmp_path / "out" / "deep.npy").exists()


def test_overrides_reach_the_settings(tmp_path: Path) -> None:
    write_images(tmp_path, ["a.png"])
    run(
        tmp_path / "a.png",
        "--weights",
        "some/model",
        "--backend",
        "dpt",
        "--depth-kind",
        "metric",
        "--batch-size",
        "3",
        "--device",
        "cpu",
        "--precision",
        "bfloat16",
        "--output-directory",
        tmp_path / "out",
    )
    settings: EstimatorSettings = StubEstimator.received_settings[0]
    assert (settings.weights_path, settings.batch_size, settings.depth_kind) == ("some/model", 3, DepthKind.METRIC)


def test_config_file_is_read(tmp_path: Path) -> None:
    write_images(tmp_path, ["a.png"])
    config: Path = tmp_path / "settings.yaml"
    config.write_text("estimator:\n  backend: glpn\n  weights_path: local\n  depth_kind: metric\n", encoding="utf-8")
    run(tmp_path / "a.png", "--config", config, "--output-directory", tmp_path / "out")
    assert StubEstimator.received_settings[0].backend is DepthBackend.GLPN


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--preset", "dpt/vit_large"], "required"),
        (["IMAGE", "--output-directory", "OUT"], "exactly one"),
        (["IMAGE", "--output-directory", "OUT", "--preset", "dpt/missing"], "no preset"),
        (["IMAGE", "--output-directory", "OUT", "--preset", "unknown/x"], "not a valid DepthBackend"),
        (["IMAGE", "--output-directory", "OUT", "--preset", "dpt/vit_large", "--backend", "dpt"], "only apply"),
        (["IMAGE", "--output-directory", "OUT", "--weights", "x"], "needs --backend"),
        (["IMAGE", "--output-directory", "OUT", "--preset", "dpt/vit_large", "--batch-size", "0"], "batch_size"),
        (["IMAGE", "--output-directory", "OUT", "--preset", "prompt_depth_anything/small"], "Python API"),
        (["MISSING", "--output-directory", "OUT", "--preset", "dpt/vit_large"], "no image file"),
    ],
)
def test_invalid_arguments_stop_before_loading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    write_images(tmp_path, ["image.png"])
    replaced: dict[str, str] = {
        "IMAGE": str(tmp_path / "image.png"),
        "OUT": str(tmp_path / "out"),
        "MISSING": str(tmp_path / "missing.png"),
    }
    with pytest.raises(SystemExit):
        run(*[replaced.get(argument, argument) for argument in arguments])
    assert message in capsys.readouterr().err
    assert StubEstimator.received_settings == []


def test_images_sharing_an_output_name_are_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_images(tmp_path, ["a.png", "a.jpg"])
    with pytest.raises(SystemExit):
        run(tmp_path, "--preset", "dpt/vit_large", "--output-directory", tmp_path / "out")
    assert "overwrite" in capsys.readouterr().err
