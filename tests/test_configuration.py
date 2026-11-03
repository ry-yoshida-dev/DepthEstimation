from dataclasses import fields
from pathlib import Path

import pytest
import yaml

import depth_estimation
from depth_estimation import DepthBackend, DepthKind, Device, EstimatorSettings, Precision, PresetCatalog

PRESET_DIRECTORY: Path = Path(depth_estimation.__file__).parent / "config"
PRESET_PATHS: list[Path] = sorted(PRESET_DIRECTORY.glob("*/*.yaml"))


def test_every_backend_has_presets() -> None:
    assert {path.parent.name for path in PRESET_PATHS} == {backend.value for backend in DepthBackend}


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_preset_loads_and_lists_every_field(path: Path) -> None:
    section: object = yaml.safe_load(path.read_text(encoding="utf-8"))["estimator"]
    assert isinstance(section, dict)
    assert set(section) == {field.name for field in fields(EstimatorSettings)}
    settings: EstimatorSettings = PresetCatalog.load(DepthBackend(path.parent.name), path.stem)
    assert settings.backend.value == path.parent.name


def test_catalog_lists_names_and_rejects_unknown() -> None:
    assert "v2_small" in PresetCatalog.names(DepthBackend.DEPTH_ANYTHING)
    with pytest.raises(KeyError, match="no preset"):
        PresetCatalog.load(DepthBackend.DPT, "missing")


def test_catalog_loads_file(tmp_path: Path) -> None:
    path: Path = tmp_path / "settings.yaml"
    path.write_text("estimator:\n  backend: glpn\n  weights_path: local\n  depth_kind: metric\n", encoding="utf-8")
    settings: EstimatorSettings = PresetCatalog.load_file(path)
    assert settings == EstimatorSettings(backend=DepthBackend.GLPN, weights_path="local", depth_kind=DepthKind.METRIC)
    assert settings.device is Device.AUTO
    assert settings.precision is Precision.FLOAT32
    with pytest.raises(FileNotFoundError):
        PresetCatalog.load_file(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"backend": "dpt", "weights_path": "x"}, KeyError),
        ({"backend": "dpt", "weights_path": "x", "depth_kind": "metric", "typo": 1}, KeyError),
        ({"backend": "dpt", "weights_path": 1, "depth_kind": "metric"}, TypeError),
        ({"backend": "dpt", "weights_path": "x", "depth_kind": "metric", "batch_size": True}, TypeError),
        ({"backend": "unknown", "weights_path": "x", "depth_kind": "metric"}, ValueError),
        ({"backend": "dpt", "weights_path": "x", "depth_kind": "metric", "device": "tpu"}, ValueError),
    ],
)
def test_from_mapping_rejects_malformed_values(values: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        EstimatorSettings.from_mapping(values)


def test_invalid_settings_raise() -> None:
    with pytest.raises(ValueError, match="weights_path"):
        EstimatorSettings(backend=DepthBackend.DPT, weights_path=" ", depth_kind=DepthKind.METRIC)
    with pytest.raises(ValueError, match="batch_size"):
        EstimatorSettings(backend=DepthBackend.DPT, weights_path="x", depth_kind=DepthKind.METRIC, batch_size=0)
    with pytest.raises(ValueError, match="cannot output"):
        EstimatorSettings(backend=DepthBackend.ZOEDEPTH, weights_path="x", depth_kind=DepthKind.RELATIVE_INVERSE)


def test_backend_capabilities() -> None:
    both_kinds: frozenset[DepthKind] = frozenset(DepthKind)
    assert DepthBackend.DEPTH_ANYTHING.supported_depth_kinds == both_kinds
    assert DepthBackend.DPT.supported_depth_kinds == both_kinds
    assert DepthBackend.DEPTH_PRO.supported_depth_kinds == frozenset({DepthKind.METRIC})
    assert [backend for backend in DepthBackend if backend.is_prompt_depth_required] == [
        DepthBackend.PROMPT_DEPTH_ANYTHING
    ]
