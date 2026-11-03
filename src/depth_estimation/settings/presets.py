from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import ClassVar, cast

import yaml

from ..options import DepthBackend
from .core import EstimatorSettings


class PresetCatalog:
    """
    Reads the YAML presets shipped with the package, and YAML files of the same format, into ``EstimatorSettings``.

    Presets live in ``config/<backend>/<name>.yaml`` inside the package, e.g. ``depth_anything/v2_small``; each holds
    the settings in its ``estimator`` section.
    """

    PACKAGE: ClassVar[str] = "depth_estimation"
    PRESET_DIRECTORY: ClassVar[str] = "config"
    PRESET_SUFFIX: ClassVar[str] = ".yaml"
    SECTION: ClassVar[str] = "estimator"

    @classmethod
    def names(cls, backend: DepthBackend) -> tuple[str, ...]:
        """
        Names of the presets of one backend.

        Parameters
        ----------
        backend : DepthBackend
            Backend whose presets are listed.

        Returns
        -------
        tuple[str, ...]
            Preset names in alphabetical order, e.g. ``("distill_large", ...)``.
        """
        return tuple(
            sorted(
                entry.name.removesuffix(cls.PRESET_SUFFIX)
                for entry in cls._backend_directory(backend).iterdir()
                if entry.name.endswith(cls.PRESET_SUFFIX)
            )
        )

    @classmethod
    def load(cls, backend: DepthBackend, name: str) -> EstimatorSettings:
        """
        Read a preset shipped with the package.

        Parameters
        ----------
        backend : DepthBackend
            Backend of the preset.
        name : str
            Preset name, e.g. ``"v2_small"``.

        Returns
        -------
        EstimatorSettings
            Settings of the preset.

        Raises
        ------
        KeyError
            If the backend has no preset of that name.
        """
        if name not in cls.names(backend):
            raise KeyError(f"{backend} has no preset {name!r}. available: {list(cls.names(backend))}")
        preset: Traversable = cls._backend_directory(backend) / f"{name}{cls.PRESET_SUFFIX}"
        return cls._parse(preset.read_text(encoding="utf-8"), source=f"{backend}/{name}")

    @classmethod
    def load_file(cls, path: Path) -> EstimatorSettings:
        """
        Read a YAML file in the preset format, e.g. a copied and edited preset.

        Parameters
        ----------
        path : Path
            YAML file with a ``estimator`` section.

        Returns
        -------
        EstimatorSettings
            Settings of the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(f"no settings file at {path}")
        return cls._parse(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def _backend_directory(cls, backend: DepthBackend) -> Traversable:
        return files(cls.PACKAGE) / cls.PRESET_DIRECTORY / backend.value

    @classmethod
    def _parse(cls, document_text: str, source: str) -> EstimatorSettings:
        document: object = yaml.safe_load(document_text)
        if not isinstance(document, Mapping):
            raise TypeError(f"{source} must hold a mapping. got {type(document).__name__}")
        section: object = cls._string_keyed(cast(Mapping[object, object], document)).get(cls.SECTION)
        if not isinstance(section, Mapping):
            raise KeyError(f"{source} must have a {cls.SECTION!r} mapping.")
        return EstimatorSettings.from_mapping(cls._string_keyed(cast(Mapping[object, object], section)))

    @staticmethod
    def _string_keyed(mapping: Mapping[object, object]) -> dict[str, object]:
        return {str(key): value for key, value in mapping.items()}
