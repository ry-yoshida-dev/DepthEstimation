import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, TextIO

import numpy as np

from ..options import DepthBackend, DepthKind, Device, Precision
from ..result import DepthMap
from ..settings import EstimatorSettings, PresetCatalog
from .image_file_stream import ImageFileStream
from .image_files import ImageFiles


class EstimateCommand:
    """
    Command line estimating the depth of image files and directories.

    Settings come from a packaged preset (``--preset depth_anything/v2_small``), a YAML file in the preset format
    (``--config``) or a backend, depth kind and weights (``--backend``, ``--depth-kind``, ``--weights``);
    ``--batch-size``, ``--device`` and ``--precision`` override them. Each image gets ``<stem>.npy`` with the raw depth
    values and ``<stem>.png`` with the depth in the Turbo colormap in the output directory. Images are read one
    mini-batch at a time, upright as their EXIF orientation says; files that cannot be used are reported on the error
    stream and skipped. Invalid arguments are reported before the model is loaded.

    Prompt Depth Anything needs a prompt depth per image, so it is available through the Python API only.

    Installed as the ``depth-estimate`` script.
    """

    PRESET_SEPARATOR: ClassVar[str] = "/"

    @classmethod
    def main(cls, arguments: Sequence[str] | None = None) -> None:
        """
        Run the command, printing results to standard output and warnings to standard error.

        Parameters
        ----------
        arguments : Sequence[str] | None
            Command-line arguments without the program name; ``None`` reads ``sys.argv``.
        """
        cls.run(sys.argv[1:] if arguments is None else arguments, sys.stdout, sys.stderr)

    @classmethod
    def run(cls, arguments: Sequence[str], output: TextIO, warnings: TextIO) -> None:
        """
        Parse the arguments, estimate the depth of every image and write the files of each one.

        Parameters
        ----------
        arguments : Sequence[str]
            Command-line arguments without the program name.
        output : TextIO
            Stream one summary line per image is written to.
        warnings : TextIO
            Stream skipped image files are reported on.

        Raises
        ------
        SystemExit
            If the arguments are invalid, e.g. a missing path, an unknown preset or invalid settings.
        """
        parser: argparse.ArgumentParser = cls._parser()
        namespace: argparse.Namespace = parser.parse_args(arguments)
        if namespace.list_presets:
            cls._write_presets(output)
            return
        if not namespace.images or namespace.output_directory is None:
            parser.error("images and --output-directory are required.")
        try:
            image_paths: list[Path] = ImageFiles.find([Path(path) for path in namespace.images], namespace.recursive)
            cls._validate_output_names(image_paths)
            settings: EstimatorSettings = cls._settings(parser, namespace)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error.args[0]) if error.args else repr(error))
        output_directory: Path = namespace.output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        images: ImageFileStream = ImageFileStream(image_paths, warnings)
        for depth_map in settings.build().iter_estimate_images(images):
            path: Path = images.pending_paths.popleft()
            np.save(output_directory / f"{path.stem}.npy", depth_map.values)
            depth_map.to_colored_image().save(output_directory / f"{path.stem}.png")
            output.write(cls._summary(path, depth_map))
            output.flush()
        if images.skipped_paths:
            warnings.write(f"skipped {len(images.skipped_paths)} of {len(image_paths)} image files.\n")

    @classmethod
    def _parser(cls) -> argparse.ArgumentParser:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="depth-estimate", description="Estimate the depth of images with any supported model."
        )
        parser.add_argument("images", nargs="*", help="image files or directories")
        parser.add_argument("--output-directory", type=Path, help="directory receiving <stem>.npy and <stem>.png")
        parser.add_argument("--recursive", action="store_true", help="search sub-directories too")
        parser.add_argument("--preset", help="packaged preset as backend/name, e.g. depth_anything/v2_small")
        parser.add_argument("--config", type=Path, help="YAML file in the preset format")
        parser.add_argument("--weights", help="Hugging Face Hub model id or local checkpoint; needs --backend")
        parser.add_argument("--backend", type=DepthBackend, choices=list(DepthBackend))
        parser.add_argument("--depth-kind", type=DepthKind, choices=list(DepthKind), help="what --weights outputs")
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--device", type=Device, choices=list(Device))
        parser.add_argument("--precision", type=Precision, choices=list(Precision))
        parser.add_argument("--list-presets", action="store_true", help="print the packaged presets and exit")
        return parser

    @classmethod
    def _settings(cls, parser: argparse.ArgumentParser, namespace: argparse.Namespace) -> EstimatorSettings:
        sources: list[object] = [namespace.preset, namespace.config, namespace.weights]
        if sum(source is not None for source in sources) != 1:
            parser.error("exactly one of --preset, --config or --weights is required.")
        if namespace.weights is None and (namespace.backend is not None or namespace.depth_kind is not None):
            parser.error("--backend and --depth-kind only apply to --weights.")
        settings: EstimatorSettings
        if namespace.preset is not None:
            backend_name, _, preset_name = str(namespace.preset).partition(cls.PRESET_SEPARATOR)
            settings = PresetCatalog.load(DepthBackend(backend_name), preset_name)
        elif namespace.config is not None:
            settings = PresetCatalog.load_file(namespace.config)
        else:
            if namespace.backend is None or namespace.depth_kind is None:
                parser.error("--weights needs --backend and --depth-kind.")
            settings = EstimatorSettings(
                backend=namespace.backend, weights_path=str(namespace.weights), depth_kind=namespace.depth_kind
            )
        if settings.backend.is_prompt_depth_required:
            raise ValueError(f"{settings.backend} needs a prompt depth per image; use the Python API.")
        return replace(
            settings,
            batch_size=settings.batch_size if namespace.batch_size is None else namespace.batch_size,
            device=settings.device if namespace.device is None else namespace.device,
            precision=settings.precision if namespace.precision is None else namespace.precision,
        )

    @staticmethod
    def _validate_output_names(image_paths: Sequence[Path]) -> None:
        stem_counts: Counter[str] = Counter(path.stem for path in image_paths)
        repeated_stems: list[str] = sorted(stem for stem, count in stem_counts.items() if count > 1)
        if repeated_stems:
            raise ValueError(f"images would overwrite each other's output, sharing the names {repeated_stems}.")

    @staticmethod
    def _summary(path: Path, depth_map: DepthMap) -> str:
        return (
            f"{path}: {depth_map.width}x{depth_map.height} {depth_map.kind} "
            + f"[{depth_map.values.min():.3f}, {depth_map.values.max():.3f}]\n"
        )

    @classmethod
    def _write_presets(cls, output: TextIO) -> None:
        names: list[str] = [
            f"{backend.value}{cls.PRESET_SEPARATOR}{name}"
            for backend in DepthBackend
            for name in PresetCatalog.names(backend)
        ]
        output.write("\n".join(names) + "\n")
