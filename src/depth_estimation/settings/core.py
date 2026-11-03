from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from ..options import DepthBackend, DepthKind, Device, Precision
from .config_values import ConfigValues

if TYPE_CHECKING:
    from ..estimator import DepthEstimator


@dataclass(frozen=True, kw_only=True)
class EstimatorSettings:
    """
    Everything needed to load and run a depth estimator, identical for every backend.

    The fields map one-to-one onto the ``estimator`` section of the YAML presets in this package (enums written by
    value), read by ``from_mapping`` or ``PresetCatalog``.

    Attributes
    ----------
    backend : DepthBackend
        Estimator family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory, in the safetensors format.
    depth_kind : DepthKind
        What the checkpoint outputs; one of ``backend.supported_depth_kinds``.
    batch_size : int
        Maximum number of images per forward pass.
    device : Device
        Device to run on.
    precision : Precision
        Floating-point type of the model weights; ``float16`` needs a GPU.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank, ``batch_size`` is not positive, or the backend cannot output ``depth_kind``.
    """

    backend: DepthBackend
    weights_path: str
    depth_kind: DepthKind
    batch_size: int = 4
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive. got {self.batch_size}")
        if self.depth_kind not in self.backend.supported_depth_kinds:
            raise ValueError(
                f"{self.backend} cannot output {self.depth_kind} depth. "
                + f"supported: {sorted(kind.value for kind in self.backend.supported_depth_kinds)}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "EstimatorSettings":
        """
        Build settings from plain values, e.g. the ``estimator`` section of a parsed YAML or JSON document.

        Enums are given by value (``backend: dpt``); omitted optional fields take their defaults.

        Parameters
        ----------
        values : Mapping[str, object]
            Field values keyed by field name.

        Returns
        -------
        EstimatorSettings
            Validated settings.

        Raises
        ------
        KeyError
            If a required field is missing or an unknown field is given.
        TypeError
            If a value has the wrong type.
        ValueError
            If an enum value is unknown or the settings are invalid.
        """
        config_values: ConfigValues = ConfigValues(values)
        config_values.validate_keys(frozenset(field.name for field in fields(cls)))
        return cls(
            backend=config_values.member("backend", DepthBackend),
            weights_path=config_values.text("weights_path"),
            depth_kind=config_values.member("depth_kind", DepthKind),
            batch_size=config_values.integer("batch_size", 4),
            device=config_values.optional_member("device", Device, Device.AUTO),
            precision=config_values.optional_member("precision", Precision, Precision.FLOAT32),
        )

    def build(self) -> "DepthEstimator":
        """
        Load the estimator of ``backend`` with these settings; only the module of ``backend`` is imported.

        Returns
        -------
        DepthEstimator
            Loaded estimator.

        Raises
        ------
        ValueError
            If the checkpoint belongs to another backend or contradicts ``depth_kind``.
        OSError
            If the checkpoint cannot be found or has no safetensors weights.
        """
        match self.backend:
            case DepthBackend.DEPTH_ANYTHING | DepthBackend.DPT | DepthBackend.GLPN | DepthBackend.TIPSV2_DPT:
                from ..backends.core import TransformersDepthEstimator

                return TransformersDepthEstimator(self)
            case DepthBackend.ZOEDEPTH:
                from ..backends.zoedepth import ZoeDepthEstimator

                return ZoeDepthEstimator(self)
            case DepthBackend.DEPTH_PRO:
                from ..backends.depth_pro import DepthProEstimator

                return DepthProEstimator(self)
            case DepthBackend.PROMPT_DEPTH_ANYTHING:
                from ..backends.prompt_depth_anything import PromptDepthAnythingEstimator

                return PromptDepthAnythingEstimator(self)
