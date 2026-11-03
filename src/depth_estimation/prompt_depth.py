from dataclasses import dataclass

import numpy as np
from PIL import Image

from .array_types import FloatArray


@dataclass(frozen=True, eq=False)
class PromptDepth:
    """
    Low-resolution metric depth of one image, e.g. from a LiDAR, giving Prompt Depth Anything its scale.

    The model maps its output onto the range between the lowest and highest prompt value, so pixels without a
    measurement (often stored as zero) must be filled in before building the prompt.

    Attributes
    ----------
    meters : FloatArray
        Distances in meters with shape (height, width).

    Raises
    ------
    ValueError
        If ``meters`` is not a non-empty 2-D array, holds non-finite or negative values, or is constant.
    """

    meters: FloatArray

    def __post_init__(self) -> None:
        if self.meters.ndim != 2 or self.meters.size == 0:
            raise ValueError(f"meters must be a non-empty 2-D array. got shape {self.meters.shape}")
        if not np.isfinite(self.meters).all():
            raise ValueError("meters must be finite.")
        lowest: float = float(self.meters.min())
        if lowest < 0.0:
            raise ValueError(f"meters must not be negative. got {lowest}")
        if lowest == float(self.meters.max()):
            raise ValueError(f"meters must not be constant, since their range sets the scale. got only {lowest}")

    @classmethod
    def from_image(cls, image: Image.Image, meters_per_unit: float) -> "PromptDepth":
        """
        Read a single-channel depth image, e.g. the 16-bit millimeter PNG of a phone LiDAR.

        Parameters
        ----------
        image : Image.Image
            Single-channel image holding depth values.
        meters_per_unit : float
            Meters per stored unit, e.g. ``0.001`` for millimeters.

        Returns
        -------
        PromptDepth
            Depth in meters.

        Raises
        ------
        ValueError
            If the image has several channels, ``meters_per_unit`` is not positive, or the depth is invalid.
        """
        if meters_per_unit <= 0.0:
            raise ValueError(f"meters_per_unit must be positive. got {meters_per_unit}")
        if len(image.getbands()) != 1:
            raise ValueError(f"a depth image must have one channel. got mode {image.mode}")
        return cls(np.asarray(image, dtype=np.float32) * np.float32(meters_per_unit))
