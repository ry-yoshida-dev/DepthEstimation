from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..array_types import ByteArray, FloatArray
from ..options import DepthKind
from .turbo_colormap import TurboColormap


@dataclass(frozen=True, eq=False)
class DepthMap:
    """
    Depth of one image at the resolution of that image, the same type for every model.

    Attributes
    ----------
    values : FloatArray
        Depth with shape (height, width); ``kind`` says what the values mean.
    kind : DepthKind
        Meters, or inverse depth up to an unknown scale and shift.
    focal_length_pixels : float | None
        Horizontal focal length in pixels of the source image, where the model estimates one.
    field_of_view_degrees : float | None
        Horizontal field of view in degrees, where the model estimates one.

    Raises
    ------
    ValueError
        If ``values`` is not a non-empty 2-D array.
    """

    values: FloatArray
    kind: DepthKind
    focal_length_pixels: float | None = None
    field_of_view_degrees: float | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.size == 0:
            raise ValueError(f"values must be a non-empty 2-D array. got shape {self.values.shape}")

    @property
    def height(self) -> int:
        """
        Height of the depth map.

        Returns
        -------
        int
            Height in pixels.
        """
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        """
        Width of the depth map.

        Returns
        -------
        int
            Width in pixels.
        """
        return int(self.values.shape[1])

    def nearness(self) -> FloatArray:
        """
        Inverse depth stretched to ``[0, 1]``, comparable in direction across kinds, for visualization and ranking.

        Metric depth is inverted first, so near structure keeps its detail as in the output of relative models. The
        stretch is per image, so values of different images are not comparable.

        Returns
        -------
        FloatArray
            Values with the shape of ``values``; ``1`` is the nearest pixel. Pixels without a valid depth, and images
            of one constant depth, are ``0``.
        """
        inverse_depth: FloatArray
        match self.kind:
            case DepthKind.METRIC:
                inverse_depth = np.full_like(self.values, np.nan)
                is_positive: NDArray[np.bool_] = self.values > 0.0
                inverse_depth[is_positive] = 1.0 / self.values[is_positive]
            case DepthKind.RELATIVE_INVERSE:
                inverse_depth = self.values
        is_valid: NDArray[np.bool_] = np.isfinite(inverse_depth)
        if not is_valid.any():
            return np.zeros_like(self.values)
        lowest: np.float32 = inverse_depth[is_valid].min()
        value_range: np.float32 = inverse_depth[is_valid].max() - lowest
        if value_range <= 0.0:
            return np.zeros_like(self.values)
        stretched: FloatArray = (inverse_depth - lowest) / value_range
        return np.where(is_valid, stretched, np.float32(0.0))

    def to_grayscale_image(self) -> Image.Image:
        """
        Render the nearness as an 8-bit image; brighter is nearer.

        Returns
        -------
        Image.Image
            Image of mode ``L``.
        """
        pixels: ByteArray = (self.nearness() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(pixels)

    def to_colored_image(self) -> Image.Image:
        """
        Render the nearness in the Turbo colormap; red is near and blue is far.

        Returns
        -------
        Image.Image
            Image of mode ``RGB``.
        """
        return Image.fromarray(TurboColormap.apply(self.nearness()))
