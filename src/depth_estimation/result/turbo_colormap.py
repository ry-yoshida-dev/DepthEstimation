from typing import ClassVar

import numpy as np

from ..array_types import ByteArray, FloatArray


class TurboColormap:
    """
    Google's Turbo colormap in the polynomial approximation of its author, so no plotting library is needed.
    """

    COEFFICIENTS: ClassVar[tuple[tuple[float, ...], ...]] = (
        (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
        (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
        (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
    )

    @classmethod
    def apply(cls, values: FloatArray) -> ByteArray:
        """
        Map values in ``[0, 1]`` to colors, from dark blue through green to dark red.

        Parameters
        ----------
        values : FloatArray
            Values of any shape; values outside ``[0, 1]`` are clipped.

        Returns
        -------
        ByteArray
            RGB colors with shape ``values.shape + (3,)``.
        """
        clipped: FloatArray = np.clip(values, 0.0, 1.0)
        powers: FloatArray = np.stack([clipped**exponent for exponent in range(6)], axis=-1)
        coefficients: FloatArray = np.asarray(cls.COEFFICIENTS, dtype=np.float32)
        colors: FloatArray = np.clip(powers @ coefficients.T, 0.0, 1.0)
        return (colors * 255.0 + 0.5).astype(np.uint8)
