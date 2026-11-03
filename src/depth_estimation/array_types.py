"""
Shared NumPy array type aliases for depth outputs.
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type ByteArray = NDArray[np.uint8]
