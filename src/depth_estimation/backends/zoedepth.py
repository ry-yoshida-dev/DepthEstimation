from collections.abc import Sequence
from typing import ClassVar

from transformers.modeling_outputs import DepthEstimatorOutput

from ..options import DepthBackend
from .core import TransformersDepthEstimator
from .prediction import Prediction


class ZoeDepthEstimator(TransformersDepthEstimator):
    """
    ZoeDepth, whose processor pads images against boundary artifacts, so the padding is cut from its output.
    """

    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset({DepthBackend.ZOEDEPTH})

    def _post_process(self, outputs: DepthEstimatorOutput, image_sizes: Sequence[tuple[int, int]]) -> list[Prediction]:
        return self._processor.post_process_depth_estimation(
            outputs, source_sizes=list(image_sizes), target_sizes=list(image_sizes)
        )
