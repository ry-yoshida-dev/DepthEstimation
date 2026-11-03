from collections.abc import Sequence
from typing import ClassVar

from ..options import DepthBackend
from ..prompt_depth import PromptDepth
from .core import TransformersDepthEstimator


class PromptDepthAnythingEstimator(TransformersDepthEstimator):
    """
    Prompt Depth Anything, which turns a prompt depth into model input next to the image.

    ``PromptDepth`` holds meters already, so the processor's default millimeter scale is replaced.
    """

    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset({DepthBackend.PROMPT_DEPTH_ANYTHING})

    def _processor_arguments(self, prompt_depths: Sequence[PromptDepth | None]) -> dict[str, object]:
        return {
            "prompt_depth": [prompt_depth.meters for prompt_depth in prompt_depths if prompt_depth is not None],
            "prompt_scale_to_meter": 1.0,
        }
