from typing import ClassVar

import torch
from PIL import Image

from ..options import DepthBackend
from ..result import DepthMap
from ..settings import EstimatorSettings
from .core import TransformersDepthEstimator
from .prediction import Prediction


class DepthProEstimator(TransformersDepthEstimator):
    """
    Apple Depth Pro, whose field of view head gives the focal length that turns its output into meters.
    """

    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset({DepthBackend.DEPTH_PRO})

    def __init__(self, settings: EstimatorSettings) -> None:
        """
        Load the image processor and model.

        Parameters
        ----------
        settings : EstimatorSettings
            Checkpoint, depth kind, batching, device and precision.

        Raises
        ------
        ValueError
            If the checkpoint belongs to another backend or has no field of view head, without which its output is
            not metric.
        OSError
            If the checkpoint cannot be found or has no safetensors weights.
        """
        super().__init__(settings)
        if not getattr(self._model.config, "use_fov_model", False):
            raise ValueError(f"{settings.weights_path} has no field of view head, so its depth is not metric.")

    def _to_depth_map(self, prediction: Prediction, image: Image.Image) -> DepthMap:
        field_of_view: torch.Tensor | None = prediction.get("field_of_view")
        focal_length: torch.Tensor | None = prediction.get("focal_length")
        return DepthMap(
            values=self._to_array(prediction, image),
            kind=self.settings.depth_kind,
            focal_length_pixels=None if focal_length is None else float(focal_length),
            field_of_view_degrees=None if field_of_view is None else float(field_of_view),
        )
