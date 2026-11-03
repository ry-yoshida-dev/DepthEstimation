from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
from PIL import Image
from transformers.modeling_outputs import DepthEstimatorOutput

from .prediction import Prediction


class DepthProcessor(Protocol):
    """
    The part of a ``transformers`` depth image processor the estimators use.
    """

    def __call__(self, *, images: list[Image.Image], **kwargs: object) -> Mapping[str, Sequence[torch.Tensor]]:
        """
        Preprocess images without stacking them, since their sizes may differ.

        Parameters
        ----------
        images : list[Image.Image]
            RGB images.
        **kwargs : object
            Extra inputs of the model family.

        Returns
        -------
        Mapping[str, Sequence[torch.Tensor]]
            Model inputs by argument name, one tensor per image.
        """
        ...

    def post_process_depth_estimation(
        self, outputs: DepthEstimatorOutput, *, target_sizes: list[tuple[int, int]], **kwargs: object
    ) -> list[Prediction]:
        """
        Turn raw model outputs into depth at the target sizes.

        Parameters
        ----------
        outputs : DepthEstimatorOutput
            Raw outputs of the model.
        target_sizes : list[tuple[int, int]]
            Height and width of each image.
        **kwargs : object
            Extra arguments of the model family.

        Returns
        -------
        list[Prediction]
            Prediction of each image.
        """
        ...
