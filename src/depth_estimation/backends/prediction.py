from typing import NotRequired, TypedDict

import torch


class Prediction(TypedDict):
    """
    One image's entry in the output of ``post_process_depth_estimation``.

    Attributes
    ----------
    predicted_depth : torch.Tensor
        Depth at the requested size.
    field_of_view : NotRequired[torch.Tensor | None]
        Horizontal field of view in degrees; Depth Pro only.
    focal_length : NotRequired[torch.Tensor | None]
        Focal length in pixels; Depth Pro only.
    """

    predicted_depth: torch.Tensor
    field_of_view: NotRequired[torch.Tensor | None]
    focal_length: NotRequired[torch.Tensor | None]
