from enum import StrEnum

from .depth_kind import DepthKind


class DepthBackend(StrEnum):
    """
    Depth estimator family; each value is the ``model_type`` of the checkpoints the family loads.

    Model-specific values (weights, depth kind, batch size) live in the YAML presets; only what code must know about
    a family, the depth kinds it can output and whether it needs a prompt depth, is defined here.

    Attributes
    ----------
    DEPTH_ANYTHING : str
        Depth Anything, Depth Anything V2 and Distill-Any-Depth; relative or metric by checkpoint.
    DPT : str
        DPT and MiDaS 3.1 (ViT, BEiT and SwinV2 backbones).
    ZOEDEPTH : str
        ZoeDepth; metric.
    DEPTH_PRO : str
        Apple Depth Pro; metric, with the estimated field of view and focal length.
    GLPN : str
        GLPN; metric.
    PROMPT_DEPTH_ANYTHING : str
        Prompt Depth Anything; metric, scaled by a low-resolution depth prompt, e.g. from a LiDAR.
    TIPSV2_DPT : str
        TIPSv2 with a DPT head; metric.
    """

    DEPTH_ANYTHING = "depth_anything"
    DPT = "dpt"
    ZOEDEPTH = "zoedepth"
    DEPTH_PRO = "depth_pro"
    GLPN = "glpn"
    PROMPT_DEPTH_ANYTHING = "prompt_depth_anything"
    TIPSV2_DPT = "tipsv2_dpt"

    @property
    def supported_depth_kinds(self) -> frozenset[DepthKind]:
        """
        Depth kinds checkpoints of the family can output.

        Returns
        -------
        frozenset[DepthKind]
            Both kinds for ``DEPTH_ANYTHING`` and ``DPT``, which have checkpoints trained for either, otherwise only
            ``METRIC``.
        """
        match self:
            case DepthBackend.DEPTH_ANYTHING | DepthBackend.DPT:
                return frozenset({DepthKind.METRIC, DepthKind.RELATIVE_INVERSE})
            case (
                DepthBackend.ZOEDEPTH
                | DepthBackend.DEPTH_PRO
                | DepthBackend.GLPN
                | DepthBackend.PROMPT_DEPTH_ANYTHING
                | DepthBackend.TIPSV2_DPT
            ):
                return frozenset({DepthKind.METRIC})

    @property
    def is_prompt_depth_required(self) -> bool:
        """
        Whether every image needs a prompt depth; without one, Prompt Depth Anything has no metric scale.

        Returns
        -------
        bool
            ``True`` for ``PROMPT_DEPTH_ANYTHING``; other families accept no prompt depth.
        """
        return self is DepthBackend.PROMPT_DEPTH_ANYTHING
