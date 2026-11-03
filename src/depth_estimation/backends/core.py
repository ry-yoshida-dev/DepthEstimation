from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.modeling_outputs import DepthEstimatorOutput

from ..array_types import FloatArray
from ..estimator import DepthEstimator
from ..options import DepthBackend
from ..prompt_depth import PromptDepth
from ..result import DepthMap
from ..runtime import TorchRuntime
from ..settings import EstimatorSettings
from .checkpoint_config import CheckpointConfig
from .depth_processor import DepthProcessor
from .prediction import Prediction


class TransformersDepthEstimator(DepthEstimator):
    """
    Depth estimator backed by a ``transformers`` depth estimation model and its image processor.

    Processors keep the aspect ratio, so images of one mini-batch may reach the model at different sizes. Images whose
    model inputs have the same shape share one forward pass; depth maps are returned in input order at the size of
    their image. Only safetensors weights are loaded, since other formats can run code when unpickled.

    Subclasses adapt what a family does differently: extra processor inputs and how outputs are post-processed.
    """

    BACKENDS: ClassVar[frozenset[DepthBackend]] = frozenset(
        {DepthBackend.DEPTH_ANYTHING, DepthBackend.DPT, DepthBackend.GLPN, DepthBackend.TIPSV2_DPT}
    )

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
            If the checkpoint belongs to another backend or contradicts ``depth_kind``.
        OSError
            If the checkpoint cannot be found or has no safetensors weights.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        CheckpointConfig.read(settings.weights_path).validate(settings)
        self._processor: DepthProcessor = cast(
            DepthProcessor, AutoImageProcessor.from_pretrained(settings.weights_path)
        )
        self._model: torch.nn.Module = cast(
            torch.nn.Module, AutoModelForDepthEstimation.from_pretrained(settings.weights_path, use_safetensors=True)
        )
        self._runtime.prepare_model(self._model)

    def _estimate_mini_batch(
        self, images: Sequence[Image.Image], prompt_depths: Sequence[PromptDepth | None]
    ) -> list[DepthMap]:
        inputs: Mapping[str, Sequence[torch.Tensor]] = self._processor(
            images=list(images), **self._processor_arguments(prompt_depths)
        )
        groups: defaultdict[tuple[torch.Size, ...], list[int]] = defaultdict(list)
        for index in range(len(images)):
            groups[tuple(tensors[index].shape for tensors in inputs.values())].append(index)
        depth_maps: dict[int, DepthMap] = {}
        for indices in groups.values():
            depth_maps |= zip(
                indices, self._estimate_group(inputs, indices, [images[index] for index in indices]), strict=True
            )
        return [depth_maps[index] for index in range(len(images))]

    def _estimate_group(
        self, inputs: Mapping[str, Sequence[torch.Tensor]], indices: Sequence[int], images: Sequence[Image.Image]
    ) -> list[DepthMap]:
        batch: dict[str, torch.Tensor] = {
            name: self._runtime.to_model_input(torch.stack([tensors[index] for index in indices]))
            for name, tensors in inputs.items()
        }
        with torch.inference_mode():
            outputs: DepthEstimatorOutput = self._model(**batch)
        # Interpolation of the post-processing does not support half precision on every device.
        for name, value in cast(Mapping[str, object], outputs).items():
            if isinstance(value, torch.Tensor):
                outputs[name] = value.float()
        predictions: list[Prediction] = self._post_process(outputs, [(image.height, image.width) for image in images])
        return [self._to_depth_map(prediction, image) for prediction, image in zip(predictions, images, strict=True)]

    def _processor_arguments(self, prompt_depths: Sequence[PromptDepth | None]) -> dict[str, object]:
        return {}

    def _post_process(self, outputs: DepthEstimatorOutput, image_sizes: Sequence[tuple[int, int]]) -> list[Prediction]:
        return self._processor.post_process_depth_estimation(outputs, target_sizes=list(image_sizes))

    def _to_depth_map(self, prediction: Prediction, image: Image.Image) -> DepthMap:
        return DepthMap(values=self._to_array(prediction, image), kind=self.settings.depth_kind)

    @staticmethod
    def _to_array(prediction: Prediction, image: Image.Image) -> FloatArray:
        depth: torch.Tensor = prediction["predicted_depth"]
        return depth.reshape(image.height, image.width).float().cpu().numpy()
