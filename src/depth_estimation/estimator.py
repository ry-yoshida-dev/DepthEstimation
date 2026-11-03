from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched
from typing import ClassVar

from PIL import Image

from .options import DepthBackend
from .prompt_depth import PromptDepth
from .result import DepthMap
from .rgb_conversion import RgbConversion
from .settings import EstimatorSettings


class DepthEstimator(ABC):
    """
    Common base of every depth estimator, independent of the inference runtime.

    Subclasses declare the ``BACKENDS`` they implement. The base owns the settings, mini-batching, RGB conversion and
    the prompt depth rules of the backend, so a wrong call is rejected before any computation. How the model runs
    (PyTorch, ONNX Runtime, ...) is up to each subclass and its runtime.

    Attributes
    ----------
    settings : EstimatorSettings
        Backend, weights, depth kind, batching, device and precision.
    """

    BACKENDS: ClassVar[frozenset[DepthBackend]]

    def __init__(self, settings: EstimatorSettings) -> None:
        """
        Parameters
        ----------
        settings : EstimatorSettings
            Backend, weights, depth kind, batching, device and precision.

        Raises
        ------
        ValueError
            If the settings are for a backend this class does not implement.
        """
        if settings.backend not in self.BACKENDS:
            raise ValueError(
                f"{type(self).__name__} needs settings of {sorted(backend.value for backend in self.BACKENDS)}. "
                + f"got {settings.backend}"
            )
        self.settings: EstimatorSettings = settings

    @abstractmethod
    def _estimate_mini_batch(
        self, images: Sequence[Image.Image], prompt_depths: Sequence[PromptDepth | None]
    ) -> list[DepthMap]:
        """
        Estimate the depth of at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images of arbitrary sizes.
        prompt_depths : Sequence[PromptDepth | None]
            Prompt depth of each image, validated against the rules of the backend.

        Returns
        -------
        list[DepthMap]
            One depth map per image at the size of the image, in input order.
        """

    def estimate(self, image: Image.Image, prompt_depth: PromptDepth | None = None) -> DepthMap:
        """
        Estimate the depth of a single image.

        Parameters
        ----------
        image : Image.Image
            Input image.
        prompt_depth : PromptDepth | None
            Prompt depth of the image; required by ``PROMPT_DEPTH_ANYTHING`` and rejected by other backends.

        Returns
        -------
        DepthMap
            Depth at the size of the image.

        Raises
        ------
        ValueError
            If the image has a high bit depth or the prompt depth breaks the rules of the backend.
        """
        return self.estimate_images([image], None if prompt_depth is None else [prompt_depth])[0]

    def estimate_images(
        self, images: Sequence[Image.Image], prompt_depths: Sequence[PromptDepth] | None = None
    ) -> list[DepthMap]:
        """
        Estimate the depth of many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and 8-bit mode.
        prompt_depths : Sequence[PromptDepth] | None
            One prompt depth per image; required by ``PROMPT_DEPTH_ANYTHING`` and rejected by other backends.

        Returns
        -------
        list[DepthMap]
            One depth map per image, in input order.

        Raises
        ------
        ValueError
            If ``images`` is empty, an image has a high bit depth, or the prompt depths break the rules of the
            backend or differ in number from the images.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        if prompt_depths is not None and len(prompt_depths) != len(images):
            raise ValueError(f"expected one prompt depth per image ({len(images)}). got {len(prompt_depths)}")
        return list(self.iter_estimate_images(images, prompt_depths))

    def iter_estimate_images(
        self, images: Iterable[Image.Image], prompt_depths: Iterable[PromptDepth] | None = None
    ) -> Iterator[DepthMap]:
        """
        Estimate the depth of a stream of images, reading one mini-batch at a time, so memory stays bounded.

        Parameters
        ----------
        images : Iterable[Image.Image]
            Input images of arbitrary size and 8-bit mode; consumed lazily.
        prompt_depths : Iterable[PromptDepth] | None
            One prompt depth per image; required by ``PROMPT_DEPTH_ANYTHING`` and rejected by other backends.

        Returns
        -------
        Iterator[DepthMap]
            One depth map per image, in input order.

        Raises
        ------
        ValueError
            If the prompt depths break the rules of the backend; raised at the call. High-bit-depth images and
            streams of different lengths raise while iterating.
        """
        self._validate_prompt_depth_use(prompt_depths is not None)
        return self._iter_mini_batches(images, prompt_depths)

    def _iter_mini_batches(
        self, images: Iterable[Image.Image], prompt_depths: Iterable[PromptDepth] | None
    ) -> Iterator[DepthMap]:
        requests: Iterable[tuple[Image.Image, PromptDepth | None]] = (
            ((image, None) for image in images) if prompt_depths is None else zip(images, prompt_depths, strict=True)
        )
        for mini_batch in batched(requests, self.settings.batch_size):
            yield from self._estimate_mini_batch(
                [RgbConversion.convert(image) for image, _ in mini_batch],
                [prompt_depth for _, prompt_depth in mini_batch],
            )

    def _validate_prompt_depth_use(self, is_prompt_depth_given: bool) -> None:
        is_required: bool = self.settings.backend.is_prompt_depth_required
        if is_required and not is_prompt_depth_given:
            raise ValueError(f"{self.settings.backend} needs a prompt depth for every image.")
        if is_prompt_depth_given and not is_required:
            raise ValueError(f"{self.settings.backend} does not accept a prompt depth.")
