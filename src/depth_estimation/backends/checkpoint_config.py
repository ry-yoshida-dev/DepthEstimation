from dataclasses import dataclass
from typing import ClassVar

from transformers import PreTrainedConfig

from ..options import DepthBackend, DepthKind
from ..settings import EstimatorSettings


@dataclass(frozen=True)
class CheckpointConfig:
    """
    The fields of a checkpoint's ``config.json`` that settings are checked against, read without loading its weights.

    A wrong depth kind would silently invert every nearness, and only Depth Anything checkpoints state their kind, so
    other checkpoints are trusted to match the settings.

    Attributes
    ----------
    model_type : str | None
        Architecture of the checkpoint.
    depth_estimation_type : str | None
        ``relative`` or ``metric`` in Depth Anything checkpoints; absent means ``relative``.
    """

    DEPTH_ANYTHING_KINDS: ClassVar[dict[str | None, DepthKind]] = {
        None: DepthKind.RELATIVE_INVERSE,
        "relative": DepthKind.RELATIVE_INVERSE,
        "metric": DepthKind.METRIC,
    }

    model_type: str | None
    depth_estimation_type: str | None

    @classmethod
    def read(cls, weights_path: str) -> "CheckpointConfig":
        """
        Read the configuration of a checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        CheckpointConfig
            Fields of its ``config.json``.

        Raises
        ------
        OSError
            If the configuration cannot be found.
        """
        values: dict[str, object] = PreTrainedConfig.get_config_dict(weights_path)[0]
        model_type: object = values.get("model_type")
        depth_estimation_type: object = values.get("depth_estimation_type")
        return cls(
            model_type=model_type if isinstance(model_type, str) else None,
            depth_estimation_type=depth_estimation_type if isinstance(depth_estimation_type, str) else None,
        )

    def validate(self, settings: EstimatorSettings) -> None:
        """
        Check that the checkpoint fits the settings.

        Parameters
        ----------
        settings : EstimatorSettings
            Settings naming this checkpoint.

        Raises
        ------
        ValueError
            If the checkpoint has the architecture of another backend, or a Depth Anything checkpoint outputs another
            depth kind.
        """
        if self.model_type != settings.backend.value:
            raise ValueError(
                f"{settings.weights_path} has model_type {self.model_type!r}. {settings.backend} needs "
                + f"{settings.backend.value!r}."
            )
        if settings.backend is not DepthBackend.DEPTH_ANYTHING:
            return
        checkpoint_kind: DepthKind | None = self.DEPTH_ANYTHING_KINDS.get(self.depth_estimation_type)
        if checkpoint_kind is not settings.depth_kind:
            raise ValueError(
                f"{settings.weights_path} outputs {checkpoint_kind} depth "
                + f"(depth_estimation_type {self.depth_estimation_type!r}). settings say {settings.depth_kind}."
            )
