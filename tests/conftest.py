from pathlib import Path

import pytest
import torch
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pretrained", action="store_true", help="run tests downloading real model weights (marker pretrained)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-pretrained"):
        return
    skip_pretrained: pytest.MarkDecorator = pytest.mark.skip(reason="needs --run-pretrained")
    for item in items:
        if "pretrained" in item.keywords:
            item.add_marker(skip_pretrained)


@pytest.fixture(scope="session")
def tiny_depth_anything_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory: Path = tmp_path_factory.mktemp("tiny_depth_anything")
    torch.manual_seed(0)
    backbone_config: Dinov2Config = Dinov2Config(
        image_size=28,
        patch_size=14,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        mlp_ratio=2,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=False,
    )
    DepthAnythingForDepthEstimation(
        DepthAnythingConfig(
            backbone_config=backbone_config,
            reassemble_hidden_size=16,
            neck_hidden_sizes=[4, 4, 4, 4],
            fusion_hidden_size=4,
            head_hidden_size=4,
        )
    ).save_pretrained(directory)
    DPTImageProcessor(
        size={"height": 28, "width": 28}, keep_aspect_ratio=True, ensure_multiple_of=14, do_pad=False
    ).save_pretrained(directory)
    return directory
