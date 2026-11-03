"""
Estimate metric depth of one image with Prompt Depth Anything, scaled by a low-resolution depth image.

Usage
-----
python examples/prompt_depth.py IMAGE DEPTH_IMAGE OUTPUT_STEM --meters-per-unit 0.001

``DEPTH_IMAGE`` is a single-channel image such as the 16-bit millimeter PNG of a phone LiDAR. The command writes
``OUTPUT_STEM.npy`` with the depth in meters and ``OUTPUT_STEM.png`` with the depth in the Turbo colormap.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from depth_estimation import DepthBackend, DepthEstimator, DepthMap, PresetCatalog, PromptDepth


def main() -> None:
    """
    Parse arguments, run the model and write the depth.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("depth_image", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--meters-per-unit", type=float, default=0.001)
    parser.add_argument("--preset", default="small", choices=PresetCatalog.names(DepthBackend.PROMPT_DEPTH_ANYTHING))
    arguments: argparse.Namespace = parser.parse_args()

    with Image.open(arguments.image) as image, Image.open(arguments.depth_image) as depth_image:
        prompt_depth: PromptDepth = PromptDepth.from_image(depth_image, arguments.meters_per_unit)
        estimator: DepthEstimator = PresetCatalog.load(DepthBackend.PROMPT_DEPTH_ANYTHING, arguments.preset).build()
        depth_map: DepthMap = estimator.estimate(ImageOps.exif_transpose(image), prompt_depth)

    output_stem: Path = arguments.output_stem
    np.save(output_stem.with_suffix(".npy"), depth_map.values)
    depth_map.to_colored_image().save(output_stem.with_suffix(".png"))
    print(f"{depth_map.width}x{depth_map.height}: {depth_map.values.min():.3f} m to {depth_map.values.max():.3f} m")


if __name__ == "__main__":
    main()
