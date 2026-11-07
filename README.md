# DepthEstimation

## Overview

`depth_estimation` puts publicly available monocular depth models behind one interface, so models are swapped and
compared by changing configuration only. A `DepthEstimator` is built from YAML presets and returns a `DepthMap` at the
size of the source image, stating whether its values are metric (meters) or relative inverse depth.

| Backend | Models | Depth kind | Output beyond depth | Presets |
| ------- | ------ | ---------- | ------------------- | ------- |
| `depth_anything` | Depth Anything, Depth Anything V2, Distill-Any-Depth | relative, or metric by checkpoint | - | `v1_*`, `v2_*`, `v2_metric_indoor_*`, `v2_metric_outdoor_*`, `distill_*` |
| `dpt` | DPT, MiDaS 3.1 | relative | - | `vit_large`, `beit_*`, `swinv2_*` |
| `zoedepth` | ZoeDepth | metric | - | `nyu`, `kitti`, `nyu_kitti` |
| `depth_pro` | Apple Depth Pro | metric | Field of view, focal length | `base` |
| `glpn` | GLPN | metric | - | `nyu` |
| `prompt_depth_anything` | Prompt Depth Anything | metric, needs a `PromptDepth` per image | - | `small`, `small_transparent`, `large` |
| `tipsv2_dpt` | TIPSv2 with a DPT head | metric | - | `b14`, `l14`, `g14` |

**Licenses differ per checkpoint, and several forbid commercial use**; see [config/](src/depth_estimation/config/README.md)
before using a preset in a product.

## Installation

```bash
pip install .
```

## Usage

```python
from PIL import Image

from depth_estimation import DepthBackend, DepthMap, PresetCatalog

estimator = PresetCatalog.load(DepthBackend.DEPTH_ANYTHING, "v2_metric_indoor_small").build()

depth_map: DepthMap = estimator.estimate(Image.open("room.jpg"))
print(depth_map.kind, depth_map.values.min(), depth_map.values.max())
depth_map.to_colored_image().save("room_depth.png")
```

Depth Pro also estimates the camera:

```python
depth_map = PresetCatalog.load(DepthBackend.DEPTH_PRO, "base").build().estimate(image)
print(depth_map.values.mean(), depth_map.focal_length_pixels, depth_map.field_of_view_degrees)
```

Prompt Depth Anything takes its metric scale from a low-resolution depth image, e.g. from a phone LiDAR:

```python
from depth_estimation import PromptDepth

prompt_depth = PromptDepth.from_image(Image.open("arkit_depth.png"), meters_per_unit=0.001)
estimator = PresetCatalog.load(DepthBackend.PROMPT_DEPTH_ANYTHING, "small").build()
depth_map = estimator.estimate(image, prompt_depth)
```

From the command line, each image gets `<stem>.npy` with its raw values and `<stem>.png` with a colored rendering:

```bash
depth-estimate --list-presets
depth-estimate images/ --preset depth_anything/v2_small --output-directory depth/
```
