# DepthEstimation

## Overview

`depth_estimation` puts publicly available monocular depth models behind one interface so that models can be swapped
and compared by changing configuration only. Application code calls one `DepthEstimator` and reads one `DepthMap`
type; which model runs is decided by an `EstimatorSettings`, normally built from a YAML preset.

- **Configuration is data**: every model-specific value (weights, depth kind, batch size, precision) lives in a YAML
  preset under [config/](src/depth_estimation/config/README.md), one folder per backend, loaded by `PresetCatalog`.
- **Depth kinds are explicit**: `METRIC` values are meters, larger is farther. `RELATIVE_INVERSE` values are inverse
  depth up to an unknown scale and shift, larger is nearer, and cannot be compared across images. Each `DepthMap`
  states its kind, and `nearness()` gives both kinds one direction in `[0, 1]`.
- **Results are comparable**: every depth map has the size of its source image, in `float32`, whatever the model's
  input resolution and precision. `to_colored_image` and `to_grayscale_image` render it.
- **Wrong use fails early**: a checkpoint with the architecture of another backend, or a Depth Anything checkpoint of
  another depth kind, is rejected before its weights are loaded.
- **Only safetensors weights load**: other formats can run code when unpickled, so a checkpoint without safetensors
  weights raises `OSError`.

| Backend | Models | Depth kind | Output beyond depth | Presets |
| ------- | ------ | ---------- | ------------------- | ------- |
| `depth_anything` | Depth Anything, Depth Anything V2, Distill-Any-Depth | relative, or metric by checkpoint | - | `v1_*`, `v2_*`, `v2_metric_indoor_*`, `v2_metric_outdoor_*`, `distill_*` |
| `dpt` | DPT, MiDaS 3.1 | relative | - | `vit_large`, `beit_*`, `swinv2_*` |
| `zoedepth` | ZoeDepth | metric | - | `nyu`, `kitti`, `nyu_kitti` |
| `depth_pro` | Apple Depth Pro | metric | Field of view, focal length | `base` |
| `glpn` | GLPN | metric | - | `nyu` |
| `prompt_depth_anything` | Prompt Depth Anything | metric, needs a `PromptDepth` per image | - | `small`, `small_transparent`, `large` |
| `tipsv2_dpt` | TIPSv2 with a DPT head | metric | - | `b14`, `l14`, `g14` |

**Licenses differ per checkpoint, and several forbid commercial use**; see the table in
[config/](src/depth_estimation/config/README.md) before using a preset in a product.

A metric model is only reliable for scenes like its training data: `nyu` and `indoor` presets are trained on rooms,
`kitti` and `outdoor` presets on road scenes.

Images are processed in mini-batches; images whose model inputs have the same shape share one forward pass, and
`iter_estimate_images` reads a stream of images one mini-batch at a time, so datasets of any size run in bounded
memory. Images with transparency are composited over white; high-bit-depth images are rejected rather than clipped.

Weights are downloaded from the Hugging Face Hub on first use. The Hub client sends anonymous usage telemetry; set
`HF_HUB_DISABLE_TELEMETRY=1` to turn it off, or `HF_HUB_OFFLINE=1` to run from the local cache only. Images are
processed locally.

For module details, see [src/depth_estimation/README.md](src/depth_estimation/README.md).

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Examples

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
depth-estimate images/ --recursive --preset zoedepth/nyu --device cpu --output-directory depth/
depth-estimate photo.jpg --weights Intel/dpt-large --backend dpt --depth-kind relative_inverse --output-directory depth/
```
