# depth_estimation

## Overview

Model-independent interface, settings and result types for monocular depth estimation, the YAML presets of every
supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [estimator.py](./estimator.py) | `DepthEstimator`: runtime-independent base estimating images eagerly or streamed, with mini-batching, RGB conversion and prompt depth validation. |
| [prompt_depth.py](./prompt_depth.py) | `PromptDepth`: validated low-resolution metric depth of one image, for Prompt Depth Anything. |
| [rgb_conversion.py](./rgb_conversion.py) | `RgbConversion`: turns 8-bit images into RGB over a white background and rejects high bit depths. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `ByteArray`). |
| [settings/](./settings/README.md) | `EstimatorSettings` with `from_mapping` and `build()`, and `PresetCatalog` loading packaged presets and YAML files. |
| [options/](./options/README.md) | `DepthBackend`, `DepthKind`, `Device` and `Precision`: options composing `EstimatorSettings`. |
| [result/](./result/README.md) | `DepthMap` and the `TurboColormap` rendering it. |
| [runtime/](./runtime/README.md) | `TorchRuntime`: device, precision, model preparation and input transfer for PyTorch-backed estimators. |
| [backends/](./backends/README.md) | Estimators of every family, on `transformers` depth estimation models. |
| [config/](./config/README.md) | YAML presets of every backend, loaded by `PresetCatalog`. |
| [cli/](./cli/README.md) | `depth-estimate` command writing the depth of image files and directories. |

## Examples

```python
estimator = PresetCatalog.load(DepthBackend.DEPTH_ANYTHING, "v2_small").build()

depth_map = estimator.estimate(image)
depth_maps = estimator.estimate_images(images)
for depth_map in estimator.iter_estimate_images(lazily_loaded_images):
    ...
```
