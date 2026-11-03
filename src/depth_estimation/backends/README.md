# backends

## Overview

Concrete `DepthEstimator` implementations on `transformers` depth estimation models. `TransformersDepthEstimator`
loads the image processor and model of a checkpoint, runs images of one input shape in one forward pass, and converts
the post-processed output to `DepthMap`. Families differing from that adapt only the difference.

Nothing is imported eagerly: `EstimatorSettings.build()` imports only the module of the requested backend, so importing
`depth_estimation` loads neither `torch` nor `transformers`.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `TransformersDepthEstimator`: the shared implementation, used as is by Depth Anything, DPT, GLPN and TIPSv2 DPT. |
| [zoedepth.py](./zoedepth.py) | `ZoeDepthEstimator`: cuts the padding added by the ZoeDepth processor from its output. |
| [depth_pro.py](./depth_pro.py) | `DepthProEstimator`: requires the field of view head and reports the camera estimate. |
| [prompt_depth_anything.py](./prompt_depth_anything.py) | `PromptDepthAnythingEstimator`: feeds each image's prompt depth to the model. |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: checks a checkpoint's `config.json` against the settings without loading its weights. |
| [depth_processor.py](./depth_processor.py) | `DepthProcessor`: protocol of the image processor methods the estimators use. |
| [prediction.py](./prediction.py) | `Prediction`: one image's entry in the post-processed output. |
