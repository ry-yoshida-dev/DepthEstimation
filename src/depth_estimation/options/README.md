# options

## Overview

Individual options composing `EstimatorSettings`, shared by every backend: which family runs, what its output means,
and where and in which precision it runs.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `DepthBackend`: estimator family, the depth kinds it can output and whether it needs a prompt depth. |
| [depth_kind.py](./depth_kind.py) | `DepthKind`: `metric` or `relative_inverse`. |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`; each runtime resolves it to its own device. |
| [precision.py](./precision.py) | `Precision`: `float32`, `float16` or `bfloat16`. |

## Examples

```python
DepthBackend.ZOEDEPTH.supported_depth_kinds
DepthBackend.PROMPT_DEPTH_ANYTHING.is_prompt_depth_required
Device.MPS
```
