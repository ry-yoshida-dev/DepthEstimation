# runtime

## Overview

How an estimator's model is executed. `DepthEstimator` itself is independent of any inference framework; every current
backend runs on PyTorch through `TorchRuntime`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`: resolves `Device` to a `torch.device`, `Precision` to a dtype, and prepares models and inputs. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
```
