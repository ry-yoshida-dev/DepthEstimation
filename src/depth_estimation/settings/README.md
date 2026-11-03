# settings

## Overview

Everything needed to load and run an estimator, and reading it from YAML. A malformed configuration fails before a
model is loaded.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EstimatorSettings`: backend, weights, depth kind, batch size, device and precision; `from_mapping` reads plain values and `build()` loads the estimator. |
| [config_values.py](./config_values.py) | `ConfigValues`: typed, validating access to the plain values of a configuration section. |
| [presets.py](./presets.py) | `PresetCatalog`: lists and reads the packaged presets and YAML files of the same format. |

## Examples

```python
settings = PresetCatalog.load(DepthBackend.DPT, "swinv2_tiny_256")
settings = PresetCatalog.load_file(Path("my_settings.yaml"))
settings = EstimatorSettings(backend=DepthBackend.GLPN, weights_path="vinvino02/glpn-nyu", depth_kind=DepthKind.METRIC)
estimator = settings.build()
```
