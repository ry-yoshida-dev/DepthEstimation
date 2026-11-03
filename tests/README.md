# tests

## Overview

Unit tests that run without downloading model weights; `--run-pretrained` adds the tests that download real ones.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | `--run-pretrained` option and a tiny random Depth Anything checkpoint. |
| [test_configuration.py](./test_configuration.py) | Presets matching the `EstimatorSettings` schema, `PresetCatalog`, settings validation and backend capabilities. |
| [test_estimator.py](./test_estimator.py) | Shared mini-batching, streaming, RGB input and prompt depth rules. |
| [test_backends.py](./test_backends.py) | Forward passes by input shape, output order and size, family-specific processing on real processors, checkpoint checks and refusal of pickle weights. |
| [test_depth_map.py](./test_depth_map.py) | `DepthMap` validation, `nearness` of both kinds, rendering and `TurboColormap`. |
| [test_prompt_depth.py](./test_prompt_depth.py) | `PromptDepth` validation and reading depth images. |
| [test_rgb_conversion.py](./test_rgb_conversion.py) | RGB conversion, transparency and rejection of high bit depths. |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, dtype and model preparation. |
| [test_cli.py](./test_cli.py) | `depth-estimate` outputs, skipped files, overrides and invalid arguments. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads neither `torch` nor `transformers`. |
| [test_pretrained.py](./test_pretrained.py) | Real models on the Hub, marked `pretrained`. |
