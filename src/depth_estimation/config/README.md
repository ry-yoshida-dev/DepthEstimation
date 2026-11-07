# config

## Overview

Ready-made YAML presets, one folder per `DepthBackend` value, whose `estimator` section maps onto the fields of
[`EstimatorSettings`](../settings/core.py). Licenses are those stated on each model card when the preset was added; the
model card remains authoritative, and "Not stated" checkpoints may still be non-commercial.

## Components

| Component | Presets | Weights | Depth kind | License |
| --------- | ------- | ------- | ---------- | ------- |
| [depth_anything/](./depth_anything/) | `v1_small`, `v1_base`, `v1_large` | `LiheYoung/depth-anything-{small,base,large}-hf` | relative | Apache-2.0 |
| [depth_anything/](./depth_anything/) | `v2_small` | `depth-anything/Depth-Anything-V2-Small-hf` | relative | Apache-2.0 |
| [depth_anything/](./depth_anything/) | `v2_base`, `v2_large` | `depth-anything/Depth-Anything-V2-{Base,Large}-hf` | relative | **CC-BY-NC-4.0** |
| [depth_anything/](./depth_anything/) | `v2_metric_indoor_{small,base,large}` | `depth-anything/Depth-Anything-V2-Metric-Indoor-*-hf` | metric, up to 20 m | Not stated |
| [depth_anything/](./depth_anything/) | `v2_metric_outdoor_{small,base}` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-*-hf` | metric, up to 80 m | Not stated |
| [depth_anything/](./depth_anything/) | `v2_metric_outdoor_large` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | metric, up to 80 m | Apache-2.0 |
| [depth_anything/](./depth_anything/) | `distill_small`, `distill_large` | `xingyang1/Distill-Any-Depth-{Small,Large}-hf` | relative | MIT |
| [dpt/](./dpt/) | `vit_large` | `Intel/dpt-large` | relative | Apache-2.0 |
| [dpt/](./dpt/) | `beit_base_384`, `beit_large_384`, `beit_large_512` | `Intel/dpt-beit-*` | relative | MIT |
| [dpt/](./dpt/) | `swinv2_tiny_256`, `swinv2_base_384`, `swinv2_large_384` | `Intel/dpt-swinv2-*` | relative | MIT |
| [zoedepth/](./zoedepth/) | `nyu`, `kitti`, `nyu_kitti` | `Intel/zoedepth-{nyu,kitti,nyu-kitti}` | metric | MIT |
| [depth_pro/](./depth_pro/) | `base` | `apple/DepthPro-hf` | metric | **apple-amlr (research only)** |
| [glpn/](./glpn/) | `nyu` | `vinvino02/glpn-nyu` | metric, up to 10 m | Apache-2.0 |
| [prompt_depth_anything/](./prompt_depth_anything/) | `small`, `small_transparent`, `large` | `depth-anything/prompt-depth-anything-{vits,vits-transparent,vitl}-hf` | metric | Apache-2.0 |
| [tipsv2_dpt/](./tipsv2_dpt/) | `b14`, `l14`, `g14` | `google/tipsv2-{b14,l14,g14}-dpt` | metric, up to 10 m | Apache-2.0 |

## Examples

```yaml
estimator:
  backend: depth_anything
  weights_path: depth-anything/Depth-Anything-V2-Small-hf
  depth_kind: relative_inverse
  batch_size: 8
  device: auto
  precision: float32
```
