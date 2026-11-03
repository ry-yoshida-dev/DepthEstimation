# result

## Overview

The output type shared by every model, and its rendering.

## Components

| Component | Description |
| --------- | ----------- |
| [depth_map.py](./depth_map.py) | `DepthMap`: depth values at image size, their kind, the camera estimate where available, `nearness` and rendered images. |
| [turbo_colormap.py](./turbo_colormap.py) | `TurboColormap`: maps values in `[0, 1]` to Turbo colors without a plotting library. |

## Examples

```python
depth_map.values
depth_map.nearness()
depth_map.to_colored_image().save("depth.png")
```
