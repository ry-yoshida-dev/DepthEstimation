# examples

## Overview

Runnable scripts using the package. To estimate the depth of a directory from the command line, use the installed
`depth-estimate` command (see [cli/](../src/depth_estimation/cli/README.md)).

## Components

| Component | Description |
| --------- | ----------- |
| [prompt_depth.py](./prompt_depth.py) | Estimates metric depth with Prompt Depth Anything, scaled by a low-resolution depth image, which the command line does not offer. |

## Examples

```bash
python examples/prompt_depth.py photo.jpg arkit_depth.png out/photo --meters-per-unit 0.001
```
