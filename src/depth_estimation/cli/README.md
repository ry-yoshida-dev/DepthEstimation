# cli

## Overview

The `depth-estimate` command, installed with the package. It writes `<stem>.npy` with the raw values and `<stem>.png` with a Turbo rendering for each image file; run `depth-estimate --help` for the options. Prompt Depth Anything is available through the Python API only.

## Components

| Component | Description |
| --------- | ----------- |
| [command.py](./command.py) | `EstimateCommand`: argument parsing and validation, settings construction, streaming estimation and output. |
| [image_files.py](./image_files.py) | `ImageFiles`: expands files and directories into image files and reads them upright. |
| [image_file_stream.py](./image_file_stream.py) | `ImageFileStream`: reads image files lazily, skipping unusable ones with a warning and tracking the path of each yielded image. |

## Examples

```bash
depth-estimate --list-presets
depth-estimate images/ --preset depth_anything/v2_small --output-directory depth/
depth-estimate images/ --recursive --preset depth_pro/base --precision float16 --output-directory depth/
depth-estimate photo.jpg --config my_settings.yaml --output-directory depth/
```
