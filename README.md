# ComfyUI_SegmentAnything

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node pack that integrates Meta's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) into ComfyUI workflows. Use a bounding box as a visual prompt to generate high-quality segmentation masks from any image.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_SegmentAnything/blob/main/ScrShot%206.png)

## Nodes

### SAM Model Loader
Loads a SAM checkpoint from disk and exposes it as a `SAM_MODEL` object for downstream nodes.

| Input | Type | Description |
|---|---|---|
| `checkpoint_path` | STRING | Absolute path to the `.pth` checkpoint file |
| `model_type` | ENUM | `vit_h` / `vit_l` / `vit_b` — must match the checkpoint |

| Output | Type | Description |
|---|---|---|
| `sam_model` | SAM_MODEL | Loaded predictor, passed to the segmenter |

---

### SAM BBox Segmenter
Runs SAM inference using a bounding box as the visual prompt. Outputs the segmented image and a float mask.

| Input | Type | Required | Description |
|---|---|---|---|
| `image` | IMAGE | Yes | ComfyUI image tensor `[B, H, W, C]` float32 0–1 |
| `sam_model` | SAM_MODEL | Yes | Output from **SAM Model Loader** |
| `x1`, `y1`, `x2`, `y2` | FLOAT | Yes | Bounding box in pixel coordinates (top-left / bottom-right) |
| `multimask_output` | BOOLEAN | Yes | When `True`, SAM returns 3 candidate masks; select one with `mask_index` |
| `mask_index` | INT (0–2) | Yes | Index of the mask to use when `multimask_output` is enabled (0 = highest-confidence) |
| `background_color` | ENUM | Yes | `black` — background is black; `white` — background is white; `original` — full original image is returned |
| `bbox_json` | STRING | No | JSON string from a detector node (e.g. FasterRCNN). When provided, overrides `x1/y1/x2/y2` |
| `bbox_index` | INT | No | Which bounding box to use when `bbox_json` contains multiple detections |

| Output | Type | Description |
|---|---|---|
| `segmented_image` | IMAGE | Original image with the background replaced according to `background_color` |
| `mask` | MASK | Float mask `[B, H, W]` — 1.0 inside the segment, 0.0 outside |

---

### BBox Extractor from JSON
Helper node that splits a detector's JSON output into individual `x1 / y1 / x2 / y2 / label` values. Connect these directly to the **SAM BBox Segmenter** inputs.

Expected JSON format (compatible with **ComfyUI_FasterRCNN**):
```json
[
  { "label": "cat", "score": 0.97, "box": [120.5, 45.0, 380.2, 310.8] },
  { "label": "dog", "score": 0.91, "box": [400.0, 60.0, 650.0, 340.0] }
]
```

| Input | Type | Description |
|---|---|---|
| `bboxes_json` | STRING | JSON array of detection results |
| `index` | INT | Index of the detection to extract |

| Output | Type | Description |
|---|---|---|
| `x1`, `y1`, `x2`, `y2` | FLOAT | Bounding box coordinates |
| `label` | STRING | Class label of the selected detection |

---

## Installation

### 1. Clone or copy this repository

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI_SegmentAnything
```

### 2. Install the `segment-anything` package

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Download a SAM checkpoint

| Model | Size | Download |
|---|---|---|
| `vit_h` (recommended) | ~2.4 GB | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `vit_l` | ~1.2 GB | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| `vit_b` | ~375 MB | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

### 4. Restart ComfyUI

The nodes will appear under the **segmentation/SAM** category.

---

## Workflow Examples

### Manual bounding box

```
[Load Image] ──────────────────────────────► [SAM BBox Segmenter] ──► segmented_image
                                        ↑                          └──► mask
[SAM Model Loader] ──── sam_model ──────┘
  (set x1/y1/x2/y2 directly on the node)
```

### Detector-driven segmentation (FasterRCNN → SAM)

```
[Load Image] ──┬──────────────────────────► [SAM BBox Segmenter] ──► segmented_image
               │                       ↑↑                         └──► mask
               └──► [FasterRCNN] ──────┤└── sam_model ◄── [SAM Model Loader]
                     bboxes_json ───────┘ (bbox_json input)
```

Or use the helper node to fan out individual coordinates:

```
[FasterRCNN] ──► [BBox Extractor from JSON] ──► x1 ─┐
                                               y1 ─┤
                                               x2 ─┼──► [SAM BBox Segmenter]
                                               y2 ─┘
```

---

## Device Support

The loader and segmenter automatically select the best available device:

| Priority | Device |
|---|---|
| 1 | CUDA (NVIDIA GPU) |
| 2 | MPS (Apple Silicon) |
| 3 | CPU |

On Apple Silicon, operations unsupported by MPS automatically fall back to CPU.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- NumPy
- Pillow

---

## License

This project is released under the MIT License.
SAM itself is released under the [Apache 2.0 License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) by Meta AI.
