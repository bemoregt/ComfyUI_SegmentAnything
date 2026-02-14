import torch
import numpy as np
from PIL import Image
import json

# segment-anything 패키지가 없을 경우 명확한 오류 메시지 제공
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

SAM_MODEL_TYPES = ["vit_h", "vit_l", "vit_b"]


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────
# 1. SAM Model Loader
# ──────────────────────────────────────────────

class SAMModelLoader:
    """Segment Anything Model (SAM) 체크포인트를 로드합니다.

    체크포인트 다운로드:
      vit_h  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
      vit_l  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
      vit_b  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "/path/to/sam_vit_h_4b8939.pth",
                    "multiline": False,
                }),
                "model_type": (SAM_MODEL_TYPES, {"default": "vit_h"}),
            }
        }

    RETURN_TYPES = ("SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "segmentation/SAM"

    def load_model(self, checkpoint_path: str, model_type: str):
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything 패키지가 설치되지 않았습니다.\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        device = _get_device()
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.eval()
        sam.to(device)

        predictor = SamPredictor(sam)
        return ({"predictor": predictor, "device": device},)


# ──────────────────────────────────────────────
# 2. SAM BBox Segmenter
# ──────────────────────────────────────────────

class SAMBBoxSegment:
    """바운딩박스를 비주얼 프롬프트로 사용하여 SAM 세그멘테이션을 수행합니다.

    bbox_json (선택):
      FasterRCNN Detector 등의 JSON 출력을 연결하면 x1/y1/x2/y2 대신 사용됩니다.
      여러 박스가 있을 경우 bbox_index 로 선택합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),          # [B, H, W, C] float32 0~1
                "sam_model": ("SAM_MODEL",),
                "x1": ("FLOAT", {"default": 0.0,   "min": 0.0, "max": 16384.0, "step": 1.0}),
                "y1": ("FLOAT", {"default": 0.0,   "min": 0.0, "max": 16384.0, "step": 1.0}),
                "x2": ("FLOAT", {"default": 256.0, "min": 0.0, "max": 16384.0, "step": 1.0}),
                "y2": ("FLOAT", {"default": 256.0, "min": 0.0, "max": 16384.0, "step": 1.0}),
                "multimask_output": ("BOOLEAN", {"default": False,
                                                  "tooltip": "True: SAM이 3개 후보 마스크 반환, mask_index로 선택"}),
                "mask_index": ("INT", {"default": 0, "min": 0, "max": 2,
                                        "tooltip": "multimask_output=True일 때 사용할 마스크 번호 (0=best)"}),
                "background_color": (["black", "white", "original"], {"default": "black",
                                      "tooltip": "마스크 외부 영역의 색상"}),
            },
            "optional": {
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "FasterRCNN JSON 출력. 지정 시 x1/y1/x2/y2 대신 사용됩니다.",
                }),
                "bbox_index": ("INT", {"default": 0, "min": 0, "max": 999,
                                        "tooltip": "bbox_json에 여러 박스가 있을 때 선택할 인덱스"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented_image", "mask")
    FUNCTION = "segment"
    CATEGORY = "segmentation/SAM"

    def segment(self, image, sam_model, x1, y1, x2, y2,
                multimask_output, mask_index, background_color,
                bbox_json="", bbox_index=0):

        predictor = sam_model["predictor"]
        results_images = []
        results_masks = []

        batch_size = image.shape[0]

        for i in range(batch_size):
            # ComfyUI IMAGE [H, W, C] float32 0~1 → numpy uint8
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # 바운딩박스 결정
            bbox = self._resolve_bbox(bbox_json, bbox_index, x1, y1, x2, y2)
            input_box = np.array(bbox, dtype=np.float32)

            # SAM 추론
            predictor.set_image(img_np)
            try:
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=multimask_output,
                )
            except (RuntimeError, NotImplementedError):
                # MPS 미지원 연산 → CPU 폴백
                sam_obj = predictor.model
                device_orig = next(sam_obj.parameters()).device
                sam_obj.to("cpu")
                predictor.set_image(img_np)
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=multimask_output,
                )
                sam_obj.to(device_orig)

            # 마스크 선택
            if multimask_output and len(masks) > 1:
                idx = min(mask_index, len(masks) - 1)
            else:
                idx = 0

            mask_bool = masks[idx]          # bool [H, W]
            mask_f = mask_bool.astype(np.float32)

            # 세그멘테이션 이미지 생성
            src = img_np.astype(np.float32) / 255.0
            if background_color == "black":
                seg = src * mask_f[:, :, None]
            elif background_color == "white":
                seg = src * mask_f[:, :, None] + (1.0 - mask_f[:, :, None])
            else:  # original — 마스크 경계만 보존, 배경도 원본
                seg = src.copy()

            results_images.append(torch.from_numpy(seg.astype(np.float32)))
            results_masks.append(torch.from_numpy(mask_f))

        out_images = torch.stack(results_images, dim=0)  # [B, H, W, C]
        out_masks  = torch.stack(results_masks,  dim=0)  # [B, H, W]
        return (out_images, out_masks)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_bbox(bbox_json: str, bbox_index: int, x1, y1, x2, y2):
        """JSON이 있으면 지정 인덱스 박스 추출, 없으면 float 값 반환."""
        if bbox_json.strip():
            try:
                data = json.loads(bbox_json)
                if isinstance(data, list) and len(data) > bbox_index:
                    item = data[bbox_index]
                    if isinstance(item, dict) and "box" in item:
                        return item["box"]          # FasterRCNN 형식
                    elif isinstance(item, (int, float)):
                        return data                 # [x1,y1,x2,y2] 직접 배열
            except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                pass
        return [x1, y1, x2, y2]


# ──────────────────────────────────────────────
# 3. BBox from JSON  (FasterRCNN ↔ SAM 연결 헬퍼)
# ──────────────────────────────────────────────

class SAMBBoxFromJSON:
    """FasterRCNN 등의 JSON 출력에서 특정 인덱스의 바운딩박스를 추출합니다.

    SAMBBoxSegment 의 x1/y1/x2/y2 입력에 직접 연결할 수 있습니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes_json": ("STRING", {"default": "[]", "multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("x1", "y1", "x2", "y2", "label")
    FUNCTION = "extract_bbox"
    CATEGORY = "segmentation/SAM"

    def extract_bbox(self, bboxes_json: str, index: int):
        try:
            data = json.loads(bboxes_json)
            if isinstance(data, list) and index < len(data):
                item = data[index]
                box   = item.get("box",   [0.0, 0.0, 100.0, 100.0])
                label = item.get("label", "")
                return (float(box[0]), float(box[1]), float(box[2]), float(box[3]), str(label))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
            pass
        return (0.0, 0.0, 100.0, 100.0, "")


# ──────────────────────────────────────────────
# Node registration
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SAMModelLoader":   SAMModelLoader,
    "SAMBBoxSegment":   SAMBBoxSegment,
    "SAMBBoxFromJSON":  SAMBBoxFromJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMModelLoader":   "SAM Model Loader",
    "SAMBBoxSegment":   "SAM BBox Segmenter",
    "SAMBBoxFromJSON":  "BBox Extractor from JSON",
}
