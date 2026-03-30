"""
TATR extractor — ONNX Runtime backend.

Exports the HuggingFace TATR model to ONNX on first use (cached to disk),
then runs inference with onnxruntime. No PyTorch required at inference time.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image

from omnidocs.tasks.table_extraction.base import BaseTableExtractor
from omnidocs.tasks.table_extraction.models import TableOutput
from omnidocs.utils.cache import get_model_cache_dir

from .config import TATRONNXConfig
from .pytorch import _detections_to_table_output

if TYPE_CHECKING:
    from omnidocs.tasks.ocr_extraction.models import OCROutput

# TATR standard input size
_TATR_SIZE = (800, 800)

# Label map for TATR structure recognition
_TATR_ID2LABEL = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}

# ImageNet normalisation (used by TATR processor)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_onnx_path(config: TATRONNXConfig) -> Path:
    cache_dir = get_model_cache_dir(config.cache_dir)
    return cache_dir / f"tatr_{config.variant.value}.onnx"


# def _export_onnx(config: TATRONNXConfig, onnx_path: Path) -> None:
#     """Export the HuggingFace TATR model to ONNX. Requires torch + transformers."""
#     try:
#         import torch
#         from transformers import TableTransformerForObjectDetection
#     except ImportError:
#         raise ImportError(
#             "torch and transformers are required to export TATR to ONNX. "
#             "Install with: pip install torch transformers\n"
#             "After export, only onnxruntime is needed for inference."
#         )


#     cache_dir = str(get_model_cache_dir(config.cache_dir))
#     model = TableTransformerForObjectDetection.from_pretrained(
#         config.repo_id, cache_dir=cache_dir
#     )
def _export_onnx(config: TATRONNXConfig, onnx_path: Path) -> None:
    """Export the HuggingFace TATR model to ONNX. Requires torch + transformers."""
    try:
        import torch
        from transformers import TableTransformerForObjectDetection
    except ImportError:
        raise ImportError(
            "torch and transformers are required to export TATR to ONNX. "
            "Install with: pip install torch transformers\n"
            "After export, only onnxruntime is needed for inference."
        )

    # Workaround: huggingface_hub >=0.24 strict validation rejects None for bool fields.
    # TATR config.json has "dilation": null — patch _validate_simple_type to allow it.
    try:
        import huggingface_hub.dataclasses as _hf_dc

        _orig = _hf_dc._validate_simple_type

        def _patched(name, value, expected_type):
            if value is None:
                return
            _orig(name, value, expected_type)

        _hf_dc._validate_simple_type = _patched
    except Exception:
        pass

    cache_dir = str(get_model_cache_dir(config.cache_dir))
    model = TableTransformerForObjectDetection.from_pretrained(config.repo_id, cache_dir=cache_dir)
    model.eval()

    dummy = torch.zeros(1, 3, _TATR_SIZE[1], _TATR_SIZE[0])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        ({"pixel_values": dummy},),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=14,
    )


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    """Resize + normalise to TATR input format, returns (1, 3, H, W) float32."""
    img = pil_image.resize(_TATR_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return arr.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert DETR center-format boxes to x1y1x2y2."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


class TATRONNXExtractor(BaseTableExtractor):
    """
        TATR table structure extractor — ONNX Runtime backend.

        Exports the TATR model to ONNX on first use (cached), then runs inference
        via onnxruntime. No PyTorch dependency required at inference time.

        Example:
    ```python
            from omnidocs.tasks.table_extraction import TATRExtractor
            from omnidocs.tasks.table_extraction.tatr import TATRONNXConfig, TATRVariant

            extractor = TATRExtractor(backend=TATRONNXConfig(
                variant=TATRVariant.ALL,
                use_gpu=False,
            ))
            result = extractor.extract(table_image)
    ```
    """

    def __init__(self, config: TATRONNXConfig):
        self.config = config
        self._session = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for TATRONNXExtractor. "
                "Install with: pip install onnxruntime  (CPU)  or  pip install onnxruntime-gpu  (CUDA)"
            )

        onnx_path = _get_onnx_path(self.config)
        if not onnx_path.exists():
            _export_onnx(self.config, onnx_path)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.config.use_gpu else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        ocr_output: Optional["OCROutput"] = None,
    ) -> TableOutput:
        pil_image = self._prepare_image(image)
        orig_w, orig_h = pil_image.size

        pixel_values = _preprocess(pil_image)

        logits, pred_boxes = self._session.run(
            ["logits", "pred_boxes"],
            {"pixel_values": pixel_values},
        )
        # logits: (1, num_queries, num_classes)  pred_boxes: (1, num_queries, 4) cxcywh normed
        logits = logits[0]  # (Q, C)
        pred_boxes = pred_boxes[0]  # (Q, 4)

        probs = _sigmoid(logits)  # (Q, C)
        scores = probs.max(axis=1)  # (Q,)
        label_ids = probs.argmax(axis=1)  # (Q,)

        # Filter by threshold and exclude "no object" (last class)
        no_obj_idx = logits.shape[1] - 1
        keep = (scores >= self.config.detection_threshold) & (label_ids != no_obj_idx)

        kept_scores = scores[keep].tolist()
        kept_labels = label_ids[keep].tolist()

        # Denormalise boxes to original image pixels
        boxes_xyxy = _box_cxcywh_to_xyxy(pred_boxes[keep])
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h
        kept_boxes = boxes_xyxy.tolist()

        return _detections_to_table_output(
            boxes=kept_boxes,
            scores=kept_scores,
            labels=kept_labels,
            image_width=orig_w,
            image_height=orig_h,
            ocr_output=ocr_output,
            model_name=f"TATR-{self.config.variant.value}-onnx",
            label_map=_TATR_ID2LABEL,
        )
