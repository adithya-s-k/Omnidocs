"""
DocLayout-YOLO layout extractor.

A YOLO-based model for document layout detection, optimized for academic papers
and technical documents.

Model: juliozhao/DocLayout-YOLO-DocStructBench
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseLayoutExtractor
from .models import (
    DOCLAYOUT_YOLO_CLASS_NAMES,
    DOCLAYOUT_YOLO_MAPPING,
    BoundingBox,
    LayoutBox,
    LayoutOutput,
)

# ============= Configuration =============


class DocLayoutYOLOConfig(BaseModel):
    """
    Configuration for DocLayout-YOLO layout extractor.

    This is a single-backend model (PyTorch only).

    Example:
        >>> config = DocLayoutYOLOConfig(device="cuda", confidence=0.3)
        >>> extractor = DocLayoutYOLO(config=config)
    """

    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'mps', 'cpu'. "
        "Auto-detects best available if 'cuda' specified but not available.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Custom path to model weights. If None, uses OMNIDOCS_MODELS_DIR env var or ~/.omnidocs/models/",
    )
    img_size: int = Field(
        default=1024,
        ge=320,
        le=1920,
        description="Input image size for inference. DocLayout-YOLO works best at 1024.",
    )
    confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections.",
    )

    model_config = ConfigDict(extra="forbid")


# ============= Constants =============

MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


# ============= DocLayout-YOLO Extractor =============


class DocLayoutYOLO(BaseLayoutExtractor):
    """
    DocLayout-YOLO layout extractor.

    A YOLO-based model optimized for document layout detection.
    Detects: title, text, figure, table, formula, captions, etc.

    This is a single-backend model (PyTorch only).

    Example:
        >>> from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        >>>
        >>> extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
        >>> result = extractor.extract(image)
        >>>
        >>> for box in result.bboxes:
        ...     print(f"{box.label.value}: {box.confidence:.2f}")
    """

    def __init__(self, config: DocLayoutYOLOConfig):
        """
        Initialize DocLayout-YOLO extractor.

        Args:
            config: Configuration object with device, model_path, etc.
        """
        self.config = config
        self._model = None
        self._device = self._resolve_device(config.device)
        self._model_path = self._resolve_model_path(config.model_path)

        # Load model
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve device, auto-detecting if needed."""
        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"
            elif device in ("cuda", "mps"):
                # Requested GPU but not available, fall back to CPU
                return "cpu"
            return device
        except ImportError:
            return "cpu"

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        """Resolve model path from config or environment."""
        if model_path:
            return Path(model_path)

        # Check environment variable
        models_dir = os.environ.get("OMNIDOCS_MODELS_DIR", os.path.expanduser("~/.omnidocs/models"))

        return Path(models_dir) / "doclayout_yolo"

    def _download_model(self) -> Path:
        """Download model from HuggingFace Hub if not present."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for model download. Install with: pip install huggingface_hub"
            )

        model_file = self._model_path / MODEL_FILENAME

        if model_file.exists():
            return model_file

        # Create directory
        self._model_path.mkdir(parents=True, exist_ok=True)

        # Download model
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=str(self._model_path),
        )

        return Path(downloaded_path)

    def _load_model(self) -> None:
        """Load DocLayout-YOLO model."""
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            raise ImportError("doclayout-yolo is required for DocLayoutYOLO. Install with: pip install doclayout-yolo")

        # Download if needed
        model_file = self._model_path / MODEL_FILENAME
        if not model_file.exists():
            model_file = self._download_model()

        # Load model
        self._model = YOLOv10(str(model_file))

    def extract(self, image: Union[Image.Image, np.ndarray, str, Path]) -> LayoutOutput:
        """
        Run layout extraction on an image.

        Args:
            image: Input image (PIL Image, numpy array, or path)

        Returns:
            LayoutOutput with detected layout boxes
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        img_width, img_height = pil_image.size

        # Run inference
        results = self._model.predict(
            pil_image,
            imgsz=self.config.img_size,
            conf=self.config.confidence,
            device=self._device,
        )

        result = results[0]

        # Parse detections
        layout_boxes = []

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Get coordinates
                bbox_coords = boxes.xyxy[i].cpu().numpy().tolist()

                # Get class and confidence
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                # Get original label from class names
                original_label = DOCLAYOUT_YOLO_CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Map to standardized label
                standard_label = DOCLAYOUT_YOLO_MAPPING.to_standard(original_label)

                layout_boxes.append(
                    LayoutBox(
                        label=standard_label,
                        bbox=BoundingBox.from_list(bbox_coords),
                        confidence=confidence,
                        class_id=class_id,
                        original_label=original_label,
                    )
                )

        # Sort by y-coordinate (top to bottom reading order)
        layout_boxes.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        return LayoutOutput(
            bboxes=layout_boxes,
            image_width=img_width,
            image_height=img_height,
            model_name="DocLayout-YOLO",
        )
