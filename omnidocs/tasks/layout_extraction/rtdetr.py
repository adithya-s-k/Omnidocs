"""
RT-DETR layout extractor.

A transformer-based real-time detection model for document layout detection.
Uses HuggingFace Transformers implementation.

Model: HuggingPanda/docling-layout
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseLayoutExtractor
from .models import (
    RTDETR_MAPPING,
    BoundingBox,
    LayoutBox,
    LayoutOutput,
)

# ============= Configuration =============


class RTDETRConfig(BaseModel):
    """
    Configuration for RT-DETR layout extractor.

    This is a single-backend model (PyTorch/Transformers only).

    Example:
        >>> config = RTDETRConfig(device="cuda", confidence=0.4)
        >>> extractor = RTDETRLayoutExtractor(config=config)
    """

    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'mps', 'cpu'. "
        "Auto-detects best available if 'cuda' specified but not available.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Custom path to model. If None, uses OMNIDOCS_MODELS_DIR env var or ~/.omnidocs/models/",
    )
    model_name: str = Field(
        default="HuggingPanda/docling-layout",
        description="HuggingFace model ID to use.",
    )
    image_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Input image size for inference.",
    )
    confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections.",
    )

    model_config = ConfigDict(extra="forbid")


# ============= RT-DETR Layout Extractor =============


class RTDETRLayoutExtractor(BaseLayoutExtractor):
    """
    RT-DETR layout extractor using HuggingFace Transformers.

    A transformer-based real-time detection model for document layout.
    Detects: title, text, table, figure, list, formula, captions, headers, footers.

    This is a single-backend model (PyTorch/Transformers only).

    Example:
        >>> from omnidocs.tasks.layout_extraction import RTDETRLayoutExtractor, RTDETRConfig
        >>>
        >>> extractor = RTDETRLayoutExtractor(config=RTDETRConfig(device="cuda"))
        >>> result = extractor.extract(image)
        >>>
        >>> for box in result.bboxes:
        ...     print(f"{box.label.value}: {box.confidence:.2f}")
    """

    def __init__(self, config: RTDETRConfig):
        """
        Initialize RT-DETR layout extractor.

        Args:
            config: Configuration object with device, model settings, etc.
        """
        self.config = config
        self._model = None
        self._processor = None
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

        return Path(models_dir) / "rtdetr_layout"

    def _load_model(self) -> None:
        """Load RT-DETR model from HuggingFace or local cache."""
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for RTDETRLayoutExtractor. Install with: pip install transformers torch timm"
            )

        # Check if model is cached locally
        config_file = self._model_path / "config.json"

        if config_file.exists():
            # Load from local cache
            self._processor = RTDetrImageProcessor.from_pretrained(str(self._model_path))
            self._model = RTDetrForObjectDetection.from_pretrained(str(self._model_path))
        else:
            # Download from HuggingFace
            self._processor = RTDetrImageProcessor.from_pretrained(self.config.model_name)
            self._model = RTDetrForObjectDetection.from_pretrained(self.config.model_name)

            # Cache locally
            self._model_path.mkdir(parents=True, exist_ok=True)
            self._processor.save_pretrained(str(self._model_path))
            self._model.save_pretrained(str(self._model_path))

        # Move to device and set eval mode
        self._model = self._model.to(self._device)
        self._model.eval()

    def extract(self, image: Union[Image.Image, np.ndarray, str, Path]) -> LayoutOutput:
        """
        Run layout extraction on an image.

        Args:
            image: Input image (PIL Image, numpy array, or path)

        Returns:
            LayoutOutput with detected layout boxes
        """
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        img_width, img_height = pil_image.size

        # Preprocess
        inputs = self._processor(
            images=pil_image,
            return_tensors="pt",
            size={"height": self.config.image_size, "width": self.config.image_size},
        )

        # Move to device
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([[img_height, img_width]])
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.confidence,
        )[0]

        # Parse detections
        layout_boxes = []

        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = float(score.item())
            class_id = int(label_id.item())

            # Get original label from model config
            # Note: The model outputs 0-indexed class IDs, but id2label has background at index 0,
            # so we add 1 to map correctly (e.g., model output 8 -> id2label[9] = "Table")
            original_label = self._model.config.id2label.get(class_id + 1, f"class_{class_id}")

            # Map to standardized label
            standard_label = RTDETR_MAPPING.to_standard(original_label)

            # Box coordinates
            box_coords = box.cpu().tolist()

            layout_boxes.append(
                LayoutBox(
                    label=standard_label,
                    bbox=BoundingBox.from_list(box_coords),
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
            model_name="RT-DETR (docling-layout)",
        )
