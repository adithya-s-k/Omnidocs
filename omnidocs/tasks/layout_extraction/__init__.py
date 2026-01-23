"""
Layout Extraction Module.

Provides extractors for detecting document layout elements such as
titles, text blocks, figures, tables, formulas, and captions.

Available Extractors:
    - DocLayoutYOLO: YOLO-based layout detector (fast, accurate)
    - RTDETRLayoutExtractor: Transformer-based detector (more categories)
    - QwenLayoutDetector: VLM-based detector with custom label support (multi-backend)

Example:
    >>> from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
    >>>
    >>> extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
    >>> result = extractor.extract(image)
    >>>
    >>> for box in result.bboxes:
    ...     print(f"{box.label.value}: {box.confidence:.2f}")

    >>> # VLM-based detection with custom labels
    >>> from omnidocs.tasks.layout_extraction import QwenLayoutDetector, CustomLabel
    >>> from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
    >>>
    >>> detector = QwenLayoutDetector(
    ...     backend=QwenLayoutPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    ... )
    >>> result = detector.extract(image, custom_labels=["code_block", "sidebar"])
"""

from .base import BaseLayoutExtractor
from .doc_layout_yolo import DocLayoutYOLO, DocLayoutYOLOConfig
from .models import (
    DOCLAYOUT_YOLO_CLASS_NAMES,
    DOCLAYOUT_YOLO_MAPPING,
    LABEL_COLORS,
    NORMALIZED_SIZE,
    RTDETR_CLASS_NAMES,
    RTDETR_MAPPING,
    BoundingBox,
    CustomLabel,
    LabelMapping,
    LayoutBox,
    LayoutLabel,
    LayoutOutput,
)
from .qwen import QwenLayoutDetector
from .rtdetr import RTDETRConfig, RTDETRLayoutExtractor

__all__ = [
    # Base
    "BaseLayoutExtractor",
    # Models
    "LayoutLabel",
    "LabelMapping",
    "BoundingBox",
    "LayoutBox",
    "LayoutOutput",
    "CustomLabel",
    # Mappings
    "DOCLAYOUT_YOLO_MAPPING",
    "DOCLAYOUT_YOLO_CLASS_NAMES",
    "RTDETR_MAPPING",
    "RTDETR_CLASS_NAMES",
    # DocLayout-YOLO
    "DocLayoutYOLO",
    "DocLayoutYOLOConfig",
    # RT-DETR
    "RTDETRLayoutExtractor",
    "RTDETRConfig",
    # Qwen3-VL (multi-backend)
    "QwenLayoutDetector",
]
