"""
Layout Extraction Module.

Provides extractors for detecting document layout elements such as
titles, text blocks, figures, tables, formulas, and captions.

Available Extractors:
    - DocLayoutYOLO: YOLO-based layout detector (fast, accurate)
    - RTDETRLayoutExtractor: Transformer-based detector (more categories)

Example:
    >>> from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
    >>>
    >>> extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
    >>> result = extractor.extract(image)
    >>>
    >>> for box in result.bboxes:
    ...     print(f"{box.label.value}: {box.confidence:.2f}")
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
    LabelMapping,
    LayoutBox,
    LayoutLabel,
    LayoutOutput,
)
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
]
