"""
VLM layout detector.

A provider-agnostic Vision-Language Model layout detector using litellm.
Works with any cloud API: Gemini, OpenRouter, Azure, OpenAI, Anthropic, etc.

Example:
    ```python
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.layout_extraction import VLMLayoutDetector

    config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
    detector = VLMLayoutDetector(config=config)
    result = detector.extract("document.png")

    for box in result.bboxes:
        print(f"{box.label.value}: {box.bbox}")
    ```
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from omnidocs.vlm import VLMAPIConfig, vlm_completion

from .base import BaseLayoutExtractor
from .models import (
    BoundingBox,
    CustomLabel,
    LayoutBox,
    LayoutLabel,
    LayoutOutput,
)

DEFAULT_LAYOUT_LABELS = [
    "title",
    "text",
    "list",
    "table",
    "figure",
    "caption",
    "formula",
    "footnote",
    "page_header",
    "page_footer",
]


def _build_layout_prompt(labels: List[str]) -> str:
    """Build detection prompt for VLM layout detection."""
    labels_str = ", ".join(labels)
    return (
        f"Detect all layout elements in this document image. "
        f"Identify elements from these categories: {labels_str}. "
        f"Output as a JSON array with format: "
        f'[{{"bbox": [x1, y1, x2, y2], "label": "element_type"}}, ...] '
        f"where coordinates are in absolute pixels relative to the image dimensions. "
        f"Return ONLY the JSON array, no other text."
    )


def _parse_layout_response(raw_output: str, image_size: tuple[int, int]) -> List[Dict[str, Any]]:
    """Parse JSON layout response, handling markdown fencing and truncation."""
    output = raw_output.strip()

    # Remove markdown fencing if present
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            output = "\n".join(lines[i + 1 :])
            output = output.split("```")[0]
            break

    # Try direct parsing
    try:
        result = json.loads(output)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    pattern = (
        r'\{"bbox"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,'
        r'\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]\s*,\s*"label"\s*:\s*"([^"]+)"\}'
    )
    matches = re.findall(pattern, output)

    results = []
    width, height = image_size
    for match in matches:
        x1, y1, x2, y2, label = match
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        # Basic validation
        if bbox[0] < bbox[2] and bbox[1] < bbox[3] and bbox[2] <= width * 1.1 and bbox[3] <= height * 1.1:
            results.append({"bbox": bbox, "label": label})

    return results


class VLMLayoutDetector(BaseLayoutExtractor):
    """
    Provider-agnostic VLM layout detector using litellm.

    Works with any cloud VLM API: Gemini, OpenRouter, Azure, OpenAI,
    Anthropic, etc. Supports custom labels for flexible detection.

    Example:
        ```python
        from omnidocs.vlm import VLMAPIConfig
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
        detector = VLMLayoutDetector(config=config)

        # Default labels
        result = detector.extract("document.png")

        # Custom labels
        result = detector.extract("document.png", custom_labels=["code_block", "sidebar"])
        ```
    """

    def __init__(self, config: VLMAPIConfig):
        """
        Initialize VLM layout detector.

        Args:
            config: VLM API configuration with model and provider details.
        """
        self.config = config
        self._loaded = True

    def _load_model(self) -> None:
        """No-op for API-only detector."""
        pass

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        custom_labels: Optional[List[Union[str, CustomLabel]]] = None,
        prompt: Optional[str] = None,
    ) -> LayoutOutput:
        """
        Run layout detection on an image.

        Args:
            image: Input image (PIL Image, numpy array, or file path).
            custom_labels: Optional custom labels to detect. Can be:
                - None: Use default labels (title, text, table, figure, etc.)
                - List[str]: Simple label names ["code_block", "sidebar"]
                - List[CustomLabel]: Typed labels with metadata
            prompt: Custom prompt. If None, builds a default detection prompt.

        Returns:
            LayoutOutput with detected layout boxes.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Normalize labels
        label_names = self._normalize_labels(custom_labels)

        # Build or use custom prompt
        final_prompt = prompt or _build_layout_prompt(label_names)

        raw_output = vlm_completion(self.config, final_prompt, pil_image)
        detections = _parse_layout_response(raw_output, (width, height))
        layout_boxes = self._build_layout_boxes(detections, width, height)

        # Sort by reading order
        layout_boxes.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        return LayoutOutput(
            bboxes=layout_boxes,
            image_width=width,
            image_height=height,
            model_name=f"VLM ({self.config.model})",
        )

    def _normalize_labels(self, custom_labels: Optional[List[Union[str, CustomLabel]]]) -> List[str]:
        """Normalize labels to list of strings."""
        if custom_labels is None:
            return DEFAULT_LAYOUT_LABELS

        label_names = []
        for label in custom_labels:
            if isinstance(label, str):
                label_names.append(label)
            elif isinstance(label, CustomLabel):
                label_names.append(label.name)
            else:
                raise TypeError(f"Expected str or CustomLabel, got {type(label).__name__}")
        return label_names

    def _build_layout_boxes(self, detections: List[Dict[str, Any]], width: int, height: int) -> List[LayoutBox]:
        """Convert parsed detections to LayoutBox objects."""
        layout_boxes = []

        for det in detections:
            if "bbox" not in det or "label" not in det:
                continue

            bbox = det["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            if not all(isinstance(c, (int, float)) for c in bbox):
                continue

            label_str = det["label"].lower()

            # Clamp to image bounds
            abs_bbox = [
                round(max(0, min(float(bbox[0]), width)), 2),
                round(max(0, min(float(bbox[1]), height)), 2),
                round(max(0, min(float(bbox[2]), width)), 2),
                round(max(0, min(float(bbox[3]), height)), 2),
            ]

            # Map to standard label if possible
            try:
                standard_label = LayoutLabel(label_str)
            except ValueError:
                standard_label = LayoutLabel.UNKNOWN

            layout_boxes.append(
                LayoutBox(
                    label=standard_label,
                    bbox=BoundingBox.from_list(abs_bbox),
                    confidence=1.0,
                    original_label=label_str,
                )
            )

        return layout_boxes
