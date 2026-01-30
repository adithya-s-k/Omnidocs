"""
EasyOCR extractor.

EasyOCR is a PyTorch-based OCR engine with excellent multi-language support.
- GPU accelerated (optional)
- Supports 80+ languages
- Good for scene text and printed documents

Python Package:
    pip install easyocr

Model Download Location:
    By default, EasyOCR downloads models to ~/.EasyOCR/
    Can be overridden with model_storage_directory parameter
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseOCRExtractor
from .models import BoundingBox, OCRGranularity, OCROutput, TextBlock


class EasyOCRConfig(BaseModel):
    """
    Configuration for EasyOCR extractor.

    This is a single-backend model (PyTorch - CPU/GPU).

    Example:
        >>> config = EasyOCRConfig(languages=["en", "ch_sim"], gpu=True)
        >>> ocr = EasyOCR(config=config)
    """

    languages: List[str] = Field(
        default=["en"],
        description="Language codes (e.g., ['en', 'ch_sim', 'ja', 'ko']). "
        "See https://www.jaided.ai/easyocr/ for full list.",
    )
    gpu: bool = Field(
        default=True,
        description="Use GPU if available. Falls back to CPU if not available.",
    )
    model_storage_directory: Optional[str] = Field(
        default=None,
        description="Custom model storage path (default: ~/.EasyOCR/)",
    )
    download_enabled: bool = Field(
        default=True,
        description="Allow automatic model downloads",
    )
    detector: bool = Field(
        default=True,
        description="Use text detection (set False for pre-cropped images)",
    )
    recognizer: bool = Field(
        default=True,
        description="Use text recognition",
    )
    quantize: bool = Field(
        default=True,
        description="Use quantized models (faster, less memory)",
    )

    model_config = ConfigDict(extra="forbid")


class EasyOCR(BaseOCRExtractor):
    """
    EasyOCR text extractor.

    Single-backend model (PyTorch - CPU/GPU).

    Example:
        >>> from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
        >>>
        >>> ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=True))
        >>> result = ocr.extract(image)
        >>>
        >>> for block in result.text_blocks:
        ...     print(f"'{block.text}' @ {block.bbox.to_list()}")
    """

    MODEL_NAME = "easyocr"

    def __init__(self, config: EasyOCRConfig):
        """
        Initialize EasyOCR extractor.

        Args:
            config: Configuration object

        Raises:
            ImportError: If easyocr is not installed
        """
        self.config = config
        self._reader = None
        self._load_model()

    def _load_model(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr is required for EasyOCR. Install with: pip install easyocr")

        # Create model directory if specified
        if self.config.model_storage_directory:
            os.makedirs(self.config.model_storage_directory, exist_ok=True)

        # Initialize reader
        self._reader = easyocr.Reader(
            lang_list=self.config.languages,
            gpu=self.config.gpu,
            model_storage_directory=self.config.model_storage_directory,
            download_enabled=self.config.download_enabled,
            detector=self.config.detector,
            recognizer=self.config.recognizer,
            verbose=False,
            quantize=self.config.quantize,
        )

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        detail: int = 1,
        paragraph: bool = False,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
    ) -> OCROutput:
        """
        Run OCR on an image.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            detail: 0 = simple output, 1 = detailed with boxes
            paragraph: Combine results into paragraphs
            min_size: Minimum text box size
            text_threshold: Text confidence threshold
            low_text: Low text bound
            link_threshold: Link threshold for text joining
            canvas_size: Max image dimension for processing
            mag_ratio: Magnification ratio

        Returns:
            OCROutput with detected text blocks

        Raises:
            ValueError: If detail is not 0 or 1
            RuntimeError: If EasyOCR is not initialized
        """
        if self._reader is None:
            raise RuntimeError("EasyOCR not initialized. Call _load_model() first.")

        # Validate detail parameter
        if detail not in (0, 1):
            raise ValueError(f"detail must be 0 or 1, got {detail}")

        # Prepare image
        pil_image = self._prepare_image(image)
        image_width, image_height = pil_image.size

        # Convert to numpy array for EasyOCR
        image_array = np.array(pil_image)

        # Run EasyOCR
        results = self._reader.readtext(
            image_array,
            detail=detail,
            paragraph=paragraph,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
        )

        # Parse results
        text_blocks = []
        full_text_parts = []

        for result in results:
            if detail == 0:
                # Simple output: just text
                text = result
                confidence = 1.0
                bbox = BoundingBox(x1=0, y1=0, x2=0, y2=0)
                polygon = None
            else:
                # Detailed output: [polygon, text, confidence]
                polygon_points, text, confidence = result

                # EasyOCR returns 4 corner points: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                # Convert to list of lists for storage
                polygon = [list(p) for p in polygon_points]

                # Convert to axis-aligned bounding box
                bbox = BoundingBox.from_polygon(polygon)

            if not text.strip():
                continue

            text_blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=float(confidence),
                    granularity=OCRGranularity.LINE if paragraph else OCRGranularity.WORD,
                    polygon=polygon,
                    language="+".join(self.config.languages),
                )
            )

            full_text_parts.append(text)

        # Sort by position
        text_blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        return OCROutput(
            text_blocks=text_blocks,
            full_text=" ".join(full_text_parts),
            image_width=image_width,
            image_height=image_height,
            model_name=self.MODEL_NAME,
            languages_detected=self.config.languages,
        )

    def extract_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        **kwargs,
    ) -> List[OCROutput]:
        """
        Run OCR on multiple images.

        Args:
            images: List of input images
            **kwargs: Arguments passed to extract()

        Returns:
            List of OCROutput objects
        """
        results = []
        for img in images:
            results.append(self.extract(img, **kwargs))
        return results
