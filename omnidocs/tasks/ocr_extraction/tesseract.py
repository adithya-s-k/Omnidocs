"""
Tesseract OCR extractor.

Tesseract is an open-source OCR engine maintained by Google.
- CPU-based (no GPU required)
- Requires system installation of Tesseract
- Good for printed text, supports 100+ languages

System Requirements:
    macOS: brew install tesseract
    Ubuntu: sudo apt-get install tesseract-ocr
    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Python Package:
    pip install pytesseract
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseOCRExtractor
from .models import BoundingBox, OCRGranularity, OCROutput, TextBlock


class TesseractOCRConfig(BaseModel):
    """
    Configuration for Tesseract OCR extractor.

    This is a single-backend model (CPU only, requires system Tesseract).

    Example:
        >>> config = TesseractOCRConfig(languages=["eng", "fra"], psm=3)
        >>> ocr = TesseractOCR(config=config)
    """

    languages: List[str] = Field(
        default=["eng"],
        description="Language codes (e.g., ['eng', 'fra', 'deu']). "
        "Use 'tesseract --list-langs' to see available languages.",
    )
    tessdata_dir: Optional[str] = Field(
        default=None,
        description="Custom tessdata directory path. Overrides TESSDATA_PREFIX env var.",
    )
    oem: int = Field(
        default=3,
        ge=0,
        le=3,
        description="OCR Engine Mode: 0=Legacy, 1=LSTM only, 2=Legacy+LSTM, 3=Default (based on availability)",
    )
    psm: int = Field(
        default=3,
        ge=0,
        le=13,
        description="Page Segmentation Mode: "
        "0=OSD only, 3=Fully automatic (default), 6=Uniform block, "
        "7=Single line, 11=Sparse text, 13=Raw line",
    )
    config_params: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional Tesseract config parameters (e.g., {'tessedit_char_whitelist': '0123456789'})",
    )

    model_config = ConfigDict(extra="forbid")


def _check_tesseract_installation() -> bool:
    """Check if Tesseract is installed on the system."""
    return shutil.which("tesseract") is not None


def _get_tesseract_languages() -> List[str]:
    """Get list of installed Tesseract languages."""
    try:
        import pytesseract

        return pytesseract.get_languages()
    except Exception:
        return []


class TesseractOCR(BaseOCRExtractor):
    """
    Tesseract OCR extractor.

    Single-backend model (CPU only). Requires system Tesseract installation.

    Example:
        >>> from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
        >>>
        >>> ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        >>> result = ocr.extract(image)
        >>>
        >>> for block in result.text_blocks:
        ...     print(f"'{block.text}' @ {block.bbox.to_list()}")
    """

    MODEL_NAME = "tesseract"

    def __init__(self, config: TesseractOCRConfig):
        """
        Initialize Tesseract OCR extractor.

        Args:
            config: Configuration object

        Raises:
            RuntimeError: If Tesseract is not installed
            ImportError: If pytesseract is not installed
        """
        self.config = config
        self._pytesseract = None
        self._load_model()

    def _load_model(self) -> None:
        """Initialize Tesseract OCR engine."""
        # Check system installation
        if not _check_tesseract_installation():
            raise RuntimeError(
                "Tesseract is not installed on your system.\n"
                "Installation instructions:\n"
                "  macOS:   brew install tesseract\n"
                "  Ubuntu:  sudo apt-get install tesseract-ocr\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )

        # Import pytesseract
        try:
            import pytesseract

            self._pytesseract = pytesseract
        except ImportError:
            raise ImportError("pytesseract is required for TesseractOCR. Install with: pip install pytesseract")

        # Set tessdata directory if provided
        if self.config.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self.config.tessdata_dir

        # Verify languages are available
        available = _get_tesseract_languages()
        for lang in self.config.languages:
            if lang not in available:
                import warnings

                warnings.warn(
                    f"Language '{lang}' may not be installed. "
                    f"Available languages: {available[:10]}{'...' if len(available) > 10 else ''}"
                )

    def extract(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
        """
        Run OCR on an image.

        Args:
            image: Input image (PIL Image, numpy array, or path)

        Returns:
            OCROutput with detected text blocks at word level
        """
        if self._pytesseract is None:
            raise RuntimeError("Tesseract not initialized. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        image_width, image_height = pil_image.size

        # Build config string
        config = f"--oem {self.config.oem} --psm {self.config.psm}"
        if self.config.config_params:
            for key, value in self.config.config_params.items():
                config += f" -c {key}={value}"

        # Language string
        lang_str = "+".join(self.config.languages)

        # Get detailed data (word-level boxes)
        data = self._pytesseract.image_to_data(
            pil_image,
            lang=lang_str,
            config=config,
            output_type=self._pytesseract.Output.DICT,
        )

        # Parse results into TextBlocks
        text_blocks = []
        full_text_parts = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            # Safely convert conf to float (handles string values from some Tesseract versions)
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1

            # Skip empty text or low confidence (-1 means no confidence)
            if not text or conf == -1:
                continue

            # Tesseract returns confidence as 0-100, normalize to 0-1
            confidence = conf / 100.0

            # Get bounding box
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            bbox = BoundingBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
            )

            text_blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    granularity=OCRGranularity.WORD,
                    language=lang_str,
                )
            )

            full_text_parts.append(text)

        # Sort by position (top to bottom, left to right)
        text_blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        return OCROutput(
            text_blocks=text_blocks,
            full_text=" ".join(full_text_parts),
            image_width=image_width,
            image_height=image_height,
            model_name=self.MODEL_NAME,
            languages_detected=self.config.languages,
        )

    def extract_lines(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
        """
        Run OCR and return line-level blocks.

        Groups words into lines based on Tesseract's line detection.

        Args:
            image: Input image (PIL Image, numpy array, or path)

        Returns:
            OCROutput with line-level text blocks
        """
        if self._pytesseract is None:
            raise RuntimeError("Tesseract not initialized. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        image_width, image_height = pil_image.size

        # Build config string (including config_params like extract method)
        config = f"--oem {self.config.oem} --psm {self.config.psm}"
        if self.config.config_params:
            for key, value in self.config.config_params.items():
                config += f" -c {key}={value}"

        # Language string
        lang_str = "+".join(self.config.languages)

        # Get detailed data
        data = self._pytesseract.image_to_data(
            pil_image,
            lang=lang_str,
            config=config,
            output_type=self._pytesseract.Output.DICT,
        )

        # Group words into lines
        lines: Dict[tuple, Dict] = {}
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            # Safely convert conf to float (handles string values from some Tesseract versions)
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1

            if not text or conf == -1:
                continue

            # Tesseract provides block_num, par_num, line_num
            line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            if line_key not in lines:
                lines[line_key] = {
                    "words": [],
                    "confidences": [],
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                }

            lines[line_key]["words"].append(text)
            lines[line_key]["confidences"].append(conf / 100.0)
            lines[line_key]["x1"] = min(lines[line_key]["x1"], x)
            lines[line_key]["y1"] = min(lines[line_key]["y1"], y)
            lines[line_key]["x2"] = max(lines[line_key]["x2"], x + w)
            lines[line_key]["y2"] = max(lines[line_key]["y2"], y + h)

        # Convert to TextBlocks
        text_blocks = []
        full_text_parts = []

        for line_key in sorted(lines.keys()):
            line = lines[line_key]
            line_text = " ".join(line["words"])
            avg_conf = sum(line["confidences"]) / len(line["confidences"])

            bbox = BoundingBox(
                x1=float(line["x1"]),
                y1=float(line["y1"]),
                x2=float(line["x2"]),
                y2=float(line["y2"]),
            )

            text_blocks.append(
                TextBlock(
                    text=line_text,
                    bbox=bbox,
                    confidence=avg_conf,
                    granularity=OCRGranularity.LINE,
                    language=lang_str,
                )
            )

            full_text_parts.append(line_text)

        return OCROutput(
            text_blocks=text_blocks,
            full_text="\n".join(full_text_parts),
            image_width=image_width,
            image_height=image_height,
            model_name=self.MODEL_NAME,
            languages_detected=self.config.languages,
        )
