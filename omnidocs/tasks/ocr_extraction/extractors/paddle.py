import os
import sys
from pathlib import Path
import time
import copy
import base64
import cv2
import numpy as np
from io import BytesIO
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import os

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText


# Set up model directory for PaddleOCR downloads
def _setup_paddle_model_dir():
    """Set up the model directory for PaddleOCR to use omnidocs/models."""
    current_file = Path(__file__)
    omnidocs_root = current_file.parent.parent.parent.parent  # Go up to omnidocs root
    models_dir = omnidocs_root / "models" / "paddleocr"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkey-patch PaddleOCR's BASE_DIR before importing
    import paddleocr.paddleocr
    paddleocr.paddleocr.BASE_DIR = str(models_dir) + "/"
    
    # Also patch the network module
    import paddleocr.ppocr.utils.network
    paddleocr.ppocr.utils.network.MODELS_DIR = str(models_dir) + "/models/"
    
    return models_dir

# Call this immediately
_MODELS_DIR = _setup_paddle_model_dir()


logger = get_logger(__name__)

# Utility functions
def alpha_to_color(img, alpha_color=(255, 255, 255)):
    """Convert transparent pixels to specified color."""
    if len(img.shape) == 4:  # RGBA
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        transparent_mask = alpha_channel == 0
        
        for i in range(3):
            rgb_channels[:, :, i][transparent_mask] = alpha_color[i]
        
        return rgb_channels
    return img

def binarize_img(img):
    """Convert image to binary (black and white)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def points_to_bbox(points):
    """Change polygon(shape: N * 8) to bbox(shape: N * 4)."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

class PaddleOCRMapper(BaseOCRMapper):
    """Label mapper for PaddleOCR model output."""
    
    def __init__(self):
        super().__init__('paddleocr')
        self._mapping = {
            'en': 'en',
            'ch': 'ch',
            'chinese_cht': 'chinese_cht',
            'ta': 'ta',
            'te': 'te',
            'ka': 'ka',
            'ja': 'japan',
            'ko': 'korean',
            'hi': 'hi',
            'ar': 'ar',
            'cyrillic': 'cyrillic',
            'devanagari': 'devanagari',
            'fr': 'fr',
            'de': 'german',
            'es': 'es',
            'pt': 'pt',
            'ru': 'ru',
            'it': 'it',
        }
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}

class PaddleOCRExtractor(BaseOCRExtractor):
    """PaddleOCR based text extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        drop_score: float = 0.5,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize PaddleOCR Extractor."""
        super().__init__(
            device=device, 
            show_log=show_log, 
            languages=languages or ['en'],
            engine_name='paddle'
        )
        
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.drop_score = drop_score
        self._label_mapper = PaddleOCRMapper()
        
        # Set default paths
        if model_path is None:
            model_path = "omnidocs/models/paddleocr"
        self.model_path = Path(model_path)
        
        # Check dependencies first
        self._check_dependencies()
        
        # Set up model directory and download if needed
        if self.model_path.exists() and any(self.model_path.iterdir()):
            if self.show_log:
                logger.info(f"Using existing PaddleOCR models from: {self.model_path}")
        elif not self.model_path.exists():
            self._download_model()
        
        # Load model
        self._load_model()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            import cv2
            import numpy as np
            from PIL import Image
            from paddleocr import PaddleOCR
            self.PaddleOCR = PaddleOCR
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install paddleocr paddlepaddle torch opencv-python"
            ) from e

    def _download_model(self) -> Path:
        """Download PaddleOCR models if needed (required abstract method)."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.show_log:
            logger.info(f"PaddleOCR models will be downloaded to: {self.model_path}")
            logger.info("Models will be downloaded automatically by PaddleOCR on first use")
        return self.model_path

    def _load_model(self) -> None:
        """Load PaddleOCR model."""
        try:
            # Map languages to PaddleOCR format
            primary_lang = self._get_primary_language()
            
            if self.show_log:
                logger.info(f"Loading PaddleOCR with language: {primary_lang}")
                logger.info(f"Models directory: {_MODELS_DIR}")
            
            # Initialize PaddleOCR - it will handle model downloads automatically
            self.model = self.PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=primary_lang,
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                drop_score=self.drop_score
            )
            
            if self.show_log:
                logger.info("PaddleOCR model loaded successfully")
        
        except Exception as e:
            logger.error("Failed to load PaddleOCR models", exc_info=True)
            raise

    def _get_primary_language(self) -> str:
        """Get primary language for PaddleOCR."""
        if not self.languages:
            return 'en'
        
        # Map first language to PaddleOCR format
        primary_lang = self._label_mapper.from_standard_language(self.languages[0])
        return primary_lang if primary_lang else 'en'
    
    def preprocess_image(self, image, alpha_color=(255, 255, 255), inv=False, bin=False):
        """Preprocess image for OCR."""
        image = alpha_to_color(image, alpha_color)
        if inv:
            image = cv2.bitwise_not(image)
        if bin:
            image = binarize_img(image)
        return image
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> OCROutput:
        """Extract text using PaddleOCR."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            img = images[0]
            
            # Convert PIL to cv2 format if needed
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Perform OCR
            result = self.model.ocr(img, cls=self.use_angle_cls)
            
            # Convert to standardized format
            texts = self._process_ocr_results(result)
            full_text_parts = [text.text for text in texts]
            
            img_size = img.shape[:2][::-1]  # (width, height)
            
            ocr_output = OCROutput(
                texts=texts,
                full_text=' '.join(full_text_parts),
                source_img_size=img_size
            )
            
            if self.show_log:
                logger.info(f"Extracted {len(texts)} text regions")
            
            return ocr_output
            
        except Exception as e:
            logger.error("Error during PaddleOCR extraction", exc_info=True)
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )

    def _process_ocr_results(self, result: List) -> List[OCRText]:
        """Process raw OCR results into OCRText objects."""
        texts = []
        
        if not result or not result[0]:
            return texts
        
        for i, detection in enumerate(result[0]):
            bbox_points, (text, confidence) = detection
            
            if confidence < self.drop_score or not text.strip():
                continue
            
            text = text.strip()
            
            # Convert points to bbox format
            bbox = points_to_bbox(bbox_points)
            polygon = [[float(x), float(y)] for x, y in bbox_points]
            
            # Detect language
            detected_lang = self.detect_text_language(text)
            
            ocr_text = OCRText(
                text=text,
                confidence=float(confidence),
                bbox=[float(coord) for coord in bbox],
                polygon=polygon,
                language=detected_lang,
                reading_order=i
            )
            texts.append(ocr_text)
        
        return texts
    
    def predict(self, img, **kwargs):
        """Predict method for compatibility with original interface."""
        try:
            result = self.extract(img, **kwargs)
            
            # Convert to original format
            ocr_res = []
            for text_obj in result.texts:
                # Convert bbox back to points format
                x0, y0, x1, y1 = text_obj.bbox
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                poly = [coord for point in points for coord in point]
                
                ocr_res.append({
                    "category_type": "text",
                    'poly': poly,
                    'score': text_obj.confidence,
                    'text': text_obj.text,
                })
            
            return ocr_res
            
        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            return []