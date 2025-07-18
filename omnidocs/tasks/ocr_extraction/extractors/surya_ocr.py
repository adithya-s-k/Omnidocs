import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText
import os

logger = get_logger(__name__)

class SuryaOCRMapper(BaseOCRMapper):
    """Label mapper for Surya OCR model output."""
    
    def __init__(self):
        super().__init__('surya')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup language mappings for Surya OCR."""
        # Surya supports many languages with standard ISO codes
        mapping = {
            'en': 'en',
            'hi': 'hi',
            'zh': 'zh',
            'es': 'es',
            'fr': 'fr',
            'ar': 'ar',
            'bn': 'bn',
            'ru': 'ru',
            'pt': 'pt',
            'ur': 'ur',
            'de': 'de',
            'ja': 'ja',
            'sw': 'sw',
            'mr': 'mr',
            'te': 'te',
            'tr': 'tr',
            'ta': 'ta',
            'vi': 'vi',
            'ko': 'ko',
            'it': 'it',
            'th': 'th',
            'gu': 'gu',
            'pl': 'pl',
            'uk': 'uk',
            'kn': 'kn',
            'ml': 'ml',
            'or': 'or',
            'pa': 'pa',
            'ne': 'ne',
            'si': 'si',
            'my': 'my',
            'km': 'km',
            'lo': 'lo',
            'ka': 'ka',
            'am': 'am',
            'he': 'he',
            'fa': 'fa',
            'ps': 'ps',
            'dv': 'dv',
            'ti': 'ti',
            'ny': 'ny',
            'so': 'so',
            'cy': 'cy',
            'eu': 'eu',
            'be': 'be',
            'is': 'is',
            'mt': 'mt',
            'lb': 'lb',
            'fo': 'fo',
            'yi': 'yi',
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class SuryaOCRExtractor(BaseOCRExtractor):
    """Surya OCR based text extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize Surya OCR Extractor."""
        super().__init__(
            device=device, 
            show_log=show_log, 
            languages=languages or ['en'],
            engine_name='surya'
        )
        
        self._label_mapper = SuryaOCRMapper()
        
        try:
            # Updated imports based on the new API structure
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            
            self.RecognitionPredictor = RecognitionPredictor
            self.DetectionPredictor = DetectionPredictor
            
        except ImportError as e:
            logger.error("Failed to import surya")
            raise ImportError(
                "surya-ocr is not available. Please install it with: pip install surya-ocr"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Path:
        """Model download handled by surya library."""
        if self.show_log:
            logger.info("Model downloading handled by surya library")
        return None
    
    def _load_model(self) -> None:
        """Load Surya OCR models."""
        try:
            # Set up omnidocs/models directory for HuggingFace cache
            current_file = Path(__file__)
            omnidocs_root = current_file.parent.parent.parent.parent  # Go up to omnidocs root
            models_dir = omnidocs_root / "models"
            models_dir.mkdir(exist_ok=True)

            # Set environment variables for HuggingFace cache
            import os
            os.environ["HF_HOME"] = str(models_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
            os.environ["HF_HUB_CACHE"] = str(models_dir)

            if self.show_log:
                logger.info("Loading Surya OCR models")
                logger.info(f"Models will be downloaded in: {models_dir}")

            # Initialize predictors - the new API handles model loading internally
            self.recognition_predictor = self.RecognitionPredictor()
            self.detection_predictor = self.DetectionPredictor()
            
            if self.show_log:
                logger.info(f"Surya OCR models loaded on device: {self.device}")
        
        except Exception as e:
            logger.error("Failed to load Surya OCR models", exc_info=True)
            raise
    
    def postprocess_output(self, raw_output: List, img_size: Tuple[int, int]) -> OCROutput:
        """Convert Surya OCR output to standardized OCROutput format."""
        texts = []
        full_text_parts = []
        
        # raw_output is a list of predictions, one per image
        # We're processing only one image, so take the first result
        if not raw_output:
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=img_size
            )
        
        prediction = raw_output[0]
        
        # Extract text lines from the prediction
        text_lines = prediction.text_lines if hasattr(prediction, 'text_lines') else []
        
        for i, text_line in enumerate(text_lines):
            if hasattr(text_line, 'text') and hasattr(text_line, 'bbox'):
                text = text_line.text.strip()
                if not text:
                    continue
                
                # Get bounding box
                bbox = text_line.bbox
                bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                
                # Get confidence if available
                confidence = getattr(text_line, 'confidence', 0.9)
                
                # Detect language
                detected_lang = self.detect_text_language(text)
                
                ocr_text = OCRText(
                    text=text,
                    confidence=float(confidence),
                    bbox=bbox_list,
                    language=detected_lang,
                    reading_order=i
                )
                
                texts.append(ocr_text)
                full_text_parts.append(text)
        
        return OCROutput(
            texts=texts,
            full_text=' '.join(full_text_parts),
            source_img_size=img_size
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> OCROutput:
        """Extract text using Surya OCR."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            img = images[0]
            
            # Map languages to Surya format
            surya_languages = []
            for lang in self.languages:
                mapped_lang = self._label_mapper.from_standard_language(lang)
                surya_languages.append(mapped_lang)
            
            # FIXED: Pass None instead of [surya_languages] to use default task
            # According to the API documentation, you should pass None for languages
            # or not pass them at all to let the model auto-detect
            raw_output = self.recognition_predictor(
                [img],
                None,  # Pass None instead of [surya_languages]
                self.detection_predictor
            )
            
            # Convert to standardized format
            result = self.postprocess_output(raw_output, img.size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.texts)} text regions")
            
            return result
            
        except Exception as e:
            logger.error("Error during Surya OCR extraction", exc_info=True)
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )