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
        
        # Set default model path
        self.model_path = Path("omnidocs/models/surya")
        
        # Check dependencies
        self._check_dependencies()
        
        # Download model if needed
        if not self.model_path.exists():
            self._download_model()
        
        self._load_model()

    def _download_model(self) -> Path:
        """Download Surya OCR models if needed."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.show_log:
            logger.info(f"Surya OCR models will be downloaded to: {self.model_path}")
        return self.model_path
    
    def _load_model(self) -> None:
        """Load Surya OCR models."""
        try:
            # Set up omnidocs/models directory for HuggingFace cache
            current_file = Path(__file__)
            omnidocs_root = current_file.parent.parent.parent.parent
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

            # Initialize predictors using the new API
            if hasattr(self, 'use_new_api') and self.use_new_api:
                # Use the new Predictor-based API
                self.det_predictor = self.DetectionPredictor()
                self.rec_predictor = self.RecognitionPredictor()
            else:
                # Fallback to old API (shouldn't happen with current version)
                from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
                from surya.model.recognition.model import load_model as load_rec_model, load_processor as load_rec_processor

                self.det_model = load_det_model()
                self.det_processor = load_det_processor()
                self.rec_model = load_rec_model()
                self.rec_processor = load_rec_processor()
            
            if self.show_log:
                logger.info(f"Surya OCR models loaded on device: {self.device}")
        
        except Exception as e:
            logger.error("Failed to load Surya OCR models", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            import cv2
            import numpy as np
            from PIL import Image
            
            # Check if surya-ocr is installed
            try:
                import surya
                if self.show_log:
                    logger.info(f"Found surya package at: {surya.__file__}")
            except ImportError:
                raise ImportError(
                    "surya-ocr package not found. Please install with: "
                    "pip install surya-ocr"
                )
            
            # Try to import the current API functions (surya-ocr 0.14.6+)
            try:
                # Current API structure uses Predictor classes
                from surya.detection import DetectionPredictor
                from surya.recognition import RecognitionPredictor, convert_if_not_rgb

                # Store the classes for later use
                self.DetectionPredictor = DetectionPredictor
                self.RecognitionPredictor = RecognitionPredictor
                self.convert_if_not_rgb = convert_if_not_rgb
                self.use_new_api = True

                if self.show_log:
                    logger.info("Successfully imported Surya OCR with new API")

            except ImportError as import_err:
                logger.error(f"Failed to import Surya OCR dependencies: {import_err}")
                raise ImportError(
                    "Required dependencies not available. Please install with: "
                    "pip install surya-ocr==0.14.6 torch opencv-python"
                ) from import_err
            
        except ImportError as e:
            logger.error(f"Failed to import Surya OCR dependencies: {e}")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install surya-ocr torch opencv-python"
            ) from e
    
    def postprocess_output(self, raw_output: Union[List, Any], img_size: Tuple[int, int]) -> OCROutput:
        """Convert Surya OCR output to standardized OCROutput format."""
        texts = []
        full_text_parts = []
        
        if not raw_output:
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=img_size
            )
        
        try:
            # Handle different output formats from different Surya versions
            if isinstance(raw_output, list) and len(raw_output) > 0:
                prediction = raw_output[0]
                
                # Check for different attribute names based on version
                text_lines = None
                if hasattr(prediction, 'text_lines'):
                    text_lines = prediction.text_lines
                elif hasattr(prediction, 'bboxes') and hasattr(prediction, 'text'):
                    # Handle case where we have separate bboxes and text
                    if hasattr(prediction, 'text') and isinstance(prediction.text, list):
                        text_lines = []
                        for i, (bbox, text) in enumerate(zip(prediction.bboxes, prediction.text)):
                            # Create a mock text_line object
                            class MockTextLine:
                                def __init__(self, text, bbox):
                                    self.text = text
                                    self.bbox = bbox
                                    self.confidence = 0.9  # Default confidence
                            text_lines.append(MockTextLine(text, bbox))
                
                if text_lines:
                    for i, text_line in enumerate(text_lines):
                        if hasattr(text_line, 'text') and hasattr(text_line, 'bbox'):
                            text = text_line.text.strip() if text_line.text else ""
                            if not text:
                                continue
                            
                            bbox = text_line.bbox
                            # Ensure bbox is in the correct format [x1, y1, x2, y2]
                            if len(bbox) >= 4:
                                bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                            else:
                                continue
                            
                            # Create polygon from bbox
                            polygon = [
                                [float(bbox[0]), float(bbox[1])], 
                                [float(bbox[2]), float(bbox[1])],
                                [float(bbox[2]), float(bbox[3])], 
                                [float(bbox[0]), float(bbox[3])]
                            ]

                            confidence = getattr(text_line, 'confidence', 0.9)
                            detected_lang = self.detect_text_language(text)

                            ocr_text = OCRText(
                                text=text,
                                confidence=float(confidence),
                                bbox=bbox_list,
                                polygon=polygon,
                                language=detected_lang,
                                reading_order=i
                            )
                            
                            texts.append(ocr_text)
                            full_text_parts.append(text)
            
        except Exception as e:
            logger.error(f"Error processing Surya OCR output: {e}", exc_info=True)
        
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
            
            # Use the new Predictor-based API
            predictions = None

            if hasattr(self, 'use_new_api') and self.use_new_api:
                # Use the new Predictor-based API based on surya scripts
                try:
                    # Convert image to RGB if needed (function expects a list)
                    img_rgb_list = self.convert_if_not_rgb([img])
                    img_rgb = img_rgb_list[0]

                    # Import TaskNames for proper task specification
                    from surya.common.surya.schema import TaskNames

                    # Call RecognitionPredictor directly with det_predictor parameter
                    # This is how it's done in surya/scripts/ocr_text.py
                    predictions = self.rec_predictor(
                        [img_rgb],
                        task_names=[TaskNames.ocr_with_boxes],
                        det_predictor=self.det_predictor,
                        math_mode=False
                    )

                except Exception as e:
                    if self.show_log:
                        logger.warning(f"New API failed: {e}")

            else:
                # Fallback to old API (shouldn't happen with current version)
                if hasattr(self, 'run_ocr'):
                    try:
                        predictions = self.run_ocr(
                            [img],
                            [surya_languages],
                            self.det_model,
                            self.det_processor,
                            self.rec_model,
                            self.rec_processor
                        )
                    except Exception as e:
                        if self.show_log:
                            logger.warning(f"run_ocr failed: {e}")

            if predictions is None:
                raise RuntimeError("Failed to run OCR with available Surya API functions")
            
            # Convert to standardized format
            result = self.postprocess_output(predictions, img.size)
            
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