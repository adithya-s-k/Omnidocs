import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText

logger = get_logger(__name__)

class EasyOCRMapper(BaseOCRMapper):
    """Label mapper for EasyOCR model output."""
    
    def __init__(self):
        super().__init__('easyocr')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup language mappings for EasyOCR."""
        # EasyOCR uses its own language codes
        mapping = {
            'en': 'en',
            'ch_sim': 'zh',
            'ch_tra': 'zh-TW',
            'ko': 'ko',
            'ja': 'ja',
            'th': 'th',
            'vi': 'vi',
            'ar': 'ar',
            'fr': 'fr',
            'de': 'de',
            'es': 'es',
            'pt': 'pt',
            'ru': 'ru',
            'it': 'it',
            'hi': 'hi',
            'tr': 'tr',
            'pl': 'pl',
            'nl': 'nl',
            'sv': 'sv',
            'da': 'da',
            'no': 'no',
            'fi': 'fi',
            'cs': 'cs',
            'hu': 'hu',
            'ro': 'ro',
            'bg': 'bg',
            'hr': 'hr',
            'sk': 'sk',
            'sl': 'sl',
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class EasyOCRExtractor(BaseOCRExtractor):
    """EasyOCR based text extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        **kwargs
    ):
        """Initialize EasyOCR Extractor."""
        super().__init__(
            device=device, 
            show_log=show_log, 
            languages=languages or ['en'],
            engine_name='easyocr'
        )
        
        self.gpu = gpu 
        self._label_mapper = EasyOCRMapper()
        
        # Set default model path
        self.model_path = Path("omnidocs/models/easyocr")
        
        # Check dependencies
        self._check_dependencies()
        
        # Download model if needed
        if not self.model_path.exists():
            self._download_model()
        
        self._load_model()

    def _download_model(self) -> Path:
        """Download EasyOCR models if needed."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.show_log:
            logger.info(f"EasyOCR models will be downloaded to: {self.model_path}")
        return self.model_path
    
    def _load_model(self) -> None:
        """Load EasyOCR model."""
        try:
            # Map languages to EasyOCR format
            easyocr_languages = []
            for lang in self.languages:
                mapped_lang = self._label_mapper.from_standard_language(lang)
                easyocr_languages.append(mapped_lang)
            
            self.model = self.easyocr.Reader(
                easyocr_languages,
                gpu=self.gpu,
                verbose=self.show_log
            )
            
            if self.show_log:
                logger.info(f"EasyOCR model loaded with languages: {easyocr_languages}")
                
        except Exception as e:
            logger.error("Failed to load EasyOCR model", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            import cv2
            import numpy as np
            from PIL import Image
            import easyocr
            self.easyocr = easyocr
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install easyocr torch opencv-python"
            ) from e
    
    def postprocess_output(self, raw_output: List, img_size: Tuple[int, int]) -> OCROutput:
        """Convert EasyOCR output to standardized OCROutput format."""
        texts = []
        full_text_parts = []
        
        for i, detection in enumerate(raw_output):
            if isinstance(detection, str):
                text = detection
                confidence = 0.9
                bbox = [0, 0, img_size[0], img_size[1]]
                polygon = [[0, 0], [img_size[0], 0], [img_size[0], img_size[1]], [0, img_size[1]]]
            elif isinstance(detection, (list, tuple)) and len(detection) == 3:
                bbox_coords, text, confidence = detection
                
                bbox_array = np.array(bbox_coords)
                x1, y1 = bbox_array.min(axis=0)
                x2, y2 = bbox_array.max(axis=0)
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                
                polygon = [[float(x), float(y)] for x, y in bbox_coords]
            else:
                continue
            
            detected_lang = self.detect_text_language(text)
            
            ocr_text = OCRText(
                text=text,
                confidence=float(confidence),
                bbox=bbox,
                polygon=polygon,
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
        detail: int = 1,  # Changed default to 1 for bbox and confidence
        paragraph: bool = False,
        width_ths: float = 0.7,
        height_ths: float = 0.7,
        **kwargs
    ) -> OCROutput:
        """Extract text using EasyOCR."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            img = images[0]
            
            # Convert PIL to numpy array
            img_array = np.array(img)
            
            # Run OCR
            raw_output = self.model.readtext(
                img_array,
                detail=detail,
                paragraph=paragraph,
                width_ths=width_ths,
                height_ths=height_ths,
                **kwargs
            )
            
            # Convert to standardized format
            result = self.postprocess_output(raw_output, img.size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.texts)} text regions")
            
            return result
            
        except Exception as e:
            logger.error("Error during EasyOCR extraction", exc_info=True)
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )