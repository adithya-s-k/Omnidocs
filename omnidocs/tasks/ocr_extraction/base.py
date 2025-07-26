from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
from pydantic import BaseModel, Field
from omnidocs.utils.logging import get_logger
from omnidocs.utils.language import global_language_mapper, detect_language

logger = get_logger(__name__)

class OCRText(BaseModel):
    """
    Container for individual OCR text detection.
    
    Attributes:
        text: Extracted text content
        confidence: Confidence score for the text detection
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        polygon: Optional polygon coordinates for irregular text regions
        language: Detected language code (e.g., 'en', 'zh', 'fr')
        reading_order: Optional reading order index for text sequencing
    """
    text: str = Field(..., description="Extracted text content")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    polygon: Optional[List[List[float]]] = Field(None, description="Polygon coordinates [[x1,y1], [x2,y2], ...]")
    language: Optional[str] = Field(None, description="Detected language code")
    reading_order: Optional[int] = Field(None, description="Reading order index")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'polygon': self.polygon,
            'language': self.language,
            'reading_order': self.reading_order
        }

class OCROutput(BaseModel):
    """
    Container for OCR extraction results.
    
    Attributes:
        texts: List of detected text objects
        full_text: Combined text from all detections
        source_img_size: Original image dimensions (width, height)
        processing_time: Time taken for OCR processing
        metadata: Additional metadata from the OCR engine
    """
    texts: List[OCRText] = Field(..., description="List of detected text objects")
    full_text: str = Field(..., description="Combined text from all detections")
    source_img_size: Optional[Tuple[int, int]] = Field(None, description="Original image dimensions (width, height)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional OCR engine metadata")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'texts': [text.to_dict() for text in self.texts],
            'full_text': self.full_text,
            'source_img_size': self.source_img_size,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save output to JSON file."""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_text_by_confidence(self, min_confidence: float = 0.5) -> List[OCRText]:
        """Filter texts by minimum confidence threshold."""
        return [text for text in self.texts if text.confidence is None or text.confidence >= min_confidence]
    
    def get_sorted_by_reading_order(self) -> List[OCRText]:
        """Get texts sorted by reading order (top-to-bottom, left-to-right if no reading_order)."""
        texts_with_order = [text for text in self.texts if text.reading_order is not None]
        texts_without_order = [text for text in self.texts if text.reading_order is None]
        
        # Sort texts with reading order
        texts_with_order.sort(key=lambda x: x.reading_order)
        
        # Sort texts without reading order by bbox (top-to-bottom, left-to-right)
        if texts_without_order:
            texts_without_order.sort(key=lambda x: (x.bbox[1], x.bbox[0]) if x.bbox else (0, 0))
        
        return texts_with_order + texts_without_order

class BaseOCRMapper:
    """Base class for mapping OCR engine-specific outputs to standardized format."""
    
    def __init__(self, engine_name: str):
        """Initialize mapper for specific OCR engine.
        
        Args:
            engine_name: Name of the OCR engine (e.g., 'tesseract', 'paddle', 'easyocr')
        """
        self.engine_name = engine_name.lower()
        self._global_mapper = global_language_mapper
    
    def to_standard_language(self, engine_language: str) -> str:
        """Convert engine-specific language code to standard ISO 639-1."""
        return self._global_mapper.to_standard(self.engine_name, engine_language)
    
    def from_standard_language(self, standard_language: str) -> str:
        """Convert standard ISO 639-1 language code to engine-specific format."""
        return self._global_mapper.from_standard(self.engine_name, standard_language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for this engine."""
        return self._global_mapper.get_supported_languages(self.engine_name)
    
    def normalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Normalize bounding box coordinates to absolute pixel values."""
        # If bbox values are between 0-1, they're normalized and need to be scaled
        if all(0 <= coord <= 1 for coord in bbox):
            return [
                bbox[0] * img_width,
                bbox[1] * img_height,
                bbox[2] * img_width,
                bbox[3] * img_height
            ]
        return bbox
    
    def detect_text_language(self, text: str) -> Optional[str]:
        """Detect language of extracted text."""
        return detect_language(text)

class BaseOCRExtractor(ABC):
    """Base class for OCR text extraction models."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 languages: Optional[List[str]] = None,
                 engine_name: Optional[str] = None):
        """Initialize the OCR extractor.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            show_log: Whether to show detailed logs
            languages: List of language codes to support (e.g., ['en', 'zh'])
            engine_name: Name of the OCR engine for language mapping
        """
        self.show_log = show_log
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.languages = languages or ['en']
        self.engine_name = engine_name or self.__class__.__name__.lower().replace('extractor', '')
        self.model = None
        self.model_path = None
        self._label_mapper: Optional[BaseOCRMapper] = None
        
        # Initialize mapper if engine name is provided
        if self.engine_name:
            self._label_mapper = BaseOCRMapper(self.engine_name)
        
        if self.show_log:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Engine: {self.engine_name}")
            logger.info(f"Supported languages: {self.languages}")
    
    @abstractmethod
    def _download_model(self) -> Path:
        """Download model from remote source."""
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load model into memory."""
        pass
    
    def preprocess_input(self, input_path: Union[str, Path, Image.Image, np.ndarray]) -> List[Image.Image]:
        """Convert input to list of PIL Images.
        
        Args:
            input_path: Input image path or image data
            
        Returns:
            List of PIL Images
        """
        if isinstance(input_path, (str, Path)):
            image = Image.open(input_path).convert('RGB')
            return [image]
        elif isinstance(input_path, Image.Image):
            return [input_path.convert('RGB')]
        elif isinstance(input_path, np.ndarray):
            return [Image.fromarray(cv2.cvtColor(input_path, cv2.COLOR_BGR2RGB))]
        else:
            raise ValueError(f"Unsupported input type: {type(input_path)}")
    
    def postprocess_output(self, raw_output: Any, img_size: Tuple[int, int]) -> OCROutput:
        """Convert raw OCR output to standardized OCROutput format.
        
        Args:
            raw_output: Raw output from OCR engine
            img_size: Original image size (width, height)
            
        Returns:
            Standardized OCROutput object
        """
        # This should be implemented by child classes based on their specific output format
        raise NotImplementedError("Child classes must implement postprocess_output method")
    
    @abstractmethod
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> OCROutput:
        """Extract text from input image.
        
        Args:
            input_path: Path to input image or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            OCROutput containing extracted text
        """
        pass
    
    def extract_all(
        self,
        input_paths: List[Union[str, Path, Image.Image]],
        **kwargs
    ) -> List[OCROutput]:
        """Extract text from multiple images.
        
        Args:
            input_paths: List of image paths or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of OCROutput objects
        """
        results = []
        for input_path in input_paths:
            try:
                result = self.extract(input_path, **kwargs)
                results.append(result)
            except Exception as e:
                if self.show_log:
                    logger.error(f"Error processing {input_path}: {str(e)}")
                raise
        return results
    
    def extract_with_layout(
        self,
        input_path: Union[str, Path, Image.Image],
        layout_regions: Optional[List[Dict]] = None,
        **kwargs
    ) -> OCROutput:
        """Extract text with optional layout information.
        
        Args:
            input_path: Path to input image or image data
            layout_regions: Optional list of layout regions to focus OCR on
            **kwargs: Additional model-specific parameters
            
        Returns:
            OCROutput containing extracted text
        """
        # Default implementation just calls extract, can be overridden by child classes
        return self.extract(input_path, **kwargs)
    
    def map_language(self, language: str) -> str:
        """Map engine-specific language to standardized format."""
        if self._label_mapper is None:
            return language
        return self._label_mapper.to_standard_language(language)
    
    def detect_text_language(self, text: str) -> Optional[str]:
        """Detect language of extracted text using script analysis."""
        if self._label_mapper is None:
            return detect_language(text)
        return self._label_mapper.detect_text_language(text)
    
    @property
    def label_mapper(self) -> BaseOCRMapper:
        """Get the label mapper for this extractor."""
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper
    
    def set_languages(self, languages: List[str]) -> None:
        """Update supported languages for OCR extraction."""
        self.languages = languages
        if self.show_log:
            logger.info(f"Updated supported languages: {self.languages}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.languages.copy()

    def visualize(self,
                  ocr_result: 'OCROutput',
                  image_path: Union[str, Path, Image.Image],
                  output_path: str = "visualized.png",
                  box_color: str = 'red',
                  box_width: int = 2,
                  show_text: bool = False,
                  text_color : str = 'blue',
                  font_size : int = 12) -> None:
        """Visualize OCR results by drawing bounding boxes on the original image.

        This method allows users to easily see which extractor is working better
        by visualizing the detected text regions with bounding boxes.
        """
        try:
            # Handle different input types
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, Image.Image):
                image = image_path.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path)}")
            
            # Create a copy to draw on
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font for text overlay
            font = None
            if show_text:
                try:
                    # Try to use a better font if available
                    font = ImageFont.truetype("arial.ttf", font_size)
                except (OSError, IOError):
                    try:
                        # Fallback to default font
                        font = ImageFont.load_default()
                    except:
                        font = None
            
            # Draw bounding boxes and text if OCR results exist
            if hasattr(ocr_result, "texts") and ocr_result.texts:
                for item in ocr_result.texts:
                    bbox = getattr(item, "bbox", None)
                    text = getattr(item, "text", "")
                    
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        
                        # Draw rectangle around text
                        draw.rectangle(
                            [(x1, y1), (x2, y2)], 
                            outline=box_color, 
                            width=box_width
                        )
            # Save the annotated image
            annotated_image.save(output_path)
            
            if self.show_log:
                logger.info(f"OCR visualization saved to {output_path}")
                logger.info(f"Visualized {len(ocr_result.texts) if ocr_result.texts else 0} text detections")
                
        except Exception as e:
            error_msg = f"Error creating OCR visualization: {str(e)}"
            if self.show_log:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def visualize_from_json(self, 
                           image_path: Union[str, Path, Image.Image],
                           json_path: Union[str, Path],
                           output_path: str = "visualized_from_json.png",
                           **kwargs) -> None:
        """
        Load OCR results from JSON file and visualize them.
        
        Args:
            image_path: Path to original image or PIL Image object
            json_path: Path to JSON file containing OCR results
            output_path: Path to save the annotated image
            **kwargs: Additional arguments passed to visualize method
        """
        import json
        
        try:
            # Load OCR results from JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to OCROutput format
            texts = []
            for text_data in data.get('texts', []):
                texts.append(OCRText(**text_data))
            
            ocr_result = OCROutput(
                texts=texts,
                full_text=data.get('full_text', ''),
                source_img_size=data.get('source_img_size'),
                processing_time=data.get('processing_time'),
                metadata=data.get('metadata')
            )
            
            # Visualize the loaded results
            self.visualize(ocr_result, image_path, output_path, **kwargs)
            
        except Exception as e:
            error_msg = f"Error loading and visualizing from JSON: {str(e)}"
            if self.show_log:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
