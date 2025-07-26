from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import torch
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field
from omnidocs.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

# TODO : add a .visualize() method to LatexOutput for visualization into an image

class LatexOutput(BaseModel):
    """
    Container for extracted LaTeX expressions.
    
    Attributes:
        expressions: List of extracted LaTeX expressions
        confidences: Optional confidence scores for each expression
        bboxes: Optional bounding boxes for each expression
        source_img_size: Optional tuple of source image dimensions
    """
    expressions: List[str] = Field(..., description="List of extracted LaTeX expressions")
    confidences: Optional[List[float]] = Field(None, description="Confidence scores for each expression")
    bboxes: Optional[List[List[float]]] = Field(None, description="Bounding boxes for expressions [x1, y1, x2, y2]")
    source_img_size: Optional[Tuple[int, int]] = Field(None, description="Original image dimensions (width, height)")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'expressions': self.expressions,
            'confidences': self.confidences,
            'bboxes': self.bboxes,
            'source_img_size': self.source_img_size
        }
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save output to JSON file."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class BaseLatexMapper:
    """Base class for mapping model-specific outputs to standardized format."""
    
    def __init__(self):
        self._mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup the mapping dictionary. Should be implemented by child classes."""
        raise NotImplementedError
    
    def to_standard(self, model_output: str) -> str:
        """Convert model-specific LaTeX to standardized format."""
        return self._mapping.get(model_output, model_output)
    
    def from_standard(self, standard_latex: str) -> str:
        """Convert standardized LaTeX to model-specific format."""
        return self._reverse_mapping.get(standard_latex, standard_latex)

class BaseLatexExtractor(ABC):
    """Base class for LaTeX expression extraction models."""
    
    def __init__(self, device: Optional[str] = None, show_log: bool = False):
        """Initialize the LaTeX extractor.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            show_log: Whether to show detailed logs
        """
        self.show_log = show_log
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = None
        self._label_mapper: Optional[BaseLatexMapper] = None
        
        if self.show_log:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.info(f"Using device: {self.device}")
    
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
    
    @abstractmethod
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions from input image.
        
        Args:
            input_path: Path to input image or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            LatexOutput containing extracted expressions
        """
        pass
    
    def extract_all(
        self,
        input_paths: List[Union[str, Path, Image.Image]],
        **kwargs
    ) -> List[LatexOutput]:
        """Extract LaTeX from multiple images.
        
        Args:
            input_paths: List of image paths or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of LatexOutput objects
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
    
    def map_expression(self, expression: str) -> str:
        """Map model-specific LaTeX to standardized format."""
        if self._label_mapper is None:
            return expression
        return self._label_mapper.to_standard(expression)
    
    @property
    def label_mapper(self) -> BaseLatexMapper:
        """Get the label mapper for this extractor."""
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper