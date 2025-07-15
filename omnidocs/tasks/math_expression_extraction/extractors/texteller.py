import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

class TextellerMapper(BaseLatexMapper):
    """Label mapper for Texteller model output."""
    
    def _setup_mapping(self):
        # Add any necessary mappings between Texteller's LaTeX format and standard format
        mapping = {
            r"\begin{align}": "",  # Remove align environment
            r"\end{align}": "",
            r"\begin{equation}": "",  # Remove equation environment
            r"\end{equation}": "",
            r"\displaystyle": "",  # Remove display style commands
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class TextellerExtractor(BaseLatexExtractor):
    """Texteller based expression extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_name: str = "OleehyO/latex-ocr",
        **kwargs
    ):
        """Initialize Texteller Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self._label_mapper = TextellerMapper()
        self.model_name = model_name
        
        try:
            from texteller import OCRModel
        except ImportError as e:
            logger.error("Failed to import texteller")
            raise ImportError(
                "texteller is not available. Please install it with: pip install texteller"
            ) from e
            
        try:
            self.model = OCRModel.from_pretrained(
                self.model_name,
                device=self.device if self.device else "auto"
            )
            if self.show_log:
                logger.success("Model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize model", exc_info=True)
            raise
    
    def _download_model(self) -> Path:
        """Model download handled by texteller library."""
        logger.info("Model downloading handled by texteller library")
        return None
    
    def _load_model(self) -> None:
        """Model loaded in __init__."""
        pass
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Texteller."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Run inference
                latex_expr = self.model.predict(img)
                
                # Map to standard format
                mapped_expr = self.map_expression(latex_expr)
                expressions.append(mapped_expr)
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during extraction", exc_info=True)
            raise