import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
import torch
import cv2
import numpy as np
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput
from pathlib import Path

logger = get_logger(__name__)

class NougatMapper(BaseLatexMapper):
    """Label mapper for Nougat model output."""
    
    def _setup_mapping(self):
        # Nougat outputs markdown-style math, convert to LaTeX
        mapping = {
            r"$$": r"$",  # Convert display math to inline
            r"\n": " ",   # Remove newlines
            r"  ": " ",   # Remove double spaces
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class NougatExtractor(BaseLatexExtractor):
    """Nougat (Meta) based expression extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_name: str = "facebook/nougat-base",
        **kwargs
    ):
        """Initialize Nougat Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self._label_mapper = NougatMapper()
        self.model_name = model_name
        
        # Check dependencies
        self._check_dependencies()
        
        try:
            self._load_model()
            if self.show_log:
                logger.success("Nougat model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Nougat model", exc_info=True)
            raise
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import transformers
            import torch
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install transformers torch torchvision"
            ) from e
    
    def _download_model(self) -> Path:
        """Model download handled by transformers library."""
        logger.info("Model downloading handled by transformers library")
        return None
    
    def _load_model(self) -> None:
        """Load Nougat model and processor."""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            import torch
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load processor and model
            self.processor = NougatProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.show_log:
                logger.info(f"Loaded Nougat model on {self.device}")
                
        except Exception as e:
            logger.error("Error loading Nougat model", exc_info=True)
            raise
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Nougat."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Prepare image for Nougat
                pixel_values = self.processor(img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=5,
                    )
                
                # Decode output
                sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Map to standard format
                mapped_expr = self.map_expression(sequence)
                expressions.append(mapped_expr)
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during Nougat extraction", exc_info=True)
            raise