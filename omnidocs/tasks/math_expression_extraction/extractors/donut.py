import sys
import logging
from typing import Union, List, Dict, Any, Optional, 
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
import json
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

class DonutMapper(BaseLatexMapper):
    """Label mapper for Donut model output."""
    
    def _setup_mapping(self):
        # Donut outputs JSON, extract math content
        mapping = {
            r"\n": " ",     # Remove newlines
            r"  ": " ",     # Remove double spaces
            r"\\": r"\\",    # Fix escaped backslashes
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class DonutExtractor(BaseLatexExtractor):
    """Donut (NAVER CLOVA) based expression extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        **kwargs
    ):
        """Initialize Donut Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self._label_mapper = DonutMapper()
        self.model_name = model_name
        
        # Check dependencies
        self._check_dependencies()
        
        try:
            self._load_model()
            if self.show_log:
                logger.success("Donut model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Donut model", exc_info=True)
            raise
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import transformers
            import torch
            import json
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
        """Load Donut model and processor."""
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load processor and model
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.show_log:
                logger.info(f"Loaded Donut model on {self.device}")
                
        except Exception as e:
            logger.error("Error loading Donut model", exc_info=True)
            raise
    
    def _extract_math_from_json(self, json_str: str) -> str:
        """Extract mathematical content from Donut's JSON output."""
        try:
            # Try to parse as JSON
            data = json.loads(json_str)
            
            # Look for math-related fields (adjust based on your use case)
            math_content = ""
            
            if isinstance(data, dict):
                # Common field names that might contain math
                math_fields = ['text', 'content', 'formula', 'equation', 'math']
                for field in math_fields:
                    if field in data:
                        math_content += str(data[field]) + " "
                
                # If no specific fields, concatenate all string values
                if not math_content.strip():
                    for value in data.values():
                        if isinstance(value, str):
                            math_content += value + " "
            
            return math_content.strip()
            
        except json.JSONDecodeError:
            # If not valid JSON, return as is
            return json_str
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Donut."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Prepare image for Donut
                pixel_values = self.processor(img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Prepare task prompt (adjust based on your specific task)
                task_prompt = "<s_cord-v2>"  # Default CORD v2 task
                decoder_input_ids = self.processor.tokenizer(
                    task_prompt, 
                    add_special_tokens=False, 
                    return_tensors="pt"
                ).input_ids
                decoder_input_ids = decoder_input_ids.to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )
                
                # Decode output
                sequence = self.processor.batch_decode(outputs.sequences)[0]
                sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
                sequence = sequence.replace(task_prompt, "")
                
                # Extract math content from JSON-like output
                math_content = self._extract_math_from_json(sequence)
                
                # Map to standard format
                mapped_expr = self.map_expression(math_content)
                expressions.append(mapped_expr)
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during Donut extraction", exc_info=True)
            raise