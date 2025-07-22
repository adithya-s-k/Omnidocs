#!/usr/bin/env python3
"""
Nougat (Neural Optical Understanding for Academic Documents) LaTeX Expression Extractor

This module provides LaTeX expression extraction using Facebook's Nougat model
via Hugging Face transformers.
"""

import hashlib
import io
import torch
from PIL import Image
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import re
import os
import sys
import logging

# Set up model directory for HuggingFace downloads
def _setup_hf_model_dir():
    """Set up the model directory for HuggingFace to use omnidocs/models."""
    current_file = Path(__file__)
    omnidocs_root = current_file.parent.parent.parent.parent  # Go up to omnidocs root
    models_dir = omnidocs_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variables BEFORE any imports
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    
    return models_dir

_MODELS_DIR = _setup_hf_model_dir()

# Import omnidocs modules
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

# Configuration - Using Hugging Face models
NOUGAT_CHECKPOINTS = {
    "base": {
        "hf_model": "facebook/nougat-base",
        "extract_dir": "nougat_ckpt"
    },
    "small": {
        "hf_model": "facebook/nougat-small",
        "extract_dir": "nougat_small_ckpt"
    }
}

class NougatMapper(BaseLatexMapper):
    """Label mapper for Nougat model output."""

    def _setup_mapping(self):
        # Nougat outputs markdown/LaTeX, minimal mapping needed
        mapping = {
            r"\\": r"\\",    # Keep LaTeX backslashes
            r"\n": " ",      # Remove newlines for single expressions
            r"  ": " ",      # Remove double spaces
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class NougatExtractor(BaseLatexExtractor):
    """Nougat (Neural Optical Understanding for Academic Documents) based expression extraction."""

    def __init__(
        self,
        model_type: str = "small",
        device: Optional[str] = None,
        show_log: bool = False,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize Nougat Extractor."""
        super().__init__(device=device, show_log=show_log)

        self._label_mapper = NougatMapper()
        self.model_type = model_type
        
        # Set default model path if not provided
        if model_path is None:
            model_path = _MODELS_DIR / f"nougat_{model_type}"
        self.model_path = Path(model_path)

        # Check dependencies
        self._check_dependencies()

        try:
            # Check if model exists locally, download if needed
            if not self._model_exists():
                if self.show_log:
                    logger.info("Model not found locally, will download from Hugging Face")
                self._download_model()
            else:
                if self.show_log:
                    logger.info("Model found locally, using that version")
            
            self._load_model()
            if self.show_log:
                logger.success("Nougat model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Nougat model", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            from PIL import Image
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install transformers torch torchvision"
            ) from e

    def _model_exists(self) -> bool:
        """Check if the model exists in the local cache."""
        try:
            from transformers.utils import cached_file
            from huggingface_hub import try_to_load_from_cache
            
            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]
            
            # Check if model files exist in HF cache
            try:
                # Check for key model files
                config_path = try_to_load_from_cache(hf_model_name, "config.json")
                model_path = try_to_load_from_cache(hf_model_name, "pytorch_model.bin")
                processor_path = try_to_load_from_cache(hf_model_name, "tokenizer.json")
                
                if config_path and model_path and processor_path:
                    return True
                    
            except Exception:
                pass
            
            # Fallback: check if model directory exists
            model_cache_dir = _MODELS_DIR / "models--facebook--" / hf_model_name.replace("/", "--")
            return model_cache_dir.exists()
            
        except Exception as e:
            if self.show_log:
                logger.warning(f"Could not check model existence: {e}")
            return False

    def _download_model(self) -> Path:
        """Download model if it doesn't exist locally."""
        try:
            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]
            
            if self.show_log:
                logger.info(f"Downloading Nougat model: {hf_model_name}")
                logger.info(f"Download location: {_MODELS_DIR}")
            
            # Import here to trigger download
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            
            # This will download the model if not cached
            NougatProcessor.from_pretrained(hf_model_name, cache_dir=_MODELS_DIR)
            VisionEncoderDecoderModel.from_pretrained(hf_model_name, cache_dir=_MODELS_DIR)
            
            if self.show_log:
                logger.success("Model download completed")
                
            return self.model_path
            
        except Exception as e:
            logger.error("Failed to download model", exc_info=True)
            raise

    def _load_model(self) -> None:
        """Load Nougat model and processor."""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel

            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Get model name from checkpoint config
            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]

            # Try to load from local path first, fallback to HuggingFace
            try:
                if self.model_path.exists():
                    if self.show_log:
                        logger.info(f"Loading Nougat model from local path: {self.model_path}")
                    
                    self.processor = NougatProcessor.from_pretrained(str(self.model_path))
                    self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
                else:
                    raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")
                    
            except (FileNotFoundError, OSError, Exception) as e:
                if self.show_log:
                    logger.info(f"Could not load from local path ({e}), falling back to HuggingFace: {hf_model_name}")
                
                # Fallback to HuggingFace
                self.processor = NougatProcessor.from_pretrained(
                    hf_model_name, 
                    cache_dir=_MODELS_DIR
                )
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    hf_model_name,
                    cache_dir=_MODELS_DIR
                )
            
            self.model.to(self.device)
            self.model.eval()

            if self.show_log:
                logger.info(f"Loaded Nougat model on {self.device}")

        except Exception as e:
            logger.error("Error loading Nougat model", exc_info=True)
            raise

    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from Nougat's markdown output."""
        import re

        expressions = []

        # Find inline math expressions (between $ ... $)
        inline_math = re.findall(r'\$([^$]+)\$', text)
        expressions.extend(inline_math)

        # Find display math expressions (between $$ ... $$)
        display_math = re.findall(r'\$\$([^$]+)\$\$', text)
        expressions.extend(display_math)

        # Find LaTeX environments
        latex_envs = re.findall(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', text, re.DOTALL)
        for env_name, content in latex_envs:
            if env_name in ['equation', 'align', 'gather', 'multline', 'eqnarray']:
                expressions.append(content.strip())

        # If no specific math found, return the whole text (might contain math)
        if not expressions:
            expressions = [text.strip()]

        return expressions

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

            all_expressions = []
            for img in images:
                # Add padding to make it look more like a document page
                from PIL import ImageOps
                padded_image = ImageOps.expand(img, border=100, fill='white')

                # Process image with Nougat processor
                pixel_values = self.processor(padded_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                # Generate text using the model
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        max_length=512,
                        num_beams=1,  # Use greedy decoding for faster inference
                        do_sample=False,
                        early_stopping=False
                    )

                # Decode the generated text
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

                # Extract mathematical expressions from the text
                expressions = self._extract_math_expressions(generated_text)

                # Map expressions to standard format
                mapped_expressions = [self.map_expression(expr) for expr in expressions]
                all_expressions.extend(mapped_expressions)

            return LatexOutput(
                expressions=all_expressions,
                source_img_size=images[0].size if images else None
            )

        except Exception as e:
            logger.error("Error during Nougat extraction", exc_info=True)
            raise