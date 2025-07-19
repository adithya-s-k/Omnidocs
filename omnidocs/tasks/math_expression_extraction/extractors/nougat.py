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

# ===================== Model API =====================
class Nougat:
    """Main Nougat API for document understanding"""
    def __init__(
        self, 
        model_dir="omnidocs/models", 
        model_type="base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.device = device
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download and set up the model
        self.setup_model()
        
    def setup_model(self):
        """Download and set up the model and tokenizer using Hugging Face"""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel

            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]

            logger.info(f"Loading Nougat model from Hugging Face: {hf_model_name}")

            # Load processor and model from Hugging Face
            self.processor = NougatProcessor.from_pretrained(hf_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model from Hugging Face: {e}")
            raise
    
    @torch.no_grad()
    @log_execution_time
    def generate(
        self,
        image_path,
        max_length=512,
        num_beams=4,
        temperature=1.0,
        no_repeat_ngram_size=3
    ):
        """Generate text from document image using Hugging Face model"""
        try:
            # Load and preprocess image
            from PIL import Image, ImageOps
            image = Image.open(image_path).convert('RGB')

            # Add padding to make it look more like a document page (helps with math recognition)
            padded_image = ImageOps.expand(image, border=100, fill='white')

            # Process image with Hugging Face processor
            pixel_values = self.processor(padded_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text using the model with optimized parameters for math
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=max(1, num_beams),  # Ensure at least 1 beam
                do_sample=False,
                early_stopping=False if num_beams == 1 else True  # Only use early_stopping with beam search
            )

            # Decode the generated text
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            return generated_text

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise
        
    def __call__(self, image_path, **kwargs):
        """Convenience method to process an image"""
        return self.generate(image_path, **kwargs)


# ===================== Main Application =====================
@log_execution_time
def process_document(image_path, output_path=None, model_type="small"):
    """Process a document image and generate text"""
    logger.info(f"Processing document: {image_path}")
    
    # Initialize Nougat model
    nougat = Nougat(model_type=model_type)
    
    # Generate text from document
    generated_text = nougat(image_path)
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        logger.info(f"Output saved to: {output_path}")
    
    return generated_text


def process_pdf(pdf_path, output_dir=None, model_type="base"):
    """Process a PDF file and extract text from each page"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
        return None
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Initialize Nougat model
    nougat = Nougat(model_type=model_type)
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    document = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(document)):
        logger.info(f"Processing page {page_num+1}/{len(document)}")
        
        # Get page as image
        page = document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Process image
        text = nougat(img)
        results.append(text)
        
        # Save individual page if output_dir is provided
        if output_dir:
            output_path = os.path.join(output_dir, f"page_{page_num+1:03d}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
    
    # Combine all results
    combined_text = "\n\n".join(results)
    
    # Save combined text if output_dir is provided
    if output_dir:
        combined_output_path = os.path.join(output_dir, "combined_output.txt")
        with open(combined_output_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
    
    return combined_text


# ===================== BaseLatexExtractor Implementation =====================
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
        **kwargs
    ):
        """Initialize Nougat Extractor."""
        super().__init__(device=device, show_log=show_log)

        self._label_mapper = NougatMapper()
        self.model_type = model_type

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
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            from PIL import Image
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

            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Get model name from checkpoint config
            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]

            logger.info(f"Loading Nougat model from Hugging Face: {hf_model_name}")
            logger.info(f"Models will be downloaded in: {_MODELS_DIR}")

            # Load processor and model
            self.processor = NougatProcessor.from_pretrained(hf_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
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
