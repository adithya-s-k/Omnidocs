# IMPORTANT: Set up model directory BEFORE any HuggingFace imports
import os
from pathlib import Path

# Set up model directory for texteller downloads
def _setup_texteller_model_dir():
    """Set up the model directory for texteller to use omnidocs/models."""
    # Get the omnidocs project root
    current_file = Path(__file__)
    omnidocs_root = current_file.parent.parent.parent.parent  # Go up to omnidocs root
    models_dir = omnidocs_root / "models"

    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)

    # Set environment variables for HuggingFace cache BEFORE any imports
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)

    return models_dir

# Set up model directory before any imports
_MODELS_DIR = _setup_texteller_model_dir()

# Now import everything else
import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
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
            from texteller import load_model, load_tokenizer, img2latex
            self._img2latex = img2latex
        except ImportError as e:
            logger.error("Failed to import texteller")
            raise ImportError(
                "texteller is not available. Please install it with: pip install texteller"
            ) from e

        try:
            # Load model and tokenizer using new API with our model directory
            if self.show_log:
                logger.info(f"Loading texteller models to: {_MODELS_DIR}")

            self.model = load_model(model_dir=None, use_onnx=False)  # Use PyTorch model
            self.tokenizer = load_tokenizer()

            # Set device
            if self.device and hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)

            if self.show_log:
                logger.success(f"Model and tokenizer initialized successfully in: {_MODELS_DIR}")
        except Exception as e:
            logger.error("Failed to initialize model", exc_info=True)
            raise
    
    def _download_model(self) -> Path:
        """Model download handled by texteller library to omnidocs/models directory."""
        logger.info(f"Model downloading handled by texteller library to: {_MODELS_DIR}")
        return _MODELS_DIR
    
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

            # Convert PIL images to numpy arrays for texteller
            import numpy as np
            image_arrays = []
            for img in images:
                if hasattr(img, 'convert'):  # PIL Image
                    img_array = np.array(img.convert('RGB'))
                    image_arrays.append(img_array)
                else:
                    image_arrays.append(img)

            # Run inference using new API
            device = getattr(self.model, 'device', None)
            latex_results = self._img2latex(
                model=self.model,
                tokenizer=self.tokenizer,
                images=image_arrays,
                device=device,
                out_format='latex',
                max_tokens=1024
            )

            # Process results
            for latex_expr in latex_results:
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