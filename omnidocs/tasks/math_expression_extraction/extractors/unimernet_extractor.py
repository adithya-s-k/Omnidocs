import os
import sys
import logging
import argparse
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image

# Import omnidocs modules
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

class UniMERNetMapper(BaseLatexMapper):
    """Label mapper for UniMERNet model output."""
    
    def _setup_mapping(self):
        # Add any necessary mappings between UniMERNet's LaTeX format and standard format
        mapping = {
            # Add specific mappings if needed for UniMERNet output
            r"\begin{aligned}": "",  # Remove aligned environment if needed
            r"\end{aligned}": "",
            # Add more mappings as needed based on UniMERNet's output format
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class UniMERNetExtractor(BaseLatexExtractor):
    """UniMERNet based expression extraction implementation."""
    
    def __init__(
        self,
        model_path: str,
        cfg_path: str = "configs/unimernet.yaml",
        batch_size: int = 1,
        device: Optional[str] = None,
        show_log: bool = False,
        **kwargs
    ):
        """Initialize UniMERNet Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.batch_size = batch_size
        self._label_mapper = UniMERNetMapper()
        
        # Import UniMERNet modules
        try:
            import unimernet.tasks as tasks
            from unimernet.common.config import Config
            from unimernet.processors import load_processor
            self.tasks = tasks
            self.Config = Config
            self.load_processor = load_processor
        except ImportError as e:
            logger.error("Failed to import UniMERNet modules")
            raise ImportError(
                "UniMERNet modules are not available. Please install UniMERNet."
            ) from e
        
        # Load model and processor
        try:
            self.model, self.vis_processor = self._load_model_and_processor()
            if self.show_log:
                logger.success("UniMERNet model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize UniMERNet model", exc_info=True)
            raise
    
    def _download_model(self) -> Path:
        """Model download handled by UniMERNet or manually."""
        logger.info("Ensure UniMERNet model is available at the specified path")
        return Path(self.model_path)
    
    def _load_model(self) -> None:
        """Model loaded in __init__."""
        pass
    
    def _load_model_and_processor(self):
        """Load UniMERNet model and processor."""
        try:
            # Create args namespace for config
            args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
            cfg = self.Config(args)
            
            # Set model paths
            cfg.config.model.pretrained = os.path.join(self.model_path, "pytorch_model.pth")
            cfg.config.model.model_config.model_name = self.model_path
            cfg.config.model.tokenizer_config.path = self.model_path
            
            # Setup task and build model
            task = self.tasks.setup_task(cfg)
            model = task.build_model(cfg).to(self.device)
            
            # Load visual processor
            vis_processor = self.load_processor(
                'formula_image_eval', 
                cfg.config.datasets.formula_rec_eval.vis_processor.eval
            )
            
            return model, vis_processor
            
        except Exception as e:
            logger.error(f"Error loading UniMERNet model and processor: {e}")
            raise
    
    def _process_single_image(self, image: Image.Image) -> str:
        """Process a single image and return LaTeX expression."""
        try:
            # Process the image using the visual processor
            processed_image = self.vis_processor(image).unsqueeze(0).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                output = self.model.generate({"image": processed_image})
                pred = output["pred_str"][0]
            
            return pred
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using UniMERNet."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Run inference
                latex_expr = self._process_single_image(img)
                
                # Map to standard format
                mapped_expr = self.map_expression(latex_expr)
                expressions.append(mapped_expr)
                
                if self.show_log:
                    logger.info(f"Extracted expression: {mapped_expr}")
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during extraction", exc_info=True)
            raise

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        **kwargs
    ) -> List[LatexOutput]:
        """Extract LaTeX expressions from multiple images."""
        results = []
        
        for img_path in images:
            try:
                result = self.extract(img_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                # Return empty result for failed images
                results.append(LatexOutput(expressions=[], source_img_size=None))
        
        return results


# Configuration class for easy setup
class UniMERNetConfig:
    """Configuration class for UniMERNet extractor."""
    
    def __init__(
        self,
        model_path: str,
        cfg_path: str = "configs/unimernet.yaml",
        batch_size: int = 1,
        device: Optional[str] = None,
        show_log: bool = False
    ):
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.show_log = show_log
    
    def create_extractor(self) -> UniMERNetExtractor:
        """Create and return a UniMERNet extractor instance."""
        return UniMERNetExtractor(
            model_path=self.model_path,
            cfg_path=self.cfg_path,
            batch_size=self.batch_size,
            device=self.device,
            show_log=self.show_log
        )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = UniMERNetConfig(
        model_path="./models/unimernet_tiny",
        cfg_path="configs/unimernet.yaml",
        batch_size=1,
        device="cuda",
        show_log=True
    )
    
    # Create extractor
    extractor = config.create_extractor()
    
    # Example usage
    try:
        # Single image extraction
        result = extractor.extract("path/to/formula/image.png")
        print(f"Extracted expressions: {result.expressions}")
        
        # Batch extraction
        image_paths = ["image1.png", "image2.png", "image3.png"]
        batch_results = extractor.extract_batch(image_paths)
        
        for i, result in enumerate(batch_results):
            print(f"Image {i+1} expressions: {result.expressions}")
            
    except Exception as e:
        logger.error(f"Error in example usage: {e}")