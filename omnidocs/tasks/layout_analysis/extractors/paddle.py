import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple, Sequence
from pathlib import Path
import cv2
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image , ImageDraw
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.layout_analysis.base import BaseLayoutDetector, BaseLayoutMapper
from omnidocs.tasks.layout_analysis.enums import LayoutLabel
from omnidocs.tasks.layout_analysis.models import LayoutBox, LayoutOutput

logger = get_logger(__name__)

class PaddleLayoutMapper(BaseLayoutMapper):
    """Label mapper for PaddleOCR layout detection model."""
    
    def _setup_mapping(self):
        mapping = {
            "text": LayoutLabel.TEXT,
            "title": LayoutLabel.TITLE,
            "list": LayoutLabel.LIST,
            "table": LayoutLabel.TABLE,
            "figure": LayoutLabel.IMAGE
        }
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}


# ================================================================================================================
            
class PaddleLayoutDetector(BaseLayoutDetector):
    """PaddleOCR-based layout detection implementation."""

    def __init__(
        self, 
        device: Optional[str] = None,
        show_log: bool = False,
        **kwargs
    ):
        """Initialize PaddleOCR Layout Detector."""
        super().__init__()
        
        # Initialize label mapper
        self._label_mapper = PaddleLayoutMapper()

        # Log initialization
        logger.info("Initializing PaddleLayoutDetector")

        # Set device if specified
        if device:
            self.device = device
        logger.info(f"Using device: {self.device}")

        try:
            from paddleocr import PPStructure
        except ImportError as ex:
            logger.error("Failed to import paddleocr")
            raise ImportError(
                "paddleocr is not available. Please install it with: pip install paddleocr"
            ) from ex


        # Initialize the model
        try:
            self.model = PPStructure(
                table=True,
                ocr=True,
                show_log=show_log,
                **kwargs
            )
            logger.success("Model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize model", exc_info=True)
            raise

    def _download_model(self) -> Path:
        """
        Download model from remote source.
        Note: PaddleOCR handles model downloading internally.
        """
        logger.info("PaddleOCR handles model downloading internally")
        return None

    def _load_model(self) -> None:
        """
        Load the model into memory.
        Note: Model is loaded in __init__.
        """
        pass

    @log_execution_time
    def detect(
        self, 
        input_path: Union[str, Path], 
        **kwargs
    ) -> Tuple[Image.Image, LayoutOutput]:
        """Run layout detection with standardized labels."""
        try:
            # Load and preprocess input
            images = self.preprocess_input(input_path)
            
            results = []
            for img in images:
                # Get detection results
                det_result = self.model(img)
                
                # Convert to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Create annotated image
                annotated_img = img.copy()
                draw = ImageDraw.Draw(annotated_img)
                
                # Convert detection results to LayoutBox objects with standardized labels
                layout_boxes = []
                
                for block in det_result:
                    # Extract coordinates and type
                    x1, y1, x2, y2 = block['bbox']
                    model_label = block['type']
                    mapped_label = self.map_label(model_label)
                    
                    if mapped_label:  # Only include boxes with valid mapped labels
                        layout_boxes.append(
                            LayoutBox(
                                label=mapped_label,
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                confidence=block.get('confidence', None)
                            )
                        )
                        
                        # Draw with standardized colors
                        color = self.color_map.get(mapped_label, 'gray')
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        draw.text((x1, y1-20), mapped_label, fill=color)
                
                results.append((
                    annotated_img,
                    LayoutOutput(bboxes=layout_boxes)
                ))

            return results[0] if results else (None, LayoutOutput(bboxes=[]))

        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            raise
        
    def visualize(
        self,
        detection_result: Tuple[Image.Image, LayoutOutput],
        output_path: Union[str, Path],
    ) -> None:
        """
        Save annotated image and layout data to files.
        
        Args:
            detection_result: Tuple containing (PIL Image, LayoutOutput)
            output_path: Path to save visualization
        """
        super().visualize(detection_result, output_path)
        