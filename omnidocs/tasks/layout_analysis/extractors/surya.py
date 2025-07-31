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

       
class SuryaLayoutMapper(BaseLayoutMapper):
    """Label mapper for Surya layout detection model."""
    
    def _setup_mapping(self):
        mapping = {
            "caption": LayoutLabel.CAPTION,
            "footnote": LayoutLabel.TEXT,  # Map footnote to text since no direct equivalent
            "formula": LayoutLabel.FORMULA,
            "list-item": LayoutLabel.LIST,
            "page-footer": LayoutLabel.TEXT,  # Map page-footer to text
            "page-header": LayoutLabel.TEXT,  # Map page-header to text
            "picture": LayoutLabel.IMAGE,
            "figure": LayoutLabel.IMAGE,  # Map figure to image
            "section-header": LayoutLabel.TITLE,  # Map section-header to title
            "table": LayoutLabel.TABLE,
            "text": LayoutLabel.TEXT,
            "title": LayoutLabel.TITLE
        }
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class SuryaLayoutDetector(BaseLayoutDetector):
    """Surya-based layout detection implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        **kwargs
    ):
        """Initialize Surya Layout Detector."""
        super().__init__(show_log=show_log)
        
        # Initialize label mapper
        self._label_mapper = SuryaLayoutMapper()
        
        if self.show_log:
            logger.info("Initializing SuryaLayoutDetector")
        
        # Set device if specified, otherwise use default from parent
        if device:
            self.device = device
            
        if self.show_log:
            logger.info(f"Using device: {self.device}")
            
        try:
            # Import required libraries - use new API
            import surya
            if self.show_log:
                logger.info(f"Found surya package at: {surya.__file__}")
        except ImportError as ex:
            if self.show_log:
                logger.error("Failed to import surya")
            raise ImportError(
                "surya is not available. Please install it with: pip install surya-ocr"
            ) from ex
            
        try:
            # Initialize detection and layout models using new API
            from surya.detection import DetectionPredictor
            from surya.layout import LayoutPredictor

            self.det_predictor = DetectionPredictor()
            self.layout_predictor = LayoutPredictor()

            if self.show_log:
                logger.success("Models initialized successfully")

        except Exception as e:
            if self.show_log:
                logger.error("Failed to initialize models", exc_info=True)
            raise

    def _download_model(self) -> Path:
        """
        Download model from remote source.
        Note: Surya handles model downloading internally.
        """
        if self.show_log:
            logger.info("Surya handles model downloading internally")
        return None

    def _load_model(self) -> None:
        """
        Load the model into memory.
        Note: Models are loaded in __init__.
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
            if isinstance(input_path, (str, Path)):
                image = Image.open(input_path).convert("RGB")
            elif isinstance(input_path, Image.Image):
                image = input_path.convert("RGB")
            elif isinstance(input_path, np.ndarray):
                image = Image.fromarray(input_path).convert("RGB")
            else:
                raise ValueError("Unsupported input type")

            # Run layout detection using new API
            layout_predictions = self.layout_predictor([image])

            # Process the layout prediction (take first since we only processed one image)
            layout_pred = layout_predictions[0]
            
            # Convert to standardized format
            layout_boxes = []
            for box in layout_pred.bboxes:
                mapped_label = self.map_label(box.label)
                if mapped_label:
                    layout_boxes.append(
                        LayoutBox(
                            label=mapped_label,
                            bbox=box.bbox,  # Already in [x1, y1, x2, y2] format
                            confidence=box.confidence
                        )
                    )
            
            # Create annotated image
            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Draw boxes with standardized colors
            for box in layout_boxes:
                color = self.color_map.get(box.label, 'gray')
                coords = box.bbox
                draw.rectangle(coords, outline=color, width=3)
                draw.text((coords[0], coords[1]-20), box.label, fill=color)
            
            # Create LayoutOutput with image size
            layout_output = LayoutOutput(
                bboxes=layout_boxes,
                image_size=image.size
            )
            
            return annotated_img, layout_output
            
        except Exception as e:
            if self.show_log:
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