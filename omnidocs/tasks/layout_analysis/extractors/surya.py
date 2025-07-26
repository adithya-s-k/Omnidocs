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
            # Import required libraries
            from surya.detection import batch_text_detection
            from surya.layout import batch_layout_detection
            from surya.model.detection.model import (
                load_model as load_detection_model,
                load_processor as load_detection_processor
            )
        except ImportError as ex:
            if self.show_log:
                logger.error("Failed to import surya")
            raise ImportError(
                "surya is not available. Please install it with: pip install surya-ocr"
            ) from ex
            
        try:
            # Initialize detection and layout models
            self.det_model = load_detection_model()
            self.det_processor = load_detection_processor()
            self.layout_model = load_detection_model(checkpoint="vikp/surya_layout3")
            self.layout_processor = load_detection_processor(checkpoint="vikp/surya_layout3")
            
            # Move models to specified device
            if self.device == "cuda":
                self.det_model = self.det_model.cuda()
                self.layout_model = self.layout_model.cuda()
                
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
            # Import here to avoid circular imports
            from surya.detection import batch_text_detection
            from surya.layout import batch_layout_detection
            from surya.postprocessing.heatmap import draw_polys_on_image
            
            # Load and preprocess input
            if isinstance(input_path, (str, Path)):
                image = Image.open(input_path).convert("RGB")
            elif isinstance(input_path, Image.Image):
                image = input_path.convert("RGB")
            elif isinstance(input_path, np.ndarray):
                image = Image.fromarray(input_path).convert("RGB")
            else:
                raise ValueError("Unsupported input type")
            
            # Run text detection first
            line_predictions = batch_text_detection(
                [image], 
                self.det_model, 
                self.det_processor
            )
            
            # Run layout detection
            layout_predictions = batch_layout_detection(
                [image], 
                self.layout_model,
                self.layout_processor, 
                line_predictions
            )
            
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