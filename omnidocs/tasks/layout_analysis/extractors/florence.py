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

        
# ================================================================================================================
class FlorenceLayoutMapper(BaseLayoutMapper):
    """Label mapper for Florence layout detection model."""
    
    def _setup_mapping(self):
        mapping = {
            "cap": LayoutLabel.CAPTION,
            "footnote": LayoutLabel.TEXT,  # Map footnote to text since no direct equivalent
            "math": LayoutLabel.FORMULA,
            "list": LayoutLabel.LIST,
            "bottom": LayoutLabel.TEXT,  # Map page-footer to text
            "header": LayoutLabel.TEXT,  # Map page-header to text
            "picture": LayoutLabel.IMAGE,
            "section": LayoutLabel.TITLE,  # Map section-header to title
            "table": LayoutLabel.TABLE,
            "text": LayoutLabel.TEXT,
            "title": LayoutLabel.TITLE
        }
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class FlorenceLayoutDetector(BaseLayoutDetector):
    """Florence-based layout detection implementation."""
    
    MODEL_REPO = "yifeihu/Florence-2-DocLayNet-Fixed"
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """Initialize Florence Layout Detector."""
        super().__init__(show_log=show_log)
        
        # Initialize label mapper
        self._label_mapper = FlorenceLayoutMapper()
        
        logger.info("Initializing FlorenceLayoutDetector")
        
        if device:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as ex:
            logger.error("Failed to import transformers")
            raise ImportError(
                "transformers is not available. Please install it with: pip install transformers"
            ) from ex
            
        # Initialize the model and processor
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_REPO,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_REPO,
                trust_remote_code=trust_remote_code
            )
            self.model.to(self.device)
            logger.success("Model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize model", exc_info=True)
            raise

    def _download_model(self) -> Path:
        """
        Download model from remote source.
        Note: Handled by transformers library.
        """
        logger.info("Model downloading handled by transformers library")
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
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        num_beams: int = 3,
        **kwargs
    ) -> Tuple[Image.Image, LayoutOutput]:
        """Run layout detection with standardized labels."""
        try:
            # Load and preprocess input
            image = Image.open(input_path).convert("RGB")
            
            # Prepare inputs
            prompt = "<OD>"
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate predictions
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                **kwargs
            )
            
            # Decode and post-process
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            
            parsed_result = self.processor.post_process_generation(
                generated_text,
                task="<OD>",
                image_size=(image.width, image.height)
            )
            
            # Convert to standard format
            layout_boxes = []
            for bbox, label in zip(
                parsed_result["<OD>"]["bboxes"],
                parsed_result["<OD>"]["labels"]
            ):
                mapped_label = self.map_label(label.lower())
                if mapped_label:
                    layout_boxes.append(
                        LayoutBox(
                            label=mapped_label,
                            bbox=[float(coord) for coord in bbox],
                            confidence=None  # Florence model doesn't provide confidence scores
                        )
                    )
            
            # Create annotated image
            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Draw boxes and labels
            for box in layout_boxes:
                color = self.color_map.get(box.label, 'gray')
                coords = box.bbox
                draw.rectangle(coords, outline=color, width=3)
                draw.text((coords[0], coords[1]-20), box.label, fill=color)
            
            return annotated_img, LayoutOutput(bboxes=layout_boxes)

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