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
from omnidocs.utils.model_config import setup_model_environment
from omnidocs.core import ModelRegistry, ModelLoader, YOLOModelConfig

logger = get_logger(__name__)

# Setup model environment
_MODELS_DIR = setup_model_environment()

class YOLOLayoutMapper(BaseLayoutMapper):
    """Label mapper for YOLO layout detection model."""
    
    def _setup_mapping(self):
        mapping = {
            "plain text": LayoutLabel.TEXT,
            "title": LayoutLabel.TITLE,
            "figure": LayoutLabel.IMAGE,
            "isolate_formula": LayoutLabel.FORMULA,
            "figure_caption": LayoutLabel.CAPTION,
            "table": LayoutLabel.TABLE
        }
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class YOLOLayoutDetector(BaseLayoutDetector):
    """YOLO-based layout detection implementation."""

    # Model ID in the registry
    MODEL_ID = "doclayout-yolo-docstructbench"

    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_path: Optional[Union[str, Path]] = None
    ):
        """Initialize YOLO Layout Detector."""
        super().__init__(show_log=show_log)

        # Get model configuration from registry
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        if not self.model_config:
            raise ValueError(f"Model '{self.MODEL_ID}' not found in registry")

        if not isinstance(self.model_config, YOLOModelConfig):
            raise TypeError(f"Expected YOLOModelConfig, got {type(self.model_config)}")

        self._label_mapper = YOLOLayoutMapper()
        if self.show_log:
            logger.info(f"Initializing YOLOLayoutDetector with model: {self.model_config.name}")

        if device:
            self.device = device
        if self.show_log:
            logger.info(f"Using device: {self.device}")

        # Set model path from config or override
        if model_path is None:
            self.model_path = ModelLoader.get_model_path(self.model_config)
        else:
            self.model_path = Path(model_path)

        if self.show_log:
            logger.info(f"Model directory: {self.model_path}")

        # Use configuration values
        self.conf_threshold = self.model_config.confidence_threshold
        self.img_size = self.model_config.imgsz

        # Check dependencies
        self._check_dependencies()

        # Download model if needed
        if not self._model_exists():
            if self.show_log:
                logger.info(f"Model not found at {self.model_path}, will download from HuggingFace")
            self._download_model()

        # Load model
        try:
            self._load_model()
            if self.show_log:
                logger.success(f"Model initialized successfully: {self.model_config.description}")
        except Exception as e:
            if self.show_log:
                logger.error("Failed to initialize model", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as ex:
            logger.error("Failed to import doclayout-yolo")
            raise ImportError(
                "doclayout-yolo is not available. Please install it with: pip install doclayout-yolo"
            ) from ex

    def _model_exists(self) -> bool:
        """Check if model files exist locally."""
        model_file = self.model_path / self.model_config.model_filename
        return model_file.exists()

    def _download_model(self) -> Path:
        """Download model from HuggingFace Hub."""
        try:
            if self.show_log:
                logger.info(f"Downloading YOLO model from {self.model_config.model_repo}...")
                logger.info(f"Saving to: {self.model_path}")

            # Use ModelLoader for consistent downloading
            model_file = ModelLoader.download_yolo_model(self.model_config)

            if self.show_log:
                logger.success(f"Model downloaded successfully to {model_file}")

            return self.model_path

        except Exception as e:
            logger.error("Error downloading YOLO model", exc_info=True)
            # Clean up partial download
            if self.model_path.exists():
                import shutil
                shutil.rmtree(self.model_path)
            raise

    def _load_model(self) -> None:
        """Load YOLOv10 model from local path."""
        try:
            from doclayout_yolo import YOLOv10

            model_file = self.model_path / self.model_config.model_filename
            if self.show_log:
                logger.info(f"Loading YOLO model from {model_file}")

            self.model = YOLOv10(str(model_file))
            if self.show_log:
                logger.success(f"YOLO model loaded successfully on {self.device}")
        except ImportError:
            if self.show_log:
                logger.error("Failed to import doclayout_yolo")
            raise ImportError(
                "Failed to import doclayout_yolo. "
                "Please install it with: pip install doclayout-yolo"
            )

    @log_execution_time
    def detect(
        self,
        input_path: Union[str, Path],
        conf_threshold: float = None,
        img_size: int = None,
        **kwargs,
    ) -> Tuple[Image.Image, LayoutOutput]:
        """Run layout detection with standardized labels."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialization failed.")

        conf = conf_threshold if conf_threshold else self.conf_threshold
        imgsz = img_size if img_size else self.img_size

        try:
            images = self.preprocess_input(input_path)
            
            results = []
            for img in images:
                # Get detection results
                det_result = self.model.predict(
                    img, imgsz=imgsz, conf=conf, device=self.device, **kwargs
                )
                
                # Convert detection results to LayoutBox objects
                layout_boxes = []
                for box in det_result[0].boxes:
                    model_label = det_result[0].names[int(box.cls[0])]
                    mapped_label = self.map_label(model_label)
                    
                    if mapped_label:
                        layout_boxes.append(
                            LayoutBox(
                                label=mapped_label,
                                bbox=box.xyxy[0].tolist(),
                                confidence=float(box.conf[0]) if box.conf is not None else None
                            )
                        )
                
                # Get the annotated image (will be a numpy array)
                annotated_img_array = det_result[0].plot(labels=False)  # Disable YOLO's default labels
                
                # Convert numpy array to PIL Image
                annotated_img = Image.fromarray(cv2.cvtColor(annotated_img_array, cv2.COLOR_BGR2RGB))
                
                # Draw standardized labels on the image
                draw = ImageDraw.Draw(annotated_img)
                for box in layout_boxes:
                    color = self.color_map.get(box.label, 'gray')
                    coords = box.bbox
                    draw.rectangle(coords, outline=color, width=3)
                    draw.text((coords[0], coords[1]-20), box.label, fill=color)
                
                results.append((
                    annotated_img,
                    LayoutOutput(bboxes=layout_boxes)
                ))

            return results[0] if results else (None, LayoutOutput(bboxes=[]))

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
        Save the annotated image to file.
        
        Args:
            detection_result: Tuple containing (PIL Image, LayoutOutput)
            output_path: Path to save visualization
        """
        annotated_image, _ = detection_result
        
        # Convert numpy array to PIL Image if necessary
        if isinstance(annotated_image, np.ndarray):
            annotated_image = Image.fromarray(annotated_image)
            
        if annotated_image is not None:
            annotated_image.save(str(output_path))
            