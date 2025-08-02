import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple, Sequence
from pathlib import Path
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image , ImageDraw
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.layout_analysis.base import BaseLayoutDetector, BaseLayoutMapper
from omnidocs.tasks.layout_analysis.enums import LayoutLabel
from omnidocs.tasks.layout_analysis.models import LayoutBox, LayoutOutput
from omnidocs.utils.model_config import setup_model_environment


logger = get_logger(__name__)

# Setup model environment
_MODELS_DIR = setup_model_environment()

# ================================================================================================================


class RTDETRLayoutMapper(BaseLayoutMapper):
    """Label mapper for RT-DETR layout detection model."""
    
    def _setup_mapping(self):
        mapping = {
            "caption": LayoutLabel.CAPTION,
            "footnote": LayoutLabel.TEXT,  # Map footnote to text
            "formula": LayoutLabel.FORMULA,
            "list-item": LayoutLabel.LIST,
            "page-footer": LayoutLabel.TEXT,  # Map footer to text
            "page-header": LayoutLabel.TEXT,  # Map header to text
            "picture": LayoutLabel.IMAGE,
            "section-header": LayoutLabel.TITLE,  # Map section header to title
            "table": LayoutLabel.TABLE,
            "text": LayoutLabel.TEXT,
            "title": LayoutLabel.TITLE
        }
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class RTDETRLayoutDetector(BaseLayoutDetector):
    """RT-DETR-based layout detection implementation."""

    MODEL_REPO = "HuggingPanda/docling-layout"

    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = 4,
        use_cpu_only: bool = True
    ):
        """Initialize RT-DETR Layout Detector with careful device handling."""
        super().__init__(show_log=show_log)

        self._label_mapper = RTDETRLayoutMapper()

        if self.show_log:
            logger.info("Initializing RTDETRLayoutDetector")

        # Set default paths
        if model_path is None:
            model_path = _MODELS_DIR / "rtdetr_layout" / self.MODEL_REPO.replace("/", "_")

        self.model_path = Path(model_path)
        self.num_threads = num_threads

        # Careful device handling
        if use_cpu_only:
            self.device = "cpu"
            if self.show_log:
                logger.info("Forced CPU usage due to use_cpu_only flag")
        elif device:
            self.device = device
            if self.show_log:
                logger.info(f"Using specified device: {device}")
        else:
            # Check CUDA availability with error handling
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.show_log:
                    logger.info(f"Automatically selected device: {self.device}")
            except Exception as e:
                self.device = "cpu"
                if self.show_log:
                    logger.warning(f"Error checking CUDA availability: {e}. Defaulting to CPU")
        
        self.num_threads = num_threads or int(os.environ.get("OMP_NUM_THREADS", 4))

        # Set thread count for CPU operations
        if self.device == "cpu":
            torch.set_num_threads(self.num_threads)
            if self.show_log:
                logger.info(f"Set CPU threads to {self.num_threads}")

        # Model parameters
        self.image_size = 640
        self.confidence_threshold = 0.6

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
                logger.success("Model initialized successfully")
        except Exception as e:
            if self.show_log:
                logger.error("Failed to initialize model", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as ex:
            logger.error("Failed to import transformers")
            raise ImportError(
                "transformers is not available. Please install it with: pip install transformers"
            ) from ex

    def _model_exists(self) -> bool:
        """Check if model files exist locally."""
        if not self.model_path.exists():
            return False

        # Check for essential files
        config_file = self.model_path / "config.json"
        model_file = self.model_path / "pytorch_model.bin"
        processor_file = self.model_path / "preprocessor_config.json"

        return config_file.exists() and (model_file.exists() or (self.model_path / "model.safetensors").exists()) and processor_file.exists()

    def _download_model(self) -> Path:
        """Download RT-DETR model from HuggingFace if it doesn't exist locally."""
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            if self.show_log:
                logger.info(f"Downloading RT-DETR model: {self.MODEL_REPO}")
                logger.info(f"Saving to: {self.model_path}")

            # Create model directory
            self.model_path.mkdir(parents=True, exist_ok=True)

            # Download and save processor
            if self.show_log:
                logger.info("Downloading processor...")
            processor = RTDetrImageProcessor.from_pretrained(self.MODEL_REPO)
            processor.save_pretrained(self.model_path)

            # Download and save model
            if self.show_log:
                logger.info("Downloading model...")
            model = RTDetrForObjectDetection.from_pretrained(self.MODEL_REPO)
            model.save_pretrained(self.model_path)

            if self.show_log:
                logger.success(f"Model downloaded successfully to {self.model_path}")

            return self.model_path

        except Exception as e:
            logger.error("Error downloading RT-DETR model", exc_info=True)
            # Clean up partial download
            if self.model_path.exists():
                import shutil
                shutil.rmtree(self.model_path)
            raise

    def _load_model(self) -> None:
        """Load RT-DETR model from local path."""
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            if self.show_log:
                logger.info("Loading RT-DETR model from local path...")

            # Load model and processor from local path
            self.image_processor = RTDetrImageProcessor.from_pretrained(str(self.model_path))
            self.model = RTDetrForObjectDetection.from_pretrained(str(self.model_path))

            # Move model to the correct device
            self.model.to(self.device)
            self.model.eval()

            if self.show_log:
                logger.success("RT-DETR model loaded successfully")
            
            # Set model to evaluation mode
            self.model.eval()
            
            if self.show_log:
                logger.info(f"Model ready on device: {self.device}")
            
        except Exception as e:
            if self.show_log:
                logger.error("Error during model loading", exc_info=True)
            raise

    @log_execution_time
    def detect(
        self,
        input_path: Union[str, Path],
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> Tuple[Image.Image, LayoutOutput]:
        """Run layout detection using RT-DETR Transformers model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialization failed.")

        try:
            # Load and preprocess image
            if isinstance(input_path, (str, Path)):
                image = Image.open(input_path).convert("RGB")
            elif isinstance(input_path, Image.Image):
                image = input_path.convert("RGB")
            elif isinstance(input_path, np.ndarray):
                image = Image.fromarray(input_path).convert("RGB")
            else:
                raise ValueError("Unsupported input type")

            # Preprocess the image using the image processor
            resize = {"height": self.image_size, "width": self.image_size}
            inputs = self.image_processor(
                images=image,
                return_tensors="pt",
                size=resize,
            )
            
            # Move inputs to the correct device
            if self.device == "cuda":
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Run inference
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except Exception as e:
                raise RuntimeError(f"Error during model inference: {e}") from e 

            # Post-process results
            threshold = confidence_threshold or self.confidence_threshold
            results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([image.size[::-1]]),
                threshold=threshold
            )

            # Process predictions
            layout_boxes = []
            
            for result in results:
                for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    score_val = float(score.item())
                    label_idx = int(label_id.item())
                    
                    # Get label from model config (add 1 because model config is 0-indexed)
                    model_label = self.model.config.id2label.get(label_idx + 1)
                    if not model_label:
                        continue

                    # Map to standardized label
                    mapped_label = self.map_label(model_label)
                    if not mapped_label:
                        continue

                    # Convert box coordinates (already in image space)
                    box = [round(i, 2) for i in box.tolist()]
                    left, top, right, bottom = box

                    layout_boxes.append(
                        LayoutBox(
                            label=mapped_label,
                            bbox=[left, top, right, bottom],
                            confidence=score_val
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

            return annotated_img, LayoutOutput(bboxes=layout_boxes)

        except Exception as e:
            if self.show_log:
                logger.error("Error during prediction", exc_info=True)
            raise