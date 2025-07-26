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


logger = get_logger(__name__)

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
    
    MODEL_REPO = "ds4sd/docling-models"
    MODEL_CHECKPOINT = "model_artifacts/layout/beehive_v0.0.5_pt/model.pt"
    DEFAULT_LOCAL_DIR = "./models/RTDETR-Layout"
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        local_dir: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = 4,
        use_cpu_only: bool = True
    ):
        """Initialize RT-DETR Layout Detector with careful device handling."""
        super().__init__(show_log=show_log)
        
        self._label_mapper = RTDETRLayoutMapper()
        
        if self.show_log:
            logger.info("Initializing RTDETRLayoutDetector")
        
        
        """
        
        TODO:Fix GPU based inference
        
        GPU mode not working because of tensors not being on the same device
        
        
        """
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
        
        self.local_dir = Path(local_dir) if local_dir else Path(self.DEFAULT_LOCAL_DIR)
        self.num_threads = num_threads or int(os.environ.get("OMP_NUM_THREADS", 4))
        
        # Set thread count for CPU operations
        if self.device == "cpu":
            torch.set_num_threads(self.num_threads)
            if self.show_log:
                logger.info(f"Set CPU threads to {self.num_threads}")
        
        # Model parameters
        self.image_size = 640
        self.confidence_threshold = 0.6
        
        try:
            self._download_model()
            self._load_model()
            if self.show_log:
                logger.success("Model initialized successfully")
        except Exception as e:
            if self.show_log:
                logger.error("Failed to initialize model", exc_info=True)
            raise

    @log_execution_time
    def _download_model(self) -> Path:
        """Download model from HuggingFace Hub with error handling."""
        try:
            if self.show_log:
                logger.info(f"Downloading model from {self.MODEL_REPO}...")
            
            model_dir = snapshot_download(
                repo_id=self.MODEL_REPO,
                local_dir=str(self.local_dir)
            )
            self.model_path = Path(model_dir) / self.MODEL_CHECKPOINT
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            if self.show_log:
                logger.success(f"Model downloaded to {self.model_path}")
            return self.model_path
            
        except Exception as e:
            if self.show_log:
                logger.error("Failed to download model", exc_info=True)
            raise

    @log_execution_time
    def _load_model(self) -> None:
        """Load RT-DETR model with robust error handling."""
        try:
            # Ensure model file exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Load model with error handling
            try:
                # Load model on CPU first
                self.model = torch.jit.load(self.model_path, map_location='cpu')
                if self.show_log:
                    logger.info("Model loaded successfully on CPU")
                
                # Move to specified device if CUDA
                if self.device == "cuda":
                    try:
                        self.model = self.model.cuda()
                        if self.show_log:
                            logger.info("Model successfully moved to CUDA")
                    except Exception as e:
                        self.device = "cpu"  # Fallback to CPU
                        if self.show_log:
                            logger.warning(f"Failed to move model to CUDA: {e}. Falling back to CPU")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
            
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
        """Run layout detection with standardized labels and robust error handling."""
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

            # Get original size and ensure it's on the same device as model
            w, h = image.size
            orig_size = torch.tensor([w, h])[None]  # Create on CPU first
            
            # Prepare transforms
            transforms = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ])

            # Transform image (on CPU first)
            img = transforms(image).unsqueeze(0)
            
            # Move everything to the correct device together
            if self.device == "cuda":
                # Ensure model is on CUDA
                self.model = self.model.cuda()
                # Move tensors to CUDA
                img = img.cuda()
                orig_size = orig_size.cuda()
            else:
                # Ensure model is on CPU
                self.model = self.model.cpu()
                # Keep tensors on CPU
                img = img.cpu()
                orig_size = orig_size.cpu()

            # Run inference with error handling
            try:
                with torch.no_grad():
                    labels, boxes, scores = self.model(img, orig_size)
            except Exception as e:
                raise RuntimeError(f"Error during model inference: {e}")

            # Process predictions
            layout_boxes = []
            threshold = confidence_threshold or self.confidence_threshold

            for label_idx, box, score in zip(labels[0], boxes[0], scores[0]):
                score_val = float(score.item())
                if score_val < threshold:
                    continue

                # Get label and map to standard format
                label_idx = int(label_idx.item()) + 1  # Add 1 as background is 0
                model_label = {
                    1: "caption", 2: "footnote", 3: "formula",
                    4: "list-item", 5: "page-footer", 6: "page-header",
                    7: "picture", 8: "section-header", 9: "table",
                    10: "text", 11: "title"
                }.get(label_idx)

                if not model_label:
                    continue

                mapped_label = self.map_label(model_label)
                if not mapped_label:
                    continue

                # Convert coordinates to image space
                l = min(w, max(0, float(box[0])))
                t = min(h, max(0, float(box[1])))
                r = min(w, max(0, float(box[2])))
                b = min(h, max(0, float(box[3])))

                layout_boxes.append(
                    LayoutBox(
                        label=mapped_label,
                        bbox=[l, t, r, b],
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
