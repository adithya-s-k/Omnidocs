
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import torch
from PIL import Image
import numpy as np
from omnidocs.tasks.layout_analysis.enums import LayoutLabel
from omnidocs.tasks.layout_analysis.models import LayoutBox, LayoutOutput
from omnidocs.utils.logging import get_logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF (fitz) not installed. PDF processing might be limited.")



class BaseLayoutMapper:
    """Base class for layout label mapping."""
    
    def __init__(self):
        self._mapping: Dict[str, LayoutLabel] = {}
        self._reverse_mapping: Dict[LayoutLabel, str] = {}
        self._setup_mapping()
        
    def _setup_mapping(self):
        """Setup the mapping dictionary. Should be implemented by child classes."""
        raise NotImplementedError
        
    def to_standard(self, model_label: str) -> Optional[LayoutLabel]:
        """Convert model-specific label to standardized LayoutLabel."""
        return self._mapping.get(model_label.lower())
        
    def from_standard(self, layout_label: LayoutLabel) -> Optional[str]:
        """Convert standardized LayoutLabel to model-specific label."""
        return self._reverse_mapping.get(layout_label)


class BaseLayoutDetector(ABC):
    """Base class for all layout detection models."""
    
    def __init__(self, show_log: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_path = None
        self._label_mapper: Optional[BaseLayoutMapper] = None
        self.show_log = show_log  # Changed from show_logs to show_log
        
        # Initialize visualization colors based on standard labels
        self.color_map = {
            str(LayoutLabel.TEXT): 'blue',
            str(LayoutLabel.TITLE): 'red',
            str(LayoutLabel.LIST): 'green',
            str(LayoutLabel.TABLE): 'orange',
            str(LayoutLabel.IMAGE): 'purple',
            str(LayoutLabel.FORMULA): 'yellow',
            str(LayoutLabel.CAPTION): 'cyan'
        }
        
        self._logger = get_logger(__name__)
        if not self.show_log:
            self._logger.setLevel(logging.ERROR)  # Only show errors when show_log is False
        else:
            self._logger.setLevel(logging.INFO)  # Show all logs when show_log is True
            
    def log(self, level: int, msg: str, *args, **kwargs):
        """Wrapper for logging that respects show_log setting."""
        if self.show_log or level >= logging.ERROR:  # Always show errors
            self._logger.log(level, msg, *args, **kwargs)


    @abstractmethod
    def _download_model(self) -> Path:
        """Download model from a remote source."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def detect(self, input_path: Union[str, Path], **kwargs) -> Tuple[Image.Image, LayoutOutput]:
        """
        Run layout detection on a single image/page.
        
        Args:
            input_path: Path to input image or PDF
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple containing:
                - Annotated PIL Image
                - LayoutOutput object with detection results
        """
        pass

    def detect_all(self, input_path: Union[str, Path], **kwargs) -> List[Tuple[Image.Image, LayoutOutput]]:
        """
        Run layout detection on all pages of a document.
        
        Args:
            input_path: Path to input image or PDF
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of tuples, each containing:
                - Annotated PIL Image
                - LayoutOutput object with detection results
        """
        images = self.preprocess_input(input_path)
        results = []
        
        for page_num, image in enumerate(images, start=1):
            # Get detection result for single page
            img_result, layout_output = self.detect(image, **kwargs)
            
            # Add page number to layout output
            layout_output.page_number = page_num
            
            # Add image size if available
            if img_result is not None:
                layout_output.image_size = img_result.size
                
            results.append((img_result, layout_output))
        
        return results

    def visualize(
        self,
        detection_result: Tuple[Image.Image, LayoutOutput],
        output_path: Union[str, Path],
    ) -> None:
        """
        Save annotated image to file.
        
        Args:
            detection_result: Tuple containing (PIL Image, LayoutOutput)
            output_path: Path to save visualization
        """
        annotated_image, layout_output = detection_result
        
        # Convert numpy array to PIL Image if necessary
        if isinstance(annotated_image, np.ndarray):
            annotated_image = Image.fromarray(annotated_image)
            
        if annotated_image is not None:
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            annotated_image.save(str(output_path))
            
            # Save JSON alongside image
            json_path = output_path.with_suffix('.json')
            layout_output.save_json(json_path)

    def visualize_all(
        self,
        detection_results: List[Tuple[Image.Image, LayoutOutput]],
        output_dir: Union[str, Path],
        prefix: str = "page"
    ) -> None:
        """
        Save all annotated images and their layout data to files.
        
        Args:
            detection_results: List of (PIL Image, LayoutOutput) tuples
            output_dir: Directory to save visualizations
            prefix: Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(detection_results, start=1):
            # Generate output paths
            image_path = output_dir / f"{prefix}_{i}.png"
            
            # Save visualization and JSON
            self.visualize(result, image_path)

    def preprocess_input(self, input_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Convert input to processable format.
        
        Args:
            input_path: Path to input image or PDF
            
        Returns:
            List of preprocessed images as numpy arrays
        """
        input_path = Path(input_path)

        if input_path.suffix.lower() == ".pdf":
            if fitz is None:
                raise ImportError("PyMuPDF (fitz) is required for PDF processing. Please install it with: pip install PyMuPDF")
            return self._convert_pdf_to_images_pymupdf(input_path)
        else:
            # Load single image
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")
            return [image]

    def _convert_pdf_to_images_pymupdf(self, pdf_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Convert PDF pages to a list of numpy arrays (images) using PyMuPDF.
        """
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                # Convert to BGR for OpenCV compatibility if needed by detect method
                if pix.n == 3: # RGB
                    images.append(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                elif pix.n == 4: # RGBA
                    images.append(cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR))
                else: # Grayscale or other
                    images.append(img_array)
            doc.close()
        except Exception as e:
            self.log(logging.ERROR, f"Error converting PDF with PyMuPDF: {e}")
            raise
        return images

    @property
    def label_mapper(self) -> BaseLayoutMapper:
        """Get the label mapper for this detector."""
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper
        
    def map_label(self, model_label: str) -> Optional[str]:
        """Map model-specific label to standardized label."""
        standard_label = self.label_mapper.to_standard(model_label)
        return str(standard_label) if standard_label else None

    def map_box(self, layout_box: LayoutBox) -> LayoutBox:
        """Map LayoutBox label to standardized label."""
        mapped_label = self.map_label(layout_box.label)
        if mapped_label:
            layout_box.label = mapped_label
        return layout_box
    