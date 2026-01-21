"""Base classes for layout detection models.

This module provides abstract base classes that define the common interface
for all layout detection implementations in OmniDocs.
"""

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
    """Base class for mapping model-specific labels to standardized labels.

    This class provides a bidirectional mapping between model-specific label
    strings and the standardized LayoutLabel enum values. Subclasses should
    override `_setup_mapping` to define their specific mappings.

    Attributes:
        _mapping: Dictionary mapping model labels (lowercase) to LayoutLabel.
        _reverse_mapping: Dictionary mapping LayoutLabel to model labels.

    Example:
        >>> class MyModelMapper(BaseLayoutMapper):
        ...     def _setup_mapping(self):
        ...         self._mapping = {"paragraph": LayoutLabel.TEXT}
        ...         self._reverse_mapping = {LayoutLabel.TEXT: "paragraph"}
        >>> mapper = MyModelMapper()
        >>> mapper.to_standard("paragraph")
        <LayoutLabel.TEXT: 'text'>
    """

    def __init__(self):
        self._mapping: Dict[str, LayoutLabel] = {}
        self._reverse_mapping: Dict[LayoutLabel, str] = {}
        self._setup_mapping()
        
    def _setup_mapping(self):
        """Set up the mapping dictionaries.

        This method should be overridden by subclasses to define the
        model-specific label mappings.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError
        
    def to_standard(self, model_label: str) -> Optional[LayoutLabel]:
        """Convert a model-specific label to a standardized LayoutLabel.

        Args:
            model_label: The model-specific label string to convert.
                Case-insensitive.

        Returns:
            The corresponding LayoutLabel enum value, or None if no mapping
            exists for the given label.

        Example:
            >>> mapper.to_standard("Plain Text")
            <LayoutLabel.TEXT: 'text'>
        """
        return self._mapping.get(model_label.lower())
        
    def from_standard(self, layout_label: LayoutLabel) -> Optional[str]:
        """Convert a standardized LayoutLabel to a model-specific label.

        Args:
            layout_label: The LayoutLabel enum value to convert.

        Returns:
            The corresponding model-specific label string, or None if no
            mapping exists.

        Example:
            >>> mapper.from_standard(LayoutLabel.TEXT)
            'plain text'
        """
        return self._reverse_mapping.get(layout_label)


class BaseLayoutDetector(ABC):
    """Abstract base class for all layout detection models.

    This class defines the common interface and shared functionality for
    layout detection implementations. Subclasses must implement the abstract
    methods `_download_model`, `_load_model`, and `detect`.

    Attributes:
        device: The device to run inference on ("cuda" or "cpu").
        model: The loaded model instance (set by subclasses).
        model_path: Path to the model files.
        show_log: Whether to show informational logs.
        color_map: Dictionary mapping label names to visualization colors.

    Example:
        >>> from omnidocs.tasks.layout_analysis import YOLOLayoutDetector
        >>> detector = YOLOLayoutDetector(device="cuda", show_log=True)
        >>> image, output = detector.detect("document.png")
        >>> print(f"Found {len(output.bboxes)} elements")
    """

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
        """Log a message respecting the show_log setting.

        Error messages (level >= ERROR) are always logged regardless of
        the show_log setting.

        Args:
            level: The logging level (e.g., logging.INFO, logging.ERROR).
            msg: The message format string.
            *args: Arguments for the message format string.
            **kwargs: Keyword arguments passed to the logger.
        """
        if self.show_log or level >= logging.ERROR:
            self._logger.log(level, msg, *args, **kwargs)


    @abstractmethod
    def _download_model(self) -> Path:
        """Download the model from a remote source.

        This method should download model weights and any required
        configuration files to the local filesystem.

        Returns:
            Path to the downloaded model directory.

        Raises:
            Exception: If the download fails.
        """
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model into memory.

        This method should load the model weights and prepare the model
        for inference. After calling this method, `self.model` should
        be set to the loaded model instance.

        Raises:
            RuntimeError: If model loading fails.
            ImportError: If required dependencies are missing.
        """
        pass

    @abstractmethod
    def detect(self, input_path: Union[str, Path], **kwargs) -> Tuple[Image.Image, LayoutOutput]:
        """Run layout detection on a single image or first page of a PDF.

        Args:
            input_path: Path to the input image file or PDF document.
            **kwargs: Additional model-specific parameters (e.g.,
                confidence threshold, image size).

        Returns:
            A tuple containing:
                - Annotated PIL Image with detected regions visualized.
                - LayoutOutput object containing all detected boxes.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the input file cannot be read.

        Example:
            >>> detector = YOLOLayoutDetector()
            >>> annotated_img, layout = detector.detect("page.png", conf_threshold=0.5)
            >>> for box in layout.bboxes:
            ...     print(f"{box.label}: {box.confidence:.2f}")
        """
        pass

    def detect_all(self, input_path: Union[str, Path], **kwargs) -> List[Tuple[Image.Image, LayoutOutput]]:
        """Run layout detection on all pages of a document.

        For single images, this returns a list with one result. For PDFs,
        it processes all pages and returns results for each.

        Args:
            input_path: Path to the input image file or PDF document.
            **kwargs: Additional model-specific parameters passed to
                the `detect` method.

        Returns:
            List of tuples, where each tuple contains:
                - Annotated PIL Image for that page.
                - LayoutOutput object with detection results and page number.

        Example:
            >>> detector = YOLOLayoutDetector()
            >>> results = detector.detect_all("document.pdf")
            >>> for img, layout in results:
            ...     print(f"Page {layout.page_number}: {len(layout.bboxes)} elements")
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
        """Save an annotated image and its layout data to files.

        Saves the annotated image to the specified path and automatically
        saves the corresponding layout data as a JSON file with the same
        name but .json extension.

        Args:
            detection_result: Tuple containing the annotated PIL Image
                and its corresponding LayoutOutput.
            output_path: Path where the annotated image will be saved.
                A JSON file will be saved alongside with .json extension.

        Example:
            >>> detector = YOLOLayoutDetector()
            >>> result = detector.detect("document.png")
            >>> detector.visualize(result, "output/annotated.png")
            # Creates: output/annotated.png and output/annotated.json
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
        """Save all annotated images and their layout data to a directory.

        Each page is saved with a numbered filename using the specified
        prefix (e.g., page_1.png, page_2.png).

        Args:
            detection_results: List of (PIL Image, LayoutOutput) tuples
                as returned by `detect_all`.
            output_dir: Directory where all files will be saved. Created
                if it doesn't exist.
            prefix: Prefix for output filenames. Defaults to "page".

        Example:
            >>> detector = YOLOLayoutDetector()
            >>> results = detector.detect_all("document.pdf")
            >>> detector.visualize_all(results, "output/", prefix="doc")
            # Creates: output/doc_1.png, output/doc_1.json, ...
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(detection_results, start=1):
            # Generate output paths
            image_path = output_dir / f"{prefix}_{i}.png"
            
            # Save visualization and JSON
            self.visualize(result, image_path)

    def preprocess_input(self, input_path: Union[str, Path]) -> List[np.ndarray]:
        """Convert input file to a list of processable images.

        Handles both single images and multi-page PDF documents. Images
        are returned as BGR numpy arrays for OpenCV compatibility.

        Args:
            input_path: Path to the input image file or PDF document.

        Returns:
            List of images as numpy arrays in BGR format. For single
            images, returns a list with one element. For PDFs, returns
            one array per page.

        Raises:
            ImportError: If input is a PDF and PyMuPDF is not installed.
            ValueError: If the image file cannot be loaded.

        Example:
            >>> images = detector.preprocess_input("document.pdf")
            >>> print(f"Document has {len(images)} pages")
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
        """Convert PDF pages to a list of numpy arrays using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of page images as numpy arrays in BGR format.

        Raises:
            Exception: If PDF conversion fails.
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
        """Get the label mapper for this detector.

        Returns:
            The BaseLayoutMapper instance used for label conversion.

        Raises:
            ValueError: If the label mapper has not been initialized.
        """
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper
        
    def map_label(self, model_label: str) -> Optional[str]:
        """Map a model-specific label to a standardized label string.

        Args:
            model_label: The model-specific label to convert.

        Returns:
            The standardized label string, or None if no mapping exists.

        Example:
            >>> detector.map_label("Plain Text")
            'text'
        """
        standard_label = self.label_mapper.to_standard(model_label)
        return str(standard_label) if standard_label else None

    def map_box(self, layout_box: LayoutBox) -> LayoutBox:
        """Map the label of a LayoutBox to its standardized form.

        Modifies the LayoutBox in place, replacing its label with the
        standardized version if a mapping exists.

        Args:
            layout_box: The LayoutBox whose label should be mapped.

        Returns:
            The same LayoutBox instance with its label updated.

        Example:
            >>> box = LayoutBox(label="Plain Text", bbox=[0, 0, 100, 50])
            >>> mapped = detector.map_box(box)
            >>> print(mapped.label)
            'text'
        """
        mapped_label = self.map_label(layout_box.label)
        if mapped_label:
            layout_box.label = mapped_label
        return layout_box
    