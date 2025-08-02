from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import time
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.text_extraction.base import BaseTextExtractor, BaseTextMapper, TextOutput, TextBlock
from omnidocs.utils.model_config import setup_model_environment

logger = get_logger(__name__)

# Setup model environment
_MODELS_DIR = setup_model_environment()

class SuryaTextMapper(BaseTextMapper):
    """Label mapper for Surya text model output."""

    def __init__(self):
        super().__init__('surya')

class SuryaTextExtractor(BaseTextExtractor):
    """Surya-based text extraction implementation for images and documents."""

    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        extract_images: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize Surya Text Extractor."""
        super().__init__(device=device, show_log=show_log, engine_name='surya', extract_images=extract_images)

        self._label_mapper = SuryaTextMapper()

        if self.show_log:
            logger.info("Initializing SuryaTextExtractor")

        # Set device if specified, otherwise use default from parent
        if device:
            self.device = device

        if self.show_log:
            logger.info(f"Using device: {self.device}")

        # Set default paths
        if model_path is None:
            model_path = _MODELS_DIR / "surya_text"

        self.model_path = Path(model_path)

        # Check dependencies and load model
        self._check_dependencies()
        self._load_model()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import surya
            if self.show_log:
                logger.info(f"Found surya package at: {surya.__file__}")
        except ImportError:
            raise ImportError(
                "surya-ocr package not found. Please install with: "
                "pip install surya-ocr"
            )

    def _download_model(self) -> Path:
        """Download model from remote source (handled by Surya automatically)."""
        if self.show_log:
            logger.info("Model downloading handled by Surya library")
        return self.model_path

    def _load_model(self):
        """Load Surya text detection and recognition models."""
        try:
            # Import Surya components
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            # Initialize predictors
            self.det_predictor = DetectionPredictor()
            self.rec_predictor = RecognitionPredictor()

            if self.show_log:
                logger.success("Surya text models loaded successfully")

        except Exception as e:
            if self.show_log:
                logger.error("Failed to load Surya text models", exc_info=True)
            raise

    def preprocess_input(self, input_path: Union[str, Path, Image.Image]) -> List[Image.Image]:
        """Preprocess input for Surya text extraction."""
        if isinstance(input_path, Image.Image):
            return [input_path.convert("RGB")]
        elif isinstance(input_path, (str, Path)):
            # Handle image files
            if str(input_path).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image = Image.open(input_path).convert("RGB")
                return [image]
            else:
                # For PDF files, we'd need to convert to images first
                # This is a simplified implementation - you might want to use pdf2image
                raise ValueError(f"Unsupported file type: {input_path}. Surya text extractor works with images.")
        else:
            raise ValueError("Unsupported input type for Surya text extractor")

    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert Surya output to standardized TextOutput format."""
        text_blocks = []
        full_text_parts = []

        if 'predictions' in raw_output:
            for page_idx, prediction in enumerate(raw_output['predictions']):
                if hasattr(prediction, 'text_lines'):
                    for line_idx, text_line in enumerate(prediction.text_lines):
                        # Create text block
                        block = TextBlock(
                            text=text_line.text.strip(),
                            bbox=text_line.bbox if hasattr(text_line, 'bbox') else None,
                            confidence=getattr(text_line, 'confidence', 1.0),
                            page_num=page_idx + 1,
                            block_type='text_line',
                            reading_order=line_idx
                        )
                        text_blocks.append(block)
                        full_text_parts.append(text_line.text.strip())

        # Build metadata
        metadata = {
            'engine': 'surya',
            'total_blocks': len(text_blocks),
            'processing_info': raw_output.get('processing_info', {})
        }

        if source_info:
            metadata.update(source_info)

        return TextOutput(
            text_blocks=text_blocks,
            full_text='\n'.join(full_text_parts),
            metadata=metadata,
            source_info=source_info,
            page_count=len(raw_output.get('predictions', []))
        )

    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TextOutput:
        """Extract text using Surya OCR."""
        start_time = time.time()

        try:
            # Preprocess input
            images = self.preprocess_input(input_path)

            predictions = []

            for img in images:
                # Run text detection and recognition
                try:
                    from surya.common.surya.schema import TaskNames

                    # Use recognition predictor for text extraction
                    prediction = self.rec_predictor(
                        [img],
                        task_names=[TaskNames.ocr_with_boxes],
                        det_predictor=self.det_predictor,
                        math_mode=False  # Standard text mode
                    )

                    if prediction and len(prediction) > 0:
                        predictions.append(prediction[0])

                except Exception as e:
                    if self.show_log:
                        logger.warning(f"Error processing image with Surya: {e}")
                    continue

            # Prepare source info
            source_info = {
                'source_path': str(input_path) if not isinstance(input_path, Image.Image) else 'PIL_Image',
                'num_images': len(images),
                'processing_time': time.time() - start_time
            }

            # Convert to standardized format
            result = self.postprocess_output({
                'predictions': predictions,
                'processing_info': {
                    'total_images': len(images),
                    'successful_predictions': len(predictions)
                }
            }, source_info)

            if self.show_log:
                logger.info(f"Extracted {len(result.text_blocks)} text blocks using Surya")

            return result

        except Exception as e:
            if self.show_log:
                logger.error("Error during Surya text extraction", exc_info=True)
            raise