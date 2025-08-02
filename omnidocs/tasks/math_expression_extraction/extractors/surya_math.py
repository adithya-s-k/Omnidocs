from pathlib import Path
from typing import Union,  Optional
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput
from omnidocs.utils.model_config import setup_model_environment

logger = get_logger(__name__)

# Setup model environment
_MODELS_DIR = setup_model_environment()

class SuryaMathMapper(BaseLatexMapper):
    """Label mapper for Surya math model output."""

    def _setup_mapping(self):
        # Surya outputs LaTeX directly, minimal mapping needed
        mapping = {
            r"\n": " ",     # Remove newlines
            r"  ": " ",     # Remove double spaces
            r"\\\\": r"\\", # Fix double backslashes
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class SuryaMathExtractor(BaseLatexExtractor):
    """Surya-based mathematical expression extraction implementation."""

    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize Surya Math Extractor."""
        super().__init__(device=device, show_log=show_log)

        self._label_mapper = SuryaMathMapper()

        if self.show_log:
            logger.info("Initializing SuryaMathExtractor")

        # Set device if specified, otherwise use default from parent
        if device:
            self.device = device

        if self.show_log:
            logger.info(f"Using device: {self.device}")

        # Set default paths
        if model_path is None:
            model_path = _MODELS_DIR / "surya_math"

        self.model_path = Path(model_path)

        # Check dependencies and load model
        self._check_dependencies()
        self._load_model()

    def _download_model(self) -> Path:
        """Download model from remote source (handled by Surya automatically)."""
        if self.show_log:
            logger.info("Model downloading handled by Surya library")
        return self.model_path

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

    def _load_model(self):
        """Load Surya math models."""
        try:
            # Import Surya math detection components
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            # Initialize predictors
            self.det_predictor = DetectionPredictor()
            self.rec_predictor = RecognitionPredictor()

            if self.show_log:
                logger.success("Surya math models loaded successfully")

        except Exception as e:
            if self.show_log:
                logger.error("Failed to load Surya math models", exc_info=True)
            raise

    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Surya."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)

            expressions = []
            confidences = []
            bboxes = []

            for img in images:
                # Convert PIL to RGB if needed
                if isinstance(img, Image.Image):
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = Image.fromarray(img).convert("RGB")

                # Run math detection and recognition
                try:
                    # Import TaskNames for proper task specification
                    from surya.common.surya.schema import TaskNames

                    # Use recognition predictor with math mode enabled
                    predictions = self.rec_predictor(
                        [img_rgb],
                        task_names=[TaskNames.ocr_with_boxes],
                        det_predictor=self.det_predictor,
                        math_mode=True  # Enable math mode for LaTeX output
                    )

                    # Process predictions
                    if predictions and len(predictions) > 0:
                        prediction = predictions[0]

                        # Extract text regions that contain math
                        for text_line in prediction.text_lines:
                            text_content = text_line.text.strip()

                            # Check if this looks like math content
                            if self._is_math_content(text_content):
                                # Map to standard format
                                mapped_expr = self.map_expression(text_content)
                                expressions.append(mapped_expr)

                                # Add confidence if available
                                if hasattr(text_line, 'confidence'):
                                    confidences.append(text_line.confidence)
                                else:
                                    confidences.append(1.0)

                                # Add bounding box if available
                                if hasattr(text_line, 'bbox'):
                                    bboxes.append(text_line.bbox)
                                else:
                                    bboxes.append([0, 0, img_rgb.width, img_rgb.height])

                except Exception as e:
                    if self.show_log:
                        logger.warning(f"Error processing image with Surya: {e}")
                    # Fallback: return empty result for this image
                    continue

            return LatexOutput(
                expressions=expressions,
                confidences=confidences if confidences else None,
                bboxes=bboxes if bboxes else None,
                source_img_size=images[0].size if images else None
            )

        except Exception as e:
            if self.show_log:
                logger.error("Error during Surya math extraction", exc_info=True)
            raise

    def _is_math_content(self, text: str) -> bool:
        """Check if text content appears to be mathematical."""
        # Simple heuristics to identify math content
        math_indicators = [
            '\\', '$', '^', '_', '{', '}',
            'frac', 'sum', 'int', 'sqrt', 'alpha', 'beta', 'gamma',
            'theta', 'pi', 'sigma', 'delta', 'epsilon'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in math_indicators)