from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell
from omnidocs.utils.model_config import setup_model_environment

logger = get_logger(__name__)

# Setup model environment
_MODELS_DIR = setup_model_environment()

class SuryaTableMapper(BaseTableMapper):
    """Label mapper for Surya table model output."""

    def __init__(self):
        super().__init__('surya')

class SuryaTableExtractor(BaseTableExtractor):
    """Surya-based table extraction implementation."""

    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize Surya Table Extractor."""
        super().__init__(device=device, show_log=show_log, engine_name='surya')

        self._label_mapper = SuryaTableMapper()

        if self.show_log:
            logger.info("Initializing SuryaTableExtractor")

        # Set device if specified, otherwise use default from parent
        if device:
            self.device = device

        if self.show_log:
            logger.info(f"Using device: {self.device}")

        # Set default paths
        if model_path is None:
            model_path = _MODELS_DIR / "surya_table"

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
            ) from None 

    def _load_model(self):
        """Load Surya table detection and recognition models."""
        try:
            # Import Surya components
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.layout import LayoutPredictor

            # Initialize predictors
            self.det_predictor = DetectionPredictor()
            self.rec_predictor = RecognitionPredictor()
            self.layout_predictor = LayoutPredictor()

            if self.show_log:
                logger.success("Surya table models loaded successfully")

        except Exception as e:
            if self.show_log:
                logger.error("Failed to load Surya table models", exc_info=True)
            raise

    def postprocess_output(self, raw_output: Any, img_size: Tuple[int, int]) -> TableOutput:
        """Convert Surya output to standardized TableOutput format."""
        tables = []

        if 'tables' in raw_output:
            for table_idx, table_data in enumerate(raw_output['tables']):
                # Extract table cells with proper mapping
                cells = []

                # Handle different possible structures from Surya
                if 'cells' in table_data:
                    # Direct cell data
                    for cell_data in table_data['cells']:
                        cell = self._create_table_cell(cell_data, table_idx)
                        if cell:
                            cells.append(cell)
                elif 'text_lines' in table_data:
                    # Convert text lines to cells
                    cells = self._text_lines_to_cells(table_data['text_lines'], table_data.get('bbox', [0, 0, img_size[0], img_size[1]]))

                if cells:
                    # Calculate table dimensions
                    num_rows = max(c.row for c in cells) + 1 if cells else 0
                    num_cols = max(c.col for c in cells) + 1 if cells else 0

                    # Create table
                    table = Table(
                        cells=cells,
                        bbox=table_data.get('bbox', [0, 0, img_size[0], img_size[1]]),
                        confidence=table_data.get('confidence', 1.0),
                        num_rows=num_rows,
                        num_cols=num_cols,
                        table_id=f"surya_table_{table_idx}"
                    )
                    tables.append(table)

        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={'engine': 'surya', 'raw_output': raw_output}
        )

    def _create_table_cell(self, cell_data: Dict, table_idx: int) -> Optional[TableCell]:
        """Create a TableCell from Surya cell data."""
        try:
            # Map different possible field names
            text = cell_data.get('text', cell_data.get('content', ''))
            bbox = cell_data.get('bbox', cell_data.get('bounding_box', [0, 0, 0, 0]))
            confidence = cell_data.get('confidence', cell_data.get('score', 1.0))

            # Handle row/col mapping
            row = cell_data.get('row', cell_data.get('row_index', 0))
            col = cell_data.get('col', cell_data.get('col_index', cell_data.get('column', 0)))

            # Handle span mapping
            rowspan = cell_data.get('rowspan', cell_data.get('row_span', 1))
            colspan = cell_data.get('colspan', cell_data.get('col_span', 1))

            return TableCell(
                text=str(text).strip(),
                row=int(row),
                col=int(col),
                rowspan=int(rowspan),
                colspan=int(colspan),
                bbox=bbox,
                confidence=float(confidence) if confidence is not None else 1.0,
                is_header=row == 0  # Simple heuristic: first row is header
            )
        except (ValueError, TypeError) as e:
            if self.show_log:
                logger.warning(f"Error creating table cell: {e}")
            return None

    def _text_lines_to_cells(self, text_lines: List, table_bbox: List[float]) -> List[TableCell]:
        """Convert Surya text lines to table cells with structure analysis."""
        if not text_lines:
            return []

        cells = []

        # Sort text lines by position (top to bottom, left to right)
        sorted_lines = sorted(text_lines, key=lambda x: (x.bbox[1], x.bbox[0]))

        # Group lines into rows based on y-coordinate proximity
        rows = self._group_lines_into_rows(sorted_lines)

        # Convert rows to cells with row/column indices
        for row_idx, row_lines in enumerate(rows):
            for col_idx, text_line in enumerate(row_lines):
                # Adjust bbox relative to original image coordinates if needed
                bbox = text_line.bbox
                if hasattr(text_line, 'bbox') and table_bbox:
                    # If bbox is relative to table, adjust to image coordinates
                    if all(coord <= 1.0 for coord in bbox):  # Normalized coordinates
                        bbox = [
                            bbox[0] * (table_bbox[2] - table_bbox[0]) + table_bbox[0],
                            bbox[1] * (table_bbox[3] - table_bbox[1]) + table_bbox[1],
                            bbox[2] * (table_bbox[2] - table_bbox[0]) + table_bbox[0],
                            bbox[3] * (table_bbox[3] - table_bbox[1]) + table_bbox[1]
                        ]

                cell = TableCell(
                    text=text_line.text.strip(),
                    row=row_idx,
                    col=col_idx,
                    rowspan=1,
                    colspan=1,
                    bbox=bbox,
                    confidence=getattr(text_line, 'confidence', 1.0),
                    is_header=row_idx == 0
                )
                cells.append(cell)

        return cells

    def _group_lines_into_rows(self, sorted_lines: List) -> List[List]:
        """Group text lines into rows based on y-coordinate proximity."""
        if not sorted_lines:
            return []

        rows = []
        current_row = []
        current_y = None
        y_tolerance = 10  # pixels

        for line in sorted_lines:
            line_y = (line.bbox[1] + line.bbox[3]) / 2  # center y

            if current_y is None or abs(line_y - current_y) <= y_tolerance:
                current_row.append(line)
                current_y = line_y if current_y is None else (current_y + line_y) / 2
            else:
                if current_row:
                    # Sort current row by x-coordinate
                    rows.append(sorted(current_row, key=lambda x: x.bbox[0]))
                current_row = [line]
                current_y = line_y

        if current_row:
            rows.append(sorted(current_row, key=lambda x: x.bbox[0]))

        return rows

    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using Surya."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            image = images[0]
            img_size = image.size

            # Convert PIL to RGB if needed
            if isinstance(image, Image.Image):
                img_rgb = image.convert("RGB")
            else:
                img_rgb = Image.fromarray(image).convert("RGB")

            # Step 1: Use layout detection to find table regions
            layout_predictions = self.layout_predictor([img_rgb])

            tables_data = []

            if layout_predictions and len(layout_predictions) > 0:
                layout_pred = layout_predictions[0]

                # Find table regions from layout
                table_regions = []
                for bbox_obj in layout_pred.bboxes:
                    if hasattr(bbox_obj, 'label') and 'table' in bbox_obj.label.lower():
                        table_regions.append({
                            'bbox': bbox_obj.bbox,
                            'confidence': getattr(bbox_obj, 'confidence', 1.0)
                        })

                # Step 2: For each table region, extract text and structure
                for table_region in table_regions:
                    bbox = table_region['bbox']

                    # Crop table region
                    table_img = img_rgb.crop(bbox)

                    # Step 3: Run OCR on table region
                    try:
                        from surya.common.surya.schema import TaskNames

                        # Use recognition predictor for table text extraction
                        predictions = self.rec_predictor(
                            [table_img],
                            task_names=[TaskNames.ocr_with_boxes],
                            det_predictor=self.det_predictor,
                            math_mode=False
                        )

                        # Process OCR results into table structure
                        if predictions and len(predictions) > 0:
                            prediction = predictions[0]

                            # Extract text lines and organize into table structure
                            cells = self._organize_text_into_table(prediction.text_lines, bbox)

                            table_data = {
                                'bbox': bbox,
                                'confidence': table_region['confidence'],
                                'cells': cells,
                                'num_rows': len(set(c['row'] for c in cells)) if cells else 0,
                                'num_cols': len(set(c['col'] for c in cells)) if cells else 0
                            }
                            tables_data.append(table_data)

                    except Exception as e:
                        if self.show_log:
                            logger.warning(f"Error processing table region: {e}")
                        continue

            # Convert to standardized format
            result = self.postprocess_output({'tables': tables_data}, img_size)

            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using Surya")

            return result

        except Exception as e:
            if self.show_log:
                logger.error("Error during Surya table extraction", exc_info=True)
            raise

    def _organize_text_into_table(self, text_lines, table_bbox: List[float]) -> List[Dict]:
        """Organize detected text lines into table structure."""
        cells = []

        if not text_lines:
            return cells

        rows = self._group_lines_into_rows(text_lines)

        # Convert rows to cells with row/column indices
        for row_idx, row in enumerate(rows):
            for col_idx, text_line in enumerate(row):
                # Adjust bbox relative to original image coordinates
                adjusted_bbox = [
                    text_line.bbox[0] + table_bbox[0],
                    text_line.bbox[1] + table_bbox[1],
                    text_line.bbox[2] + table_bbox[0],
                    text_line.bbox[3] + table_bbox[1]
                ]

                cell = {
                    'text': text_line.text.strip(),
                    'bbox': adjusted_bbox,
                    'confidence': getattr(text_line, 'confidence', 1.0),
                    'row': row_idx,
                    'col': col_idx,
                    'row_span': 1,
                    'col_span': 1
                }
                cells.append(cell)

        return cells