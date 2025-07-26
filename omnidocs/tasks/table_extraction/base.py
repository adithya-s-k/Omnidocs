from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import torch
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class TableCell(BaseModel):
    """
    Container for individual table cell.
    
    Attributes:
        text: Cell text content
        row: Row index (0-based)
        col: Column index (0-based)
        rowspan: Number of rows the cell spans
        colspan: Number of columns the cell spans
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Confidence score for cell detection
        is_header: Whether the cell is a header cell
    """
    text: str = Field(..., description="Cell text content")
    row: int = Field(..., description="Row index (0-based)")
    col: int = Field(..., description="Column index (0-based)")
    rowspan: int = Field(1, description="Number of rows the cell spans")
    colspan: int = Field(1, description="Number of columns the cell spans")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
    is_header: bool = Field(False, description="Whether the cell is a header cell")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'row': self.row,
            'col': self.col,
            'rowspan': self.rowspan,
            'colspan': self.colspan,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'is_header': self.is_header
        }

class Table(BaseModel):
    """
    Container for extracted table.
    
    Attributes:
        cells: List of table cells
        num_rows: Number of rows in the table
        num_cols: Number of columns in the table
        bbox: Bounding box of the entire table [x1, y1, x2, y2]
        confidence: Overall table detection confidence
        table_id: Optional table identifier
        caption: Optional table caption
        structure_confidence: Confidence score for table structure detection
    """
    cells: List[TableCell] = Field(..., description="List of table cells")
    num_rows: int = Field(..., description="Number of rows in the table")
    num_cols: int = Field(..., description="Number of columns in the table")
    bbox: Optional[List[float]] = Field(None, description="Table bounding box [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Overall table detection confidence")
    table_id: Optional[str] = Field(None, description="Table identifier")
    caption: Optional[str] = Field(None, description="Table caption")
    structure_confidence: Optional[float] = Field(None, description="Table structure detection confidence")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'cells': [cell.to_dict() for cell in self.cells],
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'table_id': self.table_id,
            'caption': self.caption,
            'structure_confidence': self.structure_confidence
        }
    
    def to_csv(self) -> str:
        """Convert table to CSV format."""
        import csv
        import io
        
        # Create a grid to store cell values
        grid = [[''] * self.num_cols for _ in range(self.num_rows)]
        
        # Fill the grid with cell values
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if r < self.num_rows and c < self.num_cols:
                        grid[r][c] = cell.text
        
        # Convert to CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(grid)
        return output.getvalue()
    
    def to_html(self) -> str:
        """Convert table to HTML format."""
        html = ['<table>']
        
        # Create a grid to track cell positions and spans
        grid = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        
        # Mark occupied cells
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if r < self.num_rows and c < self.num_cols:
                        grid[r][c] = cell if r == cell.row and c == cell.col else 'occupied'
        
        # Generate HTML rows
        for row_idx in range(self.num_rows):
            html.append('  <tr>')
            for col_idx in range(self.num_cols):
                cell_data = grid[row_idx][col_idx]
                if isinstance(cell_data, TableCell):
                    tag = 'th' if cell_data.is_header else 'td'
                    attrs = []
                    if cell_data.rowspan > 1:
                        attrs.append(f'rowspan="{cell_data.rowspan}"')
                    if cell_data.colspan > 1:
                        attrs.append(f'colspan="{cell_data.colspan}"')
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html.append(f'    <{tag}{attr_str}>{cell_data.text}</{tag}>')
                elif cell_data is None:
                    html.append('    <td></td>')
                # Skip 'occupied' cells as they're part of a span
            html.append('  </tr>')
        
        html.append('</table>')
        return '\n'.join(html)

class TableOutput(BaseModel):
    """
    Container for table extraction results.
    
    Attributes:
        tables: List of extracted tables
        source_img_size: Original image dimensions (width, height)
        processing_time: Time taken for table extraction
        metadata: Additional metadata from the extraction engine
    """
    tables: List[Table] = Field(..., description="List of extracted tables")
    source_img_size: Optional[Tuple[int, int]] = Field(None, description="Original image dimensions (width, height)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional extraction engine metadata")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'tables': [table.to_dict() for table in self.tables],
            'source_img_size': self.source_img_size,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save output to JSON file."""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_tables_by_confidence(self, min_confidence: float = 0.5) -> List[Table]:
        """Filter tables by minimum confidence threshold."""
        return [table for table in self.tables if table.confidence is None or table.confidence >= min_confidence]
    
    def save_tables_as_csv(self, output_dir: Union[str, Path]) -> List[Path]:
        """Save all tables as separate CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, table in enumerate(self.tables):
            filename = f"table_{table.table_id or i}.csv"
            file_path = output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(table.to_csv())
            saved_files.append(file_path)
        
        return saved_files

class BaseTableMapper:
    """Base class for mapping table extraction engine-specific outputs to standardized format."""
    
    def __init__(self, engine_name: str):
        """Initialize mapper for specific table extraction engine.
        
        Args:
            engine_name: Name of the table extraction engine
        """
        self.engine_name = engine_name.lower()
    
    def normalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Normalize bounding box coordinates to absolute pixel values."""
        if all(0 <= coord <= 1 for coord in bbox):
            return [
                bbox[0] * img_width,
                bbox[1] * img_height,
                bbox[2] * img_width,
                bbox[3] * img_height
            ]
        return bbox
    
    def detect_header_rows(self, cells: List[TableCell]) -> List[TableCell]:
        """Detect and mark header cells based on position and formatting."""
        # Simple heuristic: first row is likely header
        if not cells:
            return cells
        
        first_row_cells = [cell for cell in cells if cell.row == 0]
        for cell in first_row_cells:
            cell.is_header = True
        
        return cells

class BaseTableExtractor(ABC):
    """Base class for table extraction models."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 engine_name: Optional[str] = None):
        """Initialize the table extractor.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            show_log: Whether to show detailed logs
            engine_name: Name of the table extraction engine
        """
        self.show_log = show_log
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine_name = engine_name or self.__class__.__name__.lower().replace('extractor', '')
        self.model = None
        self.model_path = None
        self._label_mapper: Optional[BaseTableMapper] = None
        
        # Initialize mapper if engine name is provided
        if self.engine_name:
            self._label_mapper = BaseTableMapper(self.engine_name)
        
        if self.show_log:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Engine: {self.engine_name}")
    
    @abstractmethod
    def _download_model(self) -> Path:
        """Download model from remote source."""
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load model into memory."""
        pass
    
    def preprocess_input(self, input_path: Union[str, Path, Image.Image, np.ndarray]) -> List[Image.Image]:
        """Convert input to list of PIL Images.
        
        Args:
            input_path: Input image path or image data
            
        Returns:
            List of PIL Images
        """
        if isinstance(input_path, (str, Path)):
            image = Image.open(input_path).convert('RGB')
            return [image]
        elif isinstance(input_path, Image.Image):
            return [input_path.convert('RGB')]
        elif isinstance(input_path, np.ndarray):
            return [Image.fromarray(cv2.cvtColor(input_path, cv2.COLOR_BGR2RGB))]
        else:
            raise ValueError(f"Unsupported input type: {type(input_path)}")
    
    def postprocess_output(self, raw_output: Any, img_size: Tuple[int, int]) -> TableOutput:
        """Convert raw table extraction output to standardized TableOutput format.
        
        Args:
            raw_output: Raw output from table extraction engine
            img_size: Original image size (width, height)
            
        Returns:
            Standardized TableOutput object
        """
        raise NotImplementedError("Child classes must implement postprocess_output method")
    
    @abstractmethod
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables from input image.
        
        Args:
            input_path: Path to input image or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            TableOutput containing extracted tables
        """
        pass
    
    def extract_all(
        self,
        input_paths: List[Union[str, Path, Image.Image]],
        **kwargs
    ) -> List[TableOutput]:
        """Extract tables from multiple images.
        
        Args:
            input_paths: List of image paths or image data
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of TableOutput objects
        """
        results = []
        for input_path in input_paths:
            try:
                result = self.extract(input_path, **kwargs)
                results.append(result)
            except Exception as e:
                if self.show_log:
                    logger.error(f"Error processing {input_path}: {str(e)}")
                raise
        return results
    
    def extract_with_layout(
        self,
        input_path: Union[str, Path, Image.Image],
        layout_regions: Optional[List[Dict]] = None,
        **kwargs
    ) -> TableOutput:
        """Extract tables with optional layout information.
        
        Args:
            input_path: Path to input image or image data
            layout_regions: Optional list of layout regions containing tables
            **kwargs: Additional model-specific parameters
            
        Returns:
            TableOutput containing extracted tables
        """
        # Default implementation just calls extract, can be overridden by child classes
        return self.extract(input_path, **kwargs)
    
    @property
    def label_mapper(self) -> BaseTableMapper:
        """Get the label mapper for this extractor."""
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper

    def _convert_pdf_to_image(self, pdf_path: Union[str, Path]) -> Image.Image:
        """Convert PDF first page to image for visualization."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), first_page=1, last_page=1)
            if images:
                return images[0]
            else:
                raise ValueError("Could not convert PDF to image")
        except ImportError:
            logger.error("pdf2image not available. Install with: pip install pdf2image")
            raise ImportError("pdf2image is required for PDF visualization. Install with: pip install pdf2image")
        except Exception as e:
            logger.error(f"Error converting PDF to image: {str(e)}")
            raise

    def _pdf_to_img(self, bbox_pdf: List[float], pdf_size: Tuple[float, float], img_size: Tuple[int, int]) -> List[float]:
        """
        Convert [x1, y1, x2, y2] from PDF space (origin = bottom-left, points)
        to image space (origin = top-left, pixels).

        Parameters
        ----------
        bbox_pdf : list[float]
            PDF coordinates (points).
        pdf_size : (width_pts, height_pts)
        img_size : (width_px,  height_px)

        Returns
        -------
        list[float]
            Image-space bbox [x1, y1, x2, y2] in pixels.
        """
        if not bbox_pdf or len(bbox_pdf) != 4:
            return [0.0, 0.0, 0.0, 0.0]

        pdf_w, pdf_h   = pdf_size
        img_w, img_h   = img_size
        sx, sy         = img_w / pdf_w, img_h / pdf_h

        x1, y1, x2, y2 = bbox_pdf
        return [
            x1 * sx,
            (pdf_h - y2) * sy,   # flip Y
            x2 * sx,
            (pdf_h - y1) * sy
        ]

    def _safe_bbox(self, bbox: List[float], img_size: Tuple[int, int]) -> List[float]:
        """
        Keep bbox inside image borders.

        Parameters
        ----------
        bbox : list[float]
            [x1, y1, x2, y2] in pixels.
        img_size : (width_px, height_px)

        Returns
        -------
        list[float]
            Clamped bbox.
        """
        if not bbox or len(bbox) != 4:
            return bbox

        x1, y1, x2, y2 = bbox
        img_w, img_h   = img_size

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return [x1, y1, x2, y2]

    def _transform_pdf_to_image_coords(self, bbox, pdf_size, image_size):
        """Legacy method - use _pdf_to_img instead."""
        return self._pdf_to_img(bbox, pdf_size, image_size)

    def _needs_coordinate_transformation(self, table_result, image_size):
        """Check if coordinates need transformation from PDF to image space."""
        if not hasattr(table_result, "tables") or not table_result.tables:
            return False

        # Check if any bbox coordinates are outside image bounds
        img_width, img_height = image_size
        for table in table_result.tables:
            if table.bbox and len(table.bbox) == 4:
                x1, y1, x2, y2 = table.bbox
                if x2 > img_width or y2 > img_height:
                    return True
            if hasattr(table, "cells") and table.cells:
                for cell in table.cells:
                    if cell.bbox and len(cell.bbox) == 4:
                        x1, y1, x2, y2 = cell.bbox
                        if x2 > img_width or y2 > img_height:
                            return True
        return False

    def visualize(self,
                  table_result: 'TableOutput',
                  image_path: Union[str, Path, Image.Image],
                  output_path: str = "visualized_tables.png",
                  table_color: str = 'red',
                  cell_color: str = 'blue',
                  box_width: int = 2,
                  show_text: bool = False,
                  text_color: str = 'green',
                  font_size: int = 12,
                  show_table_ids: bool = True) -> None:
        """Visualize table extraction results by drawing bounding boxes on the original image.

        This method allows users to easily see which extractor is working better
        by visualizing the detected tables and cells with bounding boxes.

        Args:
            table_result: TableOutput containing extracted tables
            image_path: Path to original image or PIL Image object
            output_path: Path to save the annotated image
            table_color: Color for table bounding boxes
            cell_color: Color for cell bounding boxes
            box_width: Width of bounding box lines
            show_text: Whether to overlay cell text
            text_color: Color for text overlay
            font_size: Font size for text overlay
            show_table_ids: Whether to show table IDs
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Handle different input types
            if isinstance(image_path, (str, Path)):
                image_path = Path(image_path)

                # Check if it's a PDF file
                if image_path.suffix.lower() == '.pdf':
                    # Convert PDF to image
                    image = self._convert_pdf_to_image(image_path)
                else:
                    # Regular image file
                    image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, Image.Image):
                image = image_path.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path)}")

            # Create a copy to draw on
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)

            # Just use original coordinates - no transformation needed

            # Try to load a font for text overlay
            font = None
            if show_text or show_table_ids:
                try:
                    # Try to use a better font if available
                    font = ImageFont.truetype("arial.ttf", font_size)
                except (OSError, IOError):
                    try:
                        # Fallback to default font
                        font = ImageFont.load_default()
                    except:
                        font = None

            # Draw tables and cells if table results exist
            if hasattr(table_result, "tables") and table_result.tables:
                for table_idx, table in enumerate(table_result.tables):
                    # Draw table bounding box
                    if table.bbox and len(table.bbox) == 4:
                        x1, y1, x2, y2 = table.bbox
                        draw.rectangle(
                            [(x1, y1), (x2, y2)],
                            outline=table_color,
                            width=box_width + 1
                        )

                        # Draw table ID (only if requested)
                        if show_table_ids and font:
                            table_id = getattr(table, 'table_id', f'Table {table_idx}')
                            draw.text((x1, y1 - font_size - 2), table_id,
                                    fill=table_color, font=font)

                    # Draw cell bounding boxes
                    if hasattr(table, "cells") and table.cells:
                        for cell in table.cells:
                            if cell.bbox and len(cell.bbox) == 4:
                                x1, y1, x2, y2 = cell.bbox

                                # Draw cell rectangle - no text overlay
                                draw.rectangle(
                                    [(x1, y1), (x2, y2)],
                                    outline=cell_color,
                                    width=box_width
                                )

            # Save the annotated image
            annotated_image.save(output_path)

            if self.show_log:
                logger.info(f"Table visualization saved to {output_path}")
                num_tables = len(table_result.tables) if table_result.tables else 0
                total_cells = sum(len(table.cells) for table in table_result.tables) if table_result.tables else 0
                logger.info(f"Visualized {num_tables} tables with {total_cells} cells")

        except Exception as e:
            error_msg = f"Error creating table visualization: {str(e)}"
            if self.show_log:
                logger.error(error_msg)
            raise RuntimeError(error_msg)

    def visualize_from_json(self,
                           image_path: Union[str, Path, Image.Image],
                           json_path: Union[str, Path],
                           output_path: str = "visualized_tables_from_json.png",
                           **kwargs) -> None:
        """
        Load table extraction results from JSON file and visualize them.

        Args:
            image_path: Path to original image, PDF file, or PIL Image object
            json_path: Path to JSON file containing table extraction results
            output_path: Path to save the annotated image
            **kwargs: Additional arguments passed to visualize method
        """
        import json

        try:
            # Load table results from JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct TableOutput from JSON data
            tables = []
            if isinstance(data, list):
                # Handle list of tables format
                for table_data in data:
                    cells = []
                    if 'cells' in table_data:
                        for cell_data in table_data['cells']:
                            cell = TableCell(**cell_data)
                            cells.append(cell)

                    table = Table(
                        cells=cells,
                        num_rows=table_data.get('num_rows', 0),
                        num_cols=table_data.get('num_cols', 0),
                        bbox=table_data.get('bbox'),
                        confidence=table_data.get('confidence'),
                        table_id=table_data.get('table_id', ''),
                        structure_confidence=table_data.get('structure_confidence')
                    )
                    tables.append(table)

            # Create TableOutput object
            table_result = TableOutput(
                tables=tables,
                source_img_size=data[0].get('source_img_size') if data else None,
                metadata=data[0].get('metadata', {}) if data else {}
            )

            # Visualize the loaded results
            self.visualize(table_result, image_path, output_path, **kwargs)

        except Exception as e:
            error_msg = f"Error loading and visualizing tables from JSON: {str(e)}"
            if self.show_log:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
