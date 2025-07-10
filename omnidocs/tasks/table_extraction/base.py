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
