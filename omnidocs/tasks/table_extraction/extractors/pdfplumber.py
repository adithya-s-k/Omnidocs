import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import cv2

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

logger = get_logger(__name__)

class PDFPlumberMapper(BaseTableMapper):
    """Label mapper for PDFPlumber table extraction output."""
    
    def __init__(self):
        super().__init__('pdfplumber')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup extraction settings for PDFPlumber."""
        self._table_settings = {
            'vertical_strategy': 'lines',
            'horizontal_strategy': 'lines',
            'explicit_vertical_lines': [],
            'explicit_horizontal_lines': [],
            'snap_tolerance': 3,
            'join_tolerance': 3,
            'edge_min_length': 3,
            'min_words_vertical': 3,
            'min_words_horizontal': 1,
            'intersection_tolerance': 3,
            'text_tolerance': 3,
            'text_x_tolerance': 3,
            'text_y_tolerance': 3,
        }

class PDFPlumberExtractor(BaseTableExtractor):
    """PDFPlumber based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        table_settings: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize PDFPlumber Table Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='pdfplumber'
        )
        
        self._label_mapper = PDFPlumberMapper()
        self.table_settings = table_settings or self._label_mapper._table_settings
        
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            
        except ImportError as e:
            logger.error("Failed to import PDFPlumber")
            raise ImportError(
                "PDFPlumber is not available. Please install it with: pip install pdfplumber"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        PDFPlumber doesn't require model download, it's rule-based.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("PDFPlumber is rule-based and doesn't require model download")
        return None
    
    def _load_model(self) -> None:
        """Load PDFPlumber (no actual model loading needed)."""
        try:
            if self.show_log:
                logger.info("PDFPlumber extractor initialized")
                
        except Exception as e:
            logger.error("Failed to initialize PDFPlumber extractor", exc_info=True)
            raise
    
    def _convert_pdf_to_image(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """Convert PDF pages to images for processing."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path))
            return images
        except ImportError:
            logger.error("pdf2image not available. Install with: pip install pdf2image")
            raise

    def _estimate_cell_bbox(self, table_bbox: List[float], row: int, col: int,
                           num_rows: int, num_cols: int) -> List[float]:
        """Estimate cell bounding box based on table bbox and grid position."""
        if not table_bbox or len(table_bbox) < 4:
            return [0.0, 0.0, 100.0, 100.0]  # Default bbox

        x1, y1, x2, y2 = table_bbox

        # Calculate cell dimensions
        cell_width = (x2 - x1) / num_cols
        cell_height = (y2 - y1) / num_rows

        # Calculate cell position
        cell_x1 = x1 + (col * cell_width)
        cell_y1 = y1 + (row * cell_height)
        cell_x2 = cell_x1 + cell_width
        cell_y2 = cell_y1 + cell_height

        return [cell_x1, cell_y1, cell_x2, cell_y2]

    def postprocess_output(self, raw_output: List[Dict], img_size: Tuple[int, int], pdf_size: Tuple[int, int] = None) -> TableOutput:
        """Convert PDFPlumber output to standardized TableOutput format."""
        tables = []
        
        for i, table_data in enumerate(raw_output):
            # Get table data
            table_cells = table_data.get('cells', [])
            bbox = table_data.get('bbox', None)

            # If no bbox available, estimate based on image size
            if bbox is None:
                bbox = [0, 0, img_size[0], img_size[1]]

            # Transform PDF coordinates to image coordinates if needed
            if pdf_size and bbox:
                bbox = self._transform_pdf_to_image_coords(bbox, pdf_size, img_size)

            # Convert to our cell format
            cells = []
            max_row = 0
            max_col = 0

            for cell_data in table_cells:
                text = cell_data.get('text', '').strip()
                row = cell_data.get('row', 0)
                col = cell_data.get('col', 0)

                max_row = max(max_row, row)
                max_col = max(max_col, col)

            # Calculate table dimensions
            num_rows = max_row + 1
            num_cols = max_col + 1

            # Create cells with estimated bboxes
            for cell_data in table_cells:
                text = cell_data.get('text', '').strip()
                row = cell_data.get('row', 0)
                col = cell_data.get('col', 0)

                # Use provided bbox or estimate one
                cell_bbox = cell_data.get('bbox', None)
                if cell_bbox is None:
                    cell_bbox = self._estimate_cell_bbox(
                        bbox, row, col, num_rows, num_cols
                    )

                # Transform cell coordinates if needed
                if pdf_size and cell_bbox:
                    cell_bbox = self._transform_pdf_to_image_coords(cell_bbox, pdf_size, img_size)

                # Create cell
                cell = TableCell(
                    text=text,
                    row=row,
                    col=col,
                    rowspan=cell_data.get('rowspan', 1),
                    colspan=cell_data.get('colspan', 1),
                    bbox=cell_bbox,
                    confidence=0.9,  # PDFPlumber is generally reliable
                    is_header=(row == 0)  # Assume first row is header
                )
                cells.append(cell)
            
            # Create table object
            table = Table(
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                bbox=bbox,
                confidence=0.9,
                table_id=f"table_{i}",
                structure_confidence=0.9
            )
            
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'pdfplumber',
                'table_settings': self.table_settings
            }
        )
    
    def _extract_tables_from_page(self, page) -> List[Dict]:
        """Extract tables from a single PDF page."""
        tables = []
        
        # Find tables on the page
        found_tables = page.find_tables(table_settings=self.table_settings)
        
        for table in found_tables:
            # Extract table data
            table_data = table.extract()
            
            if not table_data:
                continue
            
            # Convert to our format
            cells = []
            for row_idx, row_data in enumerate(table_data):
                for col_idx, cell_text in enumerate(row_data):
                    if cell_text is not None:
                        cells.append({
                            'text': str(cell_text).strip(),
                            'row': row_idx,
                            'col': col_idx,
                            'rowspan': 1,
                            'colspan': 1,
                            'bbox': None  # PDFPlumber doesn't provide cell-level bbox easily
                        })
            
            table_info = {
                'cells': cells,
                'bbox': table.bbox if hasattr(table, 'bbox') else None,
                'page_number': page.page_number
            }
            
            tables.append(table_info)
        
        return tables
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using PDFPlumber."""
        try:
            # PDFPlumber works with PDF files
            if isinstance(input_path, (str, Path)):
                pdf_path = Path(input_path)
                if pdf_path.suffix.lower() != '.pdf':
                    raise ValueError("PDFPlumber only works with PDF files")
                
                all_tables = []
                
                # Open PDF and extract tables from all pages
                with self.pdfplumber.open(str(pdf_path)) as pdf:
                    for page in pdf.pages:
                        page_tables = self._extract_tables_from_page(page)
                        all_tables.extend(page_tables)
                
                # Get image size and PDF size for coordinate transformation
                try:
                    # Get actual PDF page size first
                    import fitz
                    doc = fitz.open(str(pdf_path))
                    page = doc[0]
                    pdf_size = (page.rect.width, page.rect.height)
                    doc.close()

                    # Convert PDF to image to get actual image size
                    images = self._convert_pdf_to_image(pdf_path)
                    img_size = images[0].size if images else pdf_size
                except:
                    pdf_size = (612, 792)  # Default PDF size
                    img_size = (612, 792)  # Default image size

            else:
                raise ValueError("PDFPlumber requires PDF file path, not image data")

            # Convert to standardized format
            result = self.postprocess_output(all_tables, img_size, pdf_size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using PDFPlumber")
            
            return result
            
        except Exception as e:
            logger.error("Error during PDFPlumber extraction", exc_info=True)
            return TableOutput(
                tables=[],
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )
    
    def predict(self, pdf_path: Union[str, Path], **kwargs):
        """Predict method for compatibility with original interface."""
        try:
            result = self.extract(pdf_path, **kwargs)
            
            # Convert to original format
            table_res = []
            for table in result.tables:
                table_data = {
                    "table_id": table.table_id,
                    "bbox": table.bbox,
                    "confidence": table.confidence,
                    "cells": [cell.to_dict() for cell in table.cells],
                    "num_rows": table.num_rows,
                    "num_cols": table.num_cols
                }
                table_res.append(table_data)
            
            return table_res
            
        except Exception as e:
            logger.error("Error during PDFPlumber prediction", exc_info=True)
            return []