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
        """Estimate cell bounding box based on table bbox and grid position.

        TODO: Cell bbox estimation is still broken for PDF extractors.
        Current issues:
        - Grid-based estimation doesn't match actual cell positions
        - Coordinate transformation from PDF to image space is inaccurate
        - Need better cell detection or coordinate mapping
        """
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

    def postprocess_output(
        self,
        raw_output: List[Dict],
        img_size: Tuple[int, int],
        pdf_size: Tuple[int, int] = None,
    ) -> TableOutput:
        """Convert PDFPlumber output to standardized TableOutput format."""
        tables: List[Table] = []

        for i, table_data in enumerate(raw_output):
            table_bbox = table_data.get("bbox")
            if table_bbox is None:
                table_bbox = [0, 0, img_size[0], img_size[1]]

            if pdf_size:
                table_bbox_img = self._transform_pdf_to_image_coords(
                    table_bbox, pdf_size, img_size
                )
            else:
                table_bbox_img = table_bbox

            # Get max row/col indexes to know dimensions
            max_row = max(c["row"] for c in table_data["cells"])
            max_col = max(c["col"] for c in table_data["cells"])
            num_rows = max_row + 1
            num_cols = max_col + 1

            # Pre-compute equally spaced cell rectangles inside the table bbox
            x0, y0, x1, y1 = table_bbox_img
            cell_w = (x1 - x0) / num_cols
            cell_h = (y1 - y0) / num_rows

            cells: List[TableCell] = []
            for c in table_data["cells"]:
                r, cidx = c["row"], c["col"]

                # exact rectangle in image space
                cx0 = x0 + cidx * cell_w
                cy0 = y0 + r * cell_h
                cx1 = cx0 + cell_w
                cy1 = cy0 + cell_h
                cell_bbox_img = [cx0, cy0, cx1, cy1]

                cells.append(
                    TableCell(
                        text=c["text"].strip(),
                        row=r,
                        col=cidx,
                        rowspan=c.get("rowspan", 1),
                        colspan=c.get("colspan", 1),
                        bbox=cell_bbox_img,
                        confidence=0.9,
                        is_header=(r == 0),
                    )
                )

            tables.append(
                Table(
                    cells=cells,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    bbox=table_bbox_img,
                    confidence=0.9,
                    table_id=f"table_{i}",
                    structure_confidence=0.9,
                )
            )

        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={"engine": "pdfplumber", "table_settings": self.table_settings},
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