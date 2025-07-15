import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import cv2

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

logger = get_logger(__name__)

class CamelotMapper(BaseTableMapper):
    """Label mapper for Camelot table extraction output."""
    
    def __init__(self):
        super().__init__('camelot')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup extraction method mappings for Camelot."""
        self._methods = {
            'lattice': 'lattice',  # For tables with clear borders
            'stream': 'stream',    # For tables without clear borders
        }
        self._default_method = 'lattice'

class CamelotExtractor(BaseTableExtractor):
    """Camelot based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        method: str = 'lattice',
        pages: str = '1',
        flavor: str = 'lattice',
        **kwargs
    ):
        """Initialize Camelot Table Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='camelot'
        )
        
        self._label_mapper = CamelotMapper()
        self.method = method
        self.pages = pages
        self.flavor = flavor
        
        try:
            import camelot
            self.camelot = camelot
            
        except ImportError as e:
            logger.error("Failed to import Camelot")
            raise ImportError(
                "Camelot is not available. Please install it with: pip install camelot-py[cv]"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        Camelot doesn't require model download, it's rule-based.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("Camelot is rule-based and doesn't require model download")
        return None
    
    def _load_model(self) -> None:
        """Load Camelot (no actual model loading needed)."""
        try:
            if self.show_log:
                logger.info("Camelot extractor initialized")
                
        except Exception as e:
            logger.error("Failed to initialize Camelot extractor", exc_info=True)
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
    
    def postprocess_output(self, raw_output: Any, img_size: Tuple[int, int]) -> TableOutput:
        """Convert Camelot output to standardized TableOutput format."""
        tables = []
        
        for i, camelot_table in enumerate(raw_output):
            # Get table data
            df = camelot_table.df
            
            # Convert DataFrame to cells
            cells = []
            num_rows, num_cols = df.shape
            
            for row_idx in range(num_rows):
                for col_idx in range(num_cols):
                    cell_text = str(df.iloc[row_idx, col_idx]).strip()
                    
                    # Create cell with basic info
                    cell = TableCell(
                        text=cell_text,
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        confidence=camelot_table.accuracy / 100.0,  # Convert percentage to decimal
                        is_header=(row_idx == 0)  # Assume first row is header
                    )
                    cells.append(cell)
            
            # Get table bounding box if available
            bbox = None
            if hasattr(camelot_table, '_bbox'):
                bbox = list(camelot_table._bbox)
            
            # Create table object
            table = Table(
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                bbox=bbox,
                confidence=camelot_table.accuracy / 100.0,
                table_id=f"table_{i}",
                structure_confidence=camelot_table.accuracy / 100.0
            )
            
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'camelot',
                'method': self.method,
                'flavor': self.flavor
            }
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using Camelot."""
        try:
            # Camelot works with PDF files
            if isinstance(input_path, (str, Path)):
                pdf_path = Path(input_path)
                if pdf_path.suffix.lower() != '.pdf':
                    raise ValueError("Camelot only works with PDF files")
                
                # Extract tables from PDF
                tables = self.camelot.read_pdf(
                    str(pdf_path),
                    pages=self.pages,
                    flavor=self.flavor,
                    **kwargs
                )
                
                # Get image size (estimate from first page)
                try:
                    images = self._convert_pdf_to_image(pdf_path)
                    img_size = images[0].size if images else (612, 792)  # Default PDF size
                except:
                    img_size = (612, 792)  # Default PDF size
                
            else:
                raise ValueError("Camelot requires PDF file path, not image data")
            
            # Convert to standardized format
            result = self.postprocess_output(tables, img_size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using Camelot")
            
            return result
            
        except Exception as e:
            logger.error("Error during Camelot extraction", exc_info=True)
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
            logger.error("Error during Camelot prediction", exc_info=True)
            return []