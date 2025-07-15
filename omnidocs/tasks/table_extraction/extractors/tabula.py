import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import cv2
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

logger = get_logger(__name__)

class TabulaMapper(BaseTableMapper):
    """Label mapper for Tabula table extraction output."""
    
    def __init__(self):
        super().__init__('tabula')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup extraction method mappings for Tabula."""
        self._methods = {
            'lattice': 'lattice',  # For tables with visible grid lines
            'stream': 'stream',    # For tables without visible grid lines
            'guess': 'guess'       # Auto-detect method
        }
        self._default_method = 'lattice'
        
        # Tabula-specific options
        self._tabula_options = {
            'multiple_tables': True,
            'pandas_options': {'header': 'infer'},
            'stream': False,
            'guess': True,
            'area': None,
            'columns': None,
            'format': 'dataframe',
            'java_options': [],
            'lattice': False,
            'silent': False,
            'pages': None,
            'password': None,
            'encoding': 'utf-8',
            'output_format': 'dataframe'
        }

class TabulaExtractor(BaseTableExtractor):
    """Tabula based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        method: str = 'lattice',
        pages: Optional[Union[str, List[int]]] = None,
        multiple_tables: bool = True,
        guess: bool = True,
        area: Optional[List[float]] = None,
        columns: Optional[List[float]] = None,
        **kwargs
    ):
        """Initialize Tabula Table Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='tabula'
        )
        
        self._label_mapper = TabulaMapper()
        self.method = method
        self.pages = pages or 'all'
        self.multiple_tables = multiple_tables
        self.guess = guess
        self.area = area
        self.columns = columns
        
        try:
            import tabula
            self.tabula = tabula
            
        except ImportError as e:
            logger.error("Failed to import Tabula")
            raise ImportError(
                "Tabula is not available. Please install it with: pip install tabula-py"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        Tabula doesn't require model download, it uses Java-based extraction.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("Tabula is Java-based and doesn't require model download")
        return None
    
    def _load_model(self) -> None:
        """Load Tabula (no actual model loading needed)."""
        try:
            # Test if tabula-py is working
            import subprocess
            import sys
            
            # Check if Java is available
            try:
                subprocess.run(['java', '-version'], capture_output=True, check=True)
                if self.show_log:
                    logger.info("Tabula extractor initialized successfully")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Java not found. Tabula requires Java to be installed.")
                
        except Exception as e:
            logger.error("Failed to initialize Tabula extractor", exc_info=True)
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
    
    def _prepare_tabula_options(self, **kwargs) -> Dict[str, Any]:
        """Prepare options for Tabula extraction."""
        options = {
            'pages': self.pages,
            'multiple_tables': self.multiple_tables,
            'guess': self.guess,
            'lattice': self.method == 'lattice',
            'stream': self.method == 'stream',
            'silent': not self.show_log,
            'pandas_options': {'header': None},  # We'll handle headers ourselves
        }
        
        # Add area and columns if specified
        if self.area:
            options['area'] = self.area
        if self.columns:
            options['columns'] = self.columns
        
        # Update with any additional kwargs
        options.update(kwargs)
        
        return options
    
    def _dataframe_to_cells(self, df, table_idx: int = 0) -> List[TableCell]:
        """Convert pandas DataFrame to TableCell objects."""
        cells = []
        
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                # Clean cell text
                cell_text = str(value).strip() if value is not None else ""
                if cell_text in ['nan', 'NaN', 'None']:
                    cell_text = ""
                
                # Determine if cell is header (first row heuristic)
                is_header = row_idx == 0 and cell_text != ""
                
                cell = TableCell(
                    text=cell_text,
                    row=row_idx,
                    col=col_idx,
                    rowspan=1,
                    colspan=1,
                    bbox=None,  # Tabula doesn't provide bbox info
                    confidence=None,  # Tabula doesn't provide confidence scores
                    is_header=is_header
                )
                cells.append(cell)
        
        return cells
    
    def _estimate_table_bbox(self, df, img_size: Tuple[int, int]) -> Optional[List[float]]:
        """Estimate table bounding box based on DataFrame size."""
        if df.empty:
            return None
        
        # Simple estimation - in practice, you'd need more sophisticated methods
        # This is just a placeholder
        width, height = img_size
        return [
            width * 0.1,   # x1
            height * 0.1,  # y1
            width * 0.9,   # x2
            height * 0.9   # y2
        ]
    
    def postprocess_output(self, raw_output: List, img_size: Tuple[int, int]) -> TableOutput:
        """Convert Tabula output to standardized TableOutput format."""
        tables = []
        
        for i, df in enumerate(raw_output):
            if df.empty:
                continue
            
            # Convert DataFrame to cells
            cells = self._dataframe_to_cells(df, i)
            
            # Get table dimensions
            num_rows, num_cols = df.shape
            
            # Estimate table bbox
            bbox = self._estimate_table_bbox(df, img_size)
            
            # Create table object
            table = Table(
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                bbox=bbox,
                confidence=None,  # Tabula doesn't provide confidence
                table_id=f"table_{i}",
                structure_confidence=None
            )
            
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'tabula',
                'method': self.method,
                'pages': self.pages,
                'multiple_tables': self.multiple_tables,
                'guess': self.guess
            }
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using Tabula."""
        try:
            # Tabula works with PDF files
            if isinstance(input_path, (str, Path)):
                pdf_path = Path(input_path)
                if pdf_path.suffix.lower() != '.pdf':
                    raise ValueError("Tabula only works with PDF files")
                
                # Prepare extraction options
                options = self._prepare_tabula_options(**kwargs)
                
                # Extract tables from PDF
                try:
                    tables_list = self.tabula.read_pdf(str(pdf_path), **options)
                    
                    # Ensure we have a list of DataFrames
                    if not isinstance(tables_list, list):
                        tables_list = [tables_list]
                    
                except Exception as e:
                    if self.show_log:
                        logger.error(f"Tabula extraction failed: {str(e)}")
                    tables_list = []
                
                # Get image size (estimate from first page)
                try:
                    images = self._convert_pdf_to_image(pdf_path)
                    img_size = images[0].size if images else (612, 792)  # Default PDF size
                except:
                    img_size = (612, 792)  # Default PDF size
                
            else:
                raise ValueError("Tabula requires PDF file path, not image data")
            
            # Convert to standardized format
            result = self.postprocess_output(tables_list, img_size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using Tabula")
            
            return result
            
        except Exception as e:
            logger.error("Error during Tabula extraction", exc_info=True)
            return TableOutput(
                tables=[],
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )
    
    def extract_with_area(
        self,
        input_path: Union[str, Path],
        area: List[float],
        **kwargs
    ) -> TableOutput:
        """Extract tables from specific area of PDF."""
        original_area = self.area
        self.area = area
        
        try:
            result = self.extract(input_path, **kwargs)
            return result
        finally:
            self.area = original_area
    
    def extract_with_columns(
        self,
        input_path: Union[str, Path],
        columns: List[float],
        **kwargs
    ) -> TableOutput:
        """Extract tables with specified column positions."""
        original_columns = self.columns
        self.columns = columns
        
        try:
            result = self.extract(input_path, **kwargs)
            return result
        finally:
            self.columns = original_columns
    
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
            logger.error("Error during Tabula prediction", exc_info=True)
            return []