import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import cv2

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

logger = get_logger(__name__)

class PPStructureMapper(BaseTableMapper):
    """Label mapper for PaddleOCR PPStructure table extraction output."""
    
    def __init__(self):
        super().__init__('ppstructure')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup language and model mappings for PPStructure."""
        self._language_mapping = {
            'en': 'en',
            'ch': 'ch',
            'chinese_cht': 'chinese_cht',
            'fr': 'fr',
            'german': 'german',
            'japan': 'japan',
            'korean': 'korean',
        }
        
        self._layout_models = {
            'en': 'picodet_lcnet_x1_0_fgd_layout_cdla_infer',
            'ch': 'picodet_lcnet_x1_0_fgd_layout_infer'
        }
        
        self._table_models = {
            'en': 'SLANet',
            'ch': 'SLANet_ch'
        }

class PPStructureExtractor(BaseTableExtractor):
    """PaddleOCR PPStructure based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        layout_model: Optional[str] = None,
        table_model: Optional[str] = None,
        return_ocr_result_in_table: bool = True,
        **kwargs
    ):
        """Initialize PPStructure Table Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='ppstructure'
        )
        
        self._label_mapper = PPStructureMapper()
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.layout_model = layout_model
        self.table_model = table_model
        self.return_ocr_result_in_table = return_ocr_result_in_table
        
        try:
            from paddleocr import PPStructure
            self.PPStructure = PPStructure
            
        except ImportError as e:
            logger.error("Failed to import PPStructure")
            raise ImportError(
                "PPStructure is not available. Please install it with: pip install paddlepaddle paddleocr"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        Model download is handled automatically by PPStructure library.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("Model downloading is handled automatically by PPStructure library")
        return None
    
    def _load_model(self) -> None:
        """Load PPStructure models."""
        try:
            # Get the primary language
            primary_lang = self.languages[0] if self.languages else 'en'
            mapped_lang = self._label_mapper._language_mapping.get(primary_lang, 'en')
            
            # Set default models if not specified
            if not self.layout_model:
                self.layout_model = self._label_mapper._layout_models.get(mapped_lang, 
                    self._label_mapper._layout_models['en'])
            
            if not self.table_model:
                self.table_model = self._label_mapper._table_models.get(mapped_lang,
                    self._label_mapper._table_models['en'])
            
            # Initialize PPStructure
            self.pp_structure = self.PPStructure(
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                lang=mapped_lang,
                layout_model_dir=self.layout_model,
                table_model_dir=self.table_model,
                return_ocr_result_in_table=self.return_ocr_result_in_table
            )
            
            if self.show_log:
                logger.info(f"PPStructure models loaded with language: {mapped_lang}")
                
        except Exception as e:
            logger.error("Failed to load PPStructure models", exc_info=True)
            raise
    
    def preprocess_input(self, input_path: Union[str, Path, Image.Image]) -> List[Image.Image]:
        """
        Convert input to processable format.
        PPStructure can handle both images and PDFs.

        Args:
            input_path: Path to input image or PDF, or PIL Image

        Returns:
            List of PIL Images
        """
        if isinstance(input_path, Image.Image):
            return [input_path.convert('RGB')]

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() == '.pdf':
            # Convert PDF to images using PyMuPDF (fitz)
            try:
                import fitz  # PyMuPDF
                images = []
                pdf_doc = fitz.open(str(input_path))

                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc[page_num]
                    # Render page to image with 2x scaling for better quality
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("ppm")

                    # Convert to PIL Image
                    from io import BytesIO
                    img = Image.open(BytesIO(img_data)).convert('RGB')
                    images.append(img)

                pdf_doc.close()
                return images

            except ImportError:
                logger.error("PyMuPDF not available. Install with: pip install PyMuPDF")
                raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        else:
            # Load single image
            image = Image.open(input_path).convert('RGB')
            return [image]

    def postprocess_output(self, raw_output: List[Dict], img_size: Tuple[int, int]) -> TableOutput:
        """Convert PPStructure output to standardized TableOutput format."""
        tables = []
        
        for i, result in enumerate(raw_output):
            if result['type'] != 'table':
                continue
                
            # Get table structure result
            table_res = result.get('res', {})
            
            if not table_res:
                continue
            
            # Extract table structure
            structure_str_list = table_res.get('structure_str_list', [])
            bbox_list = table_res.get('bbox_list', [])
            
            if not structure_str_list or not bbox_list:
                continue
            
            # Parse table structure
            cells = []
            num_rows = 0
            num_cols = 0
            
            for j, (structure, bbox) in enumerate(zip(structure_str_list, bbox_list)):
                # Parse structure info (simplified parsing)
                text = structure.get('text', '').strip()
                
                # Extract position info from structure
                row = structure.get('row', 0)
                col = structure.get('col', 0)
                rowspan = structure.get('rowspan', 1)
                colspan = structure.get('colspan', 1)
                
                num_rows = max(num_rows, row + rowspan)
                num_cols = max(num_cols, col + colspan)
                
                # Normalize bbox
                normalized_bbox = self._label_mapper.normalize_bbox(bbox, img_size[0], img_size[1])
                
                # Create cell
                cell = TableCell(
                    text=text,
                    row=row,
                    col=col,
                    rowspan=rowspan,
                    colspan=colspan,
                    bbox=normalized_bbox,
                    confidence=structure.get('confidence', 0.9),
                    is_header=(row == 0)  # Assume first row is header
                )
                cells.append(cell)
            
            # Get table bounding box
            table_bbox = result.get('bbox', None)
            if table_bbox:
                table_bbox = self._label_mapper.normalize_bbox(table_bbox, img_size[0], img_size[1])
            
            # Create table object
            table = Table(
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                bbox=table_bbox,
                confidence=result.get('confidence', 0.9),
                table_id=f"table_{i}",
                structure_confidence=result.get('confidence', 0.9)
            )
            
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'ppstructure',
                'language': self.languages[0] if self.languages else 'en',
                'layout_model': self.layout_model,
                'table_model': self.table_model
            }
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using PPStructure."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)

            all_tables = []
            total_img_size = None

            # Process each page/image
            for page_idx, img in enumerate(images):
                # Convert PIL to cv2 format
                if isinstance(img, Image.Image):
                    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                else:
                    img_cv2 = img

                # Get image size
                img_size = img_cv2.shape[:2][::-1]  # (width, height)
                if total_img_size is None:
                    total_img_size = img_size

                # Perform structure analysis
                result = self.pp_structure(img_cv2)

                # Convert to standardized format
                page_output = self.postprocess_output(result, img_size)

                # Add page information to tables
                for table in page_output.tables:
                    table.metadata = table.metadata or {}
                    table.metadata['page_number'] = page_idx + 1
                    all_tables.append(table)

            # Create final output
            table_output = TableOutput(
                tables=all_tables,
                source_img_size=total_img_size,
                processing_time=None,
                metadata={"total_pages": len(images)}
            )

            if self.show_log:
                logger.info(f"Extracted {len(table_output.tables)} tables using PPStructure from {len(images)} page(s)")

            return table_output

        except Exception as e:
            logger.error("Error during PPStructure extraction", exc_info=True)
            return TableOutput(
                tables=[],
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )
    
    def predict(self, img, **kwargs):
        """Predict method for compatibility with original interface."""
        try:
            result = self.extract(img, **kwargs)
            
            # Convert to original format
            table_res = []
            for table in result.tables:
                table_data = {
                    "table_id": table.table_id,
                    "bbox": table.bbox,
                    "confidence": table.confidence,
                    "cells": [cell.to_dict() for cell in table.cells],
                    "num_rows": table.num_rows,
                    "num_cols": table.num_cols,
                    "type": "table"
                }
                table_res.append(table_data)
            
            return table_res
            
        except Exception as e:
            logger.error("Error during PPStructure prediction", exc_info=True)
            return []