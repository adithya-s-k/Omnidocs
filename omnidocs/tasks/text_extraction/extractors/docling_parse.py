import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import docling
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from omnidocs.tasks.text_extraction.base import BaseTextExtractor, TextOutput, TextBlock, BaseTextMapper
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class DoclingTextMapper(BaseTextMapper):
    """Mapper for Docling text extraction output."""
    
    def __init__(self):
        super().__init__("docling")
    
    def _setup_block_type_mapping(self):
        """Setup Docling-specific block type mapping."""
        super()._setup_block_type_mapping()
        # Add Docling-specific mappings
        self._block_type_mapping.update({
            'document-header': 'heading',
            'section-header': 'heading',
            'title': 'heading',
            'subtitle': 'subheading',
            'text': 'paragraph',
            'list-item': 'list',
            'table': 'table',
            'figure': 'figure',
            'caption': 'caption',
            'footnote': 'footnote',
            'page-footer': 'footer',
            'page-header': 'header'
        })

class DoclingTextExtractor(BaseTextExtractor):
    """Text extractor using Docling."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 extract_images: bool = False,
                 ocr_enabled: bool = True,
                 table_structure_enabled: bool = True):
        """Initialize Docling text extractor.
        
        Args:
            device: Device to run on (not used for Docling)
            show_log: Whether to show detailed logs
            extract_images: Whether to extract images alongside text
            ocr_enabled: Whether to enable OCR for scanned documents
            table_structure_enabled: Whether to enable table structure detection
        """
        super().__init__(device, show_log, "docling", extract_images)
        self.ocr_enabled = ocr_enabled
        self.table_structure_enabled = table_structure_enabled
        self._label_mapper = DoclingTextMapper()
        self._load_model()
    
    def _download_model(self) -> Path:
        """Download model - not applicable for Docling."""
        return Path(".")
    
    def _load_model(self) -> None:
        """Load Docling document converter."""
        try:
            # Initialize document converter with default settings
            self.model = DocumentConverter()
            
            if self.show_log:
                logger.info("Docling DocumentConverter loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load Docling: {str(e)}")
            raise
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Path:
        """Preprocess input document."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        supported_formats = ['.pdf', '.docx', '.pptx', '.html', '.md']
        if input_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported format: {input_path.suffix}. Supported: {supported_formats}")
        
        return input_path
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert Docling output to standardized TextOutput format."""
        text_blocks = []
        
        # Process document elements
        for element in raw_output.document.texts:
            # Get bounding box if available
            bbox = None
            if hasattr(element, 'prov') and element.prov:
                for prov in element.prov:
                    if hasattr(prov, 'bbox'):
                        bbox = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]
                        break
            
            # Get page number
            page_num = 1
            if hasattr(element, 'prov') and element.prov:
                for prov in element.prov:
                    if hasattr(prov, 'page'):
                        page_num = prov.page + 1  # Convert to 1-based
                        break
            
            # Create text block
            block = TextBlock(
                text=element.text,
                bbox=bbox,
                confidence=1.0,  # Docling doesn't provide confidence scores
                page_num=page_num,
                block_type=self._label_mapper.normalize_block_type(element.label),
                reading_order=getattr(element, 'reading_order', None)
            )
            text_blocks.append(block)
        
        # Sort blocks by reading order
        text_blocks.sort(key=lambda x: (x.page_num, x.reading_order or 0))
        
        # Combine all text
        full_text = '\n\n'.join(block.text for block in text_blocks)
        
        # Get metadata
        metadata = {
            'engine': 'docling',
            'ocr_enabled': self.ocr_enabled,
            'table_structure_enabled': self.table_structure_enabled,
            'total_elements': len(text_blocks)
        }
        
        return TextOutput(
            text_blocks=text_blocks,
            full_text=full_text,
            metadata=metadata,
            source_info=source_info,
            page_count=max(block.page_num for block in text_blocks) if text_blocks else 1
        )
    
    def extract(
        self,
        input_path: Union[str, Path],
        **kwargs
    ) -> TextOutput:
        """Extract text from document using Docling.
        
        Args:
            input_path: Path to input document
            **kwargs: Additional parameters (ignored for Docling)
            
        Returns:
            TextOutput containing extracted text
        """
        start_time = time.time()
        
        # Preprocess input
        input_path = self.preprocess_input(input_path)
        
        if self.show_log:
            logger.info(f"Extracting text from {input_path}")
        
        try:
            # Convert document
            result = self.model.convert(input_path)
            
            # Create source info
            source_info = {
                'file_path': str(input_path),
                'file_name': input_path.name,
                'file_size': input_path.stat().st_size,
                'engine': 'docling'
            }
            
            # Post-process output
            output = self.postprocess_output(result, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['.pdf', '.docx', '.pptx', '.html', '.md']