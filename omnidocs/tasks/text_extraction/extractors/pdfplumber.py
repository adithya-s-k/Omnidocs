import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pdfplumber

from omnidocs.tasks.text_extraction.base import BaseTextExtractor, TextOutput, TextBlock, BaseTextMapper
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class PdfplumberTextMapper(BaseTextMapper):
    """Mapper for pdfplumber text extraction output."""
    
    def __init__(self):
        super().__init__("pdfplumber")
    
    def _setup_block_type_mapping(self):
        """Setup pdfplumber-specific block type mapping."""
        super()._setup_block_type_mapping()
        # pdfplumber doesn't provide block types, so we use generic mapping
        self._block_type_mapping.update({
            'text': 'paragraph',
            'line': 'paragraph',
            'char': 'paragraph'
        })

class PdfplumberTextExtractor(BaseTextExtractor):
    """Text extractor using pdfplumber."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 extract_images: bool = False,
                 extract_tables: bool = False,
                 use_layout: bool = True):
        """Initialize pdfplumber text extractor.
        
        Args:
            device: Device to run on (not used for pdfplumber)
            show_log: Whether to show detailed logs
            extract_images: Whether to extract images alongside text
            extract_tables: Whether to extract tables
            use_layout: Whether to use layout information for text extraction
        """
        super().__init__(device, show_log, "pdfplumber", extract_images)
        self.extract_tables = extract_tables
        self.use_layout = use_layout
        self._label_mapper = PdfplumberTextMapper()
        self._load_model()
    
    def _download_model(self) -> Path:
        """Download model - not applicable for pdfplumber."""
        return Path(".")
    
    def _load_model(self) -> None:
        """Load pdfplumber - no model loading required."""
        try:
            # pdfplumber doesn't require model loading
            self.model = pdfplumber
            
            if self.show_log:
                logger.info("pdfplumber loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load pdfplumber: {str(e)}")
            raise
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Path:
        """Preprocess input document."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() != '.pdf':
            raise ValueError(f"pdfplumber only supports PDF files. Got: {input_path.suffix}")
        
        return input_path
    
    def _extract_text_with_layout(self, page) -> List[TextBlock]:
        """Extract text with layout information."""
        text_blocks = []
        
        # Extract characters with position information
        chars = page.chars
        if not chars:
            return text_blocks
        
        # Group characters into words and lines
        words = page.extract_words()
        
        for i, word in enumerate(words):
            # Create bounding box from word coordinates
            bbox = [word['x0'], word['top'], word['x1'], word['bottom']]
            
            block = TextBlock(
                text=word['text'],
                bbox=bbox,
                confidence=1.0,  # pdfplumber doesn't provide confidence
                page_num=page.page_number,
                block_type='paragraph',
                font_info={
                    'font_name': word.get('fontname', ''),
                    'font_size': word.get('size', 0)
                },
                reading_order=i
            )
            text_blocks.append(block)
        
        return text_blocks
    
    def _extract_text_simple(self, page) -> List[TextBlock]:
        """Extract text without detailed layout information."""
        text_blocks = []
        
        # Extract text line by line
        text = page.extract_text()
        if not text:
            return text_blocks
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                block = TextBlock(
                    text=line.strip(),
                    bbox=None,  # No bbox information in simple mode
                    confidence=1.0,
                    page_num=page.page_number,
                    block_type='paragraph',
                    reading_order=i
                )
                text_blocks.append(block)
        
        return text_blocks
    
    def _extract_tables(self, page) -> List[TextBlock]:
        """Extract tables from page."""
        table_blocks = []
        
        tables = page.extract_tables()
        
        for table_idx, table in enumerate(tables):
            if table:
                # Convert table to text representation
                table_text = []
                for row in table:
                    if row:  # Skip empty rows
                        cleaned_row = [cell or '' for cell in row]
                        table_text.append(' | '.join(cleaned_row))
                
                if table_text:
                    block = TextBlock(
                        text='\n'.join(table_text),
                        bbox=None,  # pdfplumber doesn't provide table bounding boxes easily
                        confidence=1.0,
                        page_num=page.page_number,
                        block_type='table',
                        reading_order=1000 + table_idx  # Put tables after regular text
                    )
                    table_blocks.append(block)
        
        return table_blocks
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert pdfplumber output to standardized TextOutput format."""
        text_blocks = raw_output  # raw_output is already a list of TextBlocks
        
        # Sort blocks by page and reading order
        text_blocks.sort(key=lambda x: (x.page_num, x.reading_order or 0))
        
        # Combine all text
        full_text = '\n\n'.join(block.text for block in text_blocks)
        
        # Get metadata
        metadata = {
            'engine': 'pdfplumber',
            'extract_tables': self.extract_tables,
            'use_layout': self.use_layout,
            'total_blocks': len(text_blocks)
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
        """Extract text from PDF using pdfplumber.
        
        Args:
            input_path: Path to input PDF
            **kwargs: Additional parameters (ignored for pdfplumber)
            
        Returns:
            TextOutput containing extracted text
        """
        start_time = time.time()
        
        # Preprocess input
        input_path = self.preprocess_input(input_path)
        
        if self.show_log:
            logger.info(f"Extracting text from {input_path}")
        
        try:
            all_text_blocks = []
            
            with pdfplumber.open(input_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page in pdf.pages:
                    if self.use_layout:
                        page_blocks = self._extract_text_with_layout(page)
                    else:
                        page_blocks = self._extract_text_simple(page)
                    
                    all_text_blocks.extend(page_blocks)
                    
                    # Extract tables if requested
                    if self.extract_tables:
                        table_blocks = self._extract_tables(page)
                        all_text_blocks.extend(table_blocks)
            
            # Create source info
            source_info = {
                'file_path': str(input_path),
                'file_name': input_path.name,
                'file_size': input_path.stat().st_size,
                'engine': 'pdfplumber',
                'total_pages': total_pages
            }
            
            # Post-process output
            output = self.postprocess_output(all_text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks from {total_pages} pages in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['.pdf']