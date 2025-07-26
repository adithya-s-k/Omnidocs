import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import fitz  # PyMuPDF

from omnidocs.tasks.text_extraction.base import BaseTextExtractor, TextOutput, TextBlock, BaseTextMapper
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class PyMuPDFTextMapper(BaseTextMapper):
    """Mapper for PyMuPDF text extraction output."""
    
    def __init__(self):
        super().__init__("pymupdf")
    
    def _setup_block_type_mapping(self):
        """Setup PyMuPDF-specific block type mapping."""
        super()._setup_block_type_mapping()
        # PyMuPDF provides block types from layout analysis
        self._block_type_mapping.update({
            '0': 'paragraph',  # Text block
            '1': 'image',      # Image block
            'text': 'paragraph',
            'image': 'image',
            'line': 'paragraph',
            'span': 'paragraph'
        })

class PyMuPDFTextExtractor(BaseTextExtractor):
    """Text extractor using PyMuPDF (fitz)."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 extract_images: bool = False,
                 extract_tables: bool = False,
                 flags: int = 0,
                 clip: Optional[tuple] = None):
        """Initialize PyMuPDF text extractor.
        
        Args:
            device: Device to run on (not used for PyMuPDF)
            show_log: Whether to show detailed logs
            extract_images: Whether to extract images alongside text
            extract_tables: Whether to extract tables
            flags: Text extraction flags (fitz.TEXT_PRESERVE_LIGATURES, etc.)
            clip: Optional clipping rectangle (x0, y0, x1, y1)
        """
        super().__init__(device, show_log, "pymupdf", extract_images)
        self.extract_tables = extract_tables
        self.flags = flags
        self.clip = clip
        self._label_mapper = PyMuPDFTextMapper()
        self._load_model()
    
    def _download_model(self) -> Path:
        """Download model - not applicable for PyMuPDF."""
        return Path(".")
    
    def _load_model(self) -> None:
        """Load PyMuPDF - no model loading required."""
        try:
            # PyMuPDF doesn't require model loading
            self.model = fitz
            
            if self.show_log:
                logger.info("PyMuPDF loaded successfully")
                logger.info(f"PyMuPDF version: {fitz.version}")
                
        except Exception as e:
            logger.error(f"Failed to load PyMuPDF: {str(e)}")
            raise
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Path:
        """Preprocess input document."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        supported_formats = ['.pdf', '.xps', '.oxps', '.epub', '.mobi', '.fb2', '.cbz', '.svg']
        if input_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported format: {input_path.suffix}. Supported: {supported_formats}")
        
        return input_path
    
    def _extract_text_blocks(self, page) -> List[TextBlock]:
        """Extract text blocks with layout information."""
        text_blocks = []
        
        # Get text blocks with layout information
        blocks = page.get_text("dict", flags=self.flags, clip=self.clip)
        
        for block_idx, block in enumerate(blocks.get("blocks", [])):
            block_type = str(block.get("type", 0))
            
            # Skip image blocks unless extracting images
            if block_type == "1" and not self.extract_images:
                continue
            
            # Handle text blocks
            if block_type == "0":  # Text block
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Process lines within the block
                for line_idx, line in enumerate(block.get("lines", [])):
                    line_text = ""
                    line_bbox = line.get("bbox", block_bbox)
                    font_info = {}
                    
                    # Process spans within the line
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        line_text += span_text
                        
                        # Get font information from first span
                        if not font_info:
                            font_info = {
                                'font_name': span.get("font", ""),
                                'font_size': span.get("size", 0),
                                'bold': "Bold" in span.get("font", ""),
                                'italic': "Italic" in span.get("font", ""),
                                'color': span.get("color", 0)
                            }
                    
                    if line_text.strip():
                        text_block = TextBlock(
                            text=line_text.strip(),
                            bbox=line_bbox,
                            confidence=1.0,
                            page_num=page.number + 1,  # Convert to 1-based
                            block_type=self._label_mapper.normalize_block_type("text"),
                            font_info=font_info,
                            reading_order=block_idx * 100 + line_idx
                        )
                        text_blocks.append(text_block)
            
            # Handle image blocks
            elif block_type == "1" and self.extract_images:
                image_bbox = block.get("bbox", [0, 0, 0, 0])
                
                image_block = TextBlock(
                    text=f"[IMAGE: {block.get('width', 0)}x{block.get('height', 0)}]",
                    bbox=image_bbox,
                    confidence=1.0,
                    page_num=page.number + 1,
                    block_type="image",
                    reading_order=block_idx * 100
                )
                text_blocks.append(image_block)
        
        return text_blocks
    
    def _extract_text_simple(self, page) -> List[TextBlock]:
        """Extract text without detailed layout information."""
        text_blocks = []
        
        # Extract simple text
        text = page.get_text(flags=self.flags, clip=self.clip)
        
        if text.strip():
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    text_block = TextBlock(
                        text=paragraph.strip(),
                        bbox=None,
                        confidence=1.0,
                        page_num=page.number + 1,
                        block_type='paragraph',
                        reading_order=para_idx
                    )
                    text_blocks.append(text_block)
        
        return text_blocks
    
    def _extract_tables(self, page) -> List[TextBlock]:
        """Extract tables from page."""
        table_blocks = []
        
        try:
            # Find tables using PyMuPDF's table detection
            tables = page.find_tables()
            
            for table_idx, table in enumerate(tables):
                # Extract table data
                table_data = table.extract()
                
                if table_data:
                    # Convert table to text representation
                    table_text = []
                    for row in table_data:
                        if row:  # Skip empty rows
                            cleaned_row = [str(cell or '') for cell in row]
                            table_text.append(' | '.join(cleaned_row))
                    
                    if table_text:
                        # Get table bounding box
                        table_bbox = table.bbox
                        
                        table_block = TextBlock(
                            text='\n'.join(table_text),
                            bbox=table_bbox,
                            confidence=1.0,
                            page_num=page.number + 1,
                            block_type='table',
                            reading_order=2000 + table_idx  # Put tables after regular text
                        )
                        table_blocks.append(table_block)
                        
        except Exception as e:
            if self.show_log:
                logger.warning(f"Error extracting tables from page {page.number + 1}: {str(e)}")
        
        return table_blocks
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert PyMuPDF output to standardized TextOutput format."""
        text_blocks = raw_output  # raw_output is already a list of TextBlocks
        
        # Sort blocks by page and reading order
        text_blocks.sort(key=lambda x: (x.page_num, x.reading_order or 0))
        
        # Combine all text
        full_text = '\n\n'.join(block.text for block in text_blocks if block.text.strip())
        
        # Get metadata
        metadata = {
            'engine': 'pymupdf',
            'extract_tables': self.extract_tables,
            'flags': self.flags,
            'clip': self.clip,
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
        use_layout: bool = True,
        **kwargs
    ) -> TextOutput:
        """Extract text from document using PyMuPDF.
        
        Args:
            input_path: Path to input document
            use_layout: Whether to use layout information for extraction
            **kwargs: Additional parameters
            
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
            
            # Open document
            doc = fitz.open(str(input_path))
            
            try:
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    page = doc[page_num]
                    
                    # Extract text blocks
                    if use_layout:
                        page_blocks = self._extract_text_blocks(page)
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
                    'engine': 'pymupdf',
                    'total_pages': total_pages,
                    'metadata': doc.metadata
                }
                
            finally:
                doc.close()
            
            # Post-process output
            output = self.postprocess_output(all_text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks from {total_pages} pages in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def extract_from_pages(
        self,
        input_path: Union[str, Path],
        page_range: Optional[tuple] = None,
        use_layout: bool = True,
        **kwargs
    ) -> TextOutput:
        """Extract text from specific pages.
        
        Args:
            input_path: Path to input document
            page_range: Optional tuple of (start_page, end_page) (1-based, inclusive)
            use_layout: Whether to use layout information
            **kwargs: Additional parameters
            
        Returns:
            TextOutput containing extracted text from specified pages
        """
        start_time = time.time()
        
        # Preprocess input
        input_path = self.preprocess_input(input_path)
        
        if self.show_log:
            logger.info(f"Extracting text from {input_path}, pages {page_range}")
        
        try:
            all_text_blocks = []
            
            # Open document
            doc = fitz.open(str(input_path))
            
            try:
                total_pages = len(doc)
                
                if page_range is None:
                    start_page, end_page = 1, total_pages
                else:
                    start_page, end_page = page_range
                
                # Convert to 0-based indexing
                start_idx = max(0, start_page - 1)
                end_idx = min(total_pages - 1, end_page - 1)
                
                for page_num in range(start_idx, end_idx + 1):
                    page = doc[page_num]
                    
                    # Extract text blocks
                    if use_layout:
                        page_blocks = self._extract_text_blocks(page)
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
                    'engine': 'pymupdf',
                    'total_pages': total_pages,
                    'page_range': page_range,
                    'metadata': doc.metadata
                }
                
            finally:
                doc.close()
            
            # Post-process output
            output = self.postprocess_output(all_text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['.pdf', '.xps', '.oxps', '.epub', '.mobi', '.fb2', '.cbz', '.svg']