import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfFileReader as PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is not installed. Please install it with: pip install PyPDF2")

from omnidocs.tasks.text_extraction.base import BaseTextExtractor, TextOutput, TextBlock, BaseTextMapper
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class PyPDF2TextMapper(BaseTextMapper):
    """Mapper for PyPDF2 text extraction output."""
    
    def __init__(self):
        super().__init__("pypdf2")
    
    def _setup_block_type_mapping(self):
        """Setup PyPDF2-specific block type mapping."""
        super()._setup_block_type_mapping()
        # PyPDF2 provides minimal structure, mostly raw text
        self._block_type_mapping.update({
            'text': 'paragraph',
            'page': 'paragraph',
            'line': 'paragraph'
        })

class PyPDF2TextExtractor(BaseTextExtractor):
    """Text extractor using PyPDF2."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 extract_images: bool = False,
                 ignore_images: bool = True,
                 extract_forms: bool = False):
        """Initialize PyPDF2 text extractor.
        
        Args:
            device: Device to run on (not used for PyPDF2)
            show_log: Whether to show detailed logs
            extract_images: Whether to extract images alongside text
            ignore_images: Whether to ignore images during text extraction
            extract_forms: Whether to extract form fields
        """
        super().__init__(device, show_log, "pypdf2", extract_images)
        self.ignore_images = ignore_images
        self.extract_forms = extract_forms
        self._label_mapper = PyPDF2TextMapper()
        self._load_model()
    
    def _download_model(self) -> Path:
        """Download model - not applicable for PyPDF2."""
        return Path(".")
    
    def _load_model(self) -> None:
        """Load PyPDF2 - no model loading required."""
        try:
            # PyPDF2 doesn't require model loading
            self.model = PdfReader
            
            if self.show_log:
                logger.info("PyPDF2 loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load PyPDF2: {str(e)}")
            raise
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Path:
        """Preprocess input document."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() != '.pdf':
            raise ValueError(f"PyPDF2 only supports PDF files. Got: {input_path.suffix}")
        
        return input_path
    
    def _extract_page_text(self, page, page_num: int) -> List[TextBlock]:
        """Extract text from a single page."""
        text_blocks = []
        
        try:
            # Extract text from page
            page_text = page.extract_text()
            
            if page_text and page_text.strip():
                # Split into paragraphs (simple approach)
                paragraphs = page_text.split('\n\n')
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        # Clean up the text
                        clean_text = paragraph.strip()
                        clean_text = ' '.join(clean_text.split())  # Normalize whitespace
                        
                        block = TextBlock(
                            text=clean_text,
                            bbox=None,  # PyPDF2 doesn't provide bounding boxes
                            confidence=1.0,
                            page_num=page_num,
                            block_type='paragraph',
                            reading_order=para_idx
                        )
                        text_blocks.append(block)
            
            else:
                # If no text extracted, create empty block
                block = TextBlock(
                    text="",
                    bbox=None,
                    confidence=0.0,
                    page_num=page_num,
                    block_type='paragraph',
                    reading_order=0
                )
                text_blocks.append(block)
                
        except Exception as e:
            if self.show_log:
                logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
            
            # Create error block
            block = TextBlock(
                text=f"[ERROR: Could not extract text from page {page_num}]",
                bbox=None,
                confidence=0.0,
                page_num=page_num,
                block_type='paragraph',
                reading_order=0
            )
            text_blocks.append(block)
        
        return text_blocks
    
    def _extract_form_fields(self, reader) -> List[TextBlock]:
        """Extract form fields from PDF."""
        form_blocks = []
        
        try:
            # Check if PDF has form fields
            if hasattr(reader, 'get_form_text_fields'):
                form_fields = reader.get_form_text_fields()
                
                if form_fields:
                    for field_name, field_value in form_fields.items():
                        if field_value:
                            block = TextBlock(
                                text=f"{field_name}: {field_value}",
                                bbox=None,
                                confidence=1.0,
                                page_num=1,  # Forms don't have specific page info
                                block_type='form_field',
                                reading_order=9999  # Put forms at the end
                            )
                            form_blocks.append(block)
                            
        except Exception as e:
            if self.show_log:
                logger.warning(f"Error extracting form fields: {str(e)}")
        
        return form_blocks
    
    def _get_pdf_metadata(self, reader) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}
        
        try:
            if hasattr(reader, 'metadata') and reader.metadata:
                pdf_metadata = reader.metadata
                
                metadata.update({
                    'title': pdf_metadata.get('/Title', ''),
                    'author': pdf_metadata.get('/Author', ''),
                    'subject': pdf_metadata.get('/Subject', ''),
                    'creator': pdf_metadata.get('/Creator', ''),
                    'producer': pdf_metadata.get('/Producer', ''),
                    'creation_date': pdf_metadata.get('/CreationDate', ''),
                    'modification_date': pdf_metadata.get('/ModDate', '')
                })
                
        except Exception as e:
            if self.show_log:
                logger.warning(f"Error extracting PDF metadata: {str(e)}")
        
        return metadata
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert PyPDF2 output to standardized TextOutput format."""
        text_blocks = raw_output  # raw_output is already a list of TextBlocks
        
        # Sort blocks by page and reading order
        text_blocks.sort(key=lambda x: (x.page_num, x.reading_order or 0))
        
        # Combine all text
        full_text = '\n\n'.join(block.text for block in text_blocks if block.text.strip())
        
        # Get metadata
        metadata = {
            'engine': 'pypdf2',
            'ignore_images': self.ignore_images,
            'extract_forms': self.extract_forms,
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
        password: Optional[str] = None,
        **kwargs
    ) -> TextOutput:
        """Extract text from PDF using PyPDF2.
        
        Args:
            input_path: Path to input PDF
            password: Optional password for encrypted PDFs
            **kwargs: Additional parameters (ignored for PyPDF2)
            
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
            
            # Open PDF
            with open(input_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    if password:
                        if not reader.decrypt(password):
                            raise ValueError("Invalid password for encrypted PDF")
                    else:
                        raise ValueError("PDF is encrypted but no password provided")
                
                total_pages = len(reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(reader.pages, 1):
                    page_blocks = self._extract_page_text(page, page_num)
                    all_text_blocks.extend(page_blocks)
                
                # Extract form fields if requested
                if self.extract_forms:
                    form_blocks = self._extract_form_fields(reader)
                    all_text_blocks.extend(form_blocks)
                
                # Get PDF metadata
                pdf_metadata = self._get_pdf_metadata(reader)
                
                # Create source info
                source_info = {
                    'file_path': str(input_path),
                    'file_name': input_path.name,
                    'file_size': input_path.stat().st_size,
                    'engine': 'pypdf2',
                    'total_pages': total_pages,
                    'is_encrypted': reader.is_encrypted,
                    'pdf_metadata': pdf_metadata
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
    
    def extract_from_pages(
        self,
        input_path: Union[str, Path],
        page_range: Optional[tuple] = None,
        password: Optional[str] = None,
        **kwargs
    ) -> TextOutput:
        """Extract text from specific pages.
        
        Args:
            input_path: Path to input PDF
            page_range: Optional tuple of (start_page, end_page) (1-based, inclusive)
            password: Optional password for encrypted PDFs
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
            
            # Open PDF
            with open(input_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    if password:
                        if not reader.decrypt(password):
                            raise ValueError("Invalid password for encrypted PDF")
                    else:
                        raise ValueError("PDF is encrypted but no password provided")
                
                total_pages = len(reader.pages)
                
                if page_range is None:
                    start_page, end_page = 1, total_pages
                else:
                    start_page, end_page = page_range
                
                # Validate page range
                start_page = max(1, start_page)
                end_page = min(total_pages, end_page)
                
                # Extract text from specified pages
                for page_num in range(start_page, end_page + 1):
                    page = reader.pages[page_num - 1]  # Convert to 0-based index
                    page_blocks = self._extract_page_text(page, page_num)
                    all_text_blocks.extend(page_blocks)
                
                # Get PDF metadata
                pdf_metadata = self._get_pdf_metadata(reader)
                
                # Create source info
                source_info = {
                    'file_path': str(input_path),
                    'file_name': input_path.name,
                    'file_size': input_path.stat().st_size,
                    'engine': 'pypdf2',
                    'total_pages': total_pages,
                    'page_range': page_range,
                    'is_encrypted': reader.is_encrypted,
                    'pdf_metadata': pdf_metadata
                }
            
            # Post-process output
            output = self.postprocess_output(all_text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def extract_with_password(
        self,
        input_path: Union[str, Path],
        password: str,
        **kwargs
    ) -> TextOutput:
        """Extract text from password-protected PDF.
        
        Args:
            input_path: Path to input PDF
            password: Password for encrypted PDF
            **kwargs: Additional parameters
            
        Returns:
            TextOutput containing extracted text
        """
        return self.extract(input_path, password=password, **kwargs)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['.pdf']