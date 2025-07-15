import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pdftext import extraction

from omnidocs.tasks.text_extraction.base import BaseTextExtractor, TextOutput, TextBlock, BaseTextMapper
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class PdftextTextMapper(BaseTextMapper):
    """Mapper for pdftext text extraction output."""
    
    def __init__(self):
        super().__init__("pdftext")
    
    def _setup_block_type_mapping(self):
        """Setup pdftext-specific block type mapping."""
        super()._setup_block_type_mapping()
        # pdftext provides minimal structure, mostly raw text
        self._block_type_mapping.update({
            'text': 'paragraph',
            'line': 'paragraph',
            'page': 'paragraph'
        })

class PdftextTextExtractor(BaseTextExtractor):
    """Text extractor using pdftext."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 extract_images: bool = False,
                 keep_layout: bool = False,
                 physical_layout: bool = False):
        """Initialize pdftext text extractor.
        
        Args:
            device: Device to run on (not used for pdftext)
            show_log: Whether to show detailed logs
            extract_images: Whether to extract images alongside text
            keep_layout: Whether to keep original layout formatting
            physical_layout: Whether to use physical layout analysis
        """
        super().__init__(device, show_log, "pdftext", extract_images)
        self.keep_layout = keep_layout
        self.physical_layout = physical_layout
        self._label_mapper = PdftextTextMapper()
        self._load_model()
    
    def _download_model(self) -> Path:
        """Download model - not applicable for pdftext."""
        return Path(".")
    
    def _load_model(self) -> None:
        """Load pdftext - no model loading required."""
        try:
            # pdftext doesn't require model loading
            self.model = extraction
            
            if self.show_log:
                logger.info("pdftext loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load pdftext: {str(e)}")
            raise
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Path:
        """Preprocess input document."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() != '.pdf':
            raise ValueError(f"pdftext only supports PDF files. Got: {input_path.suffix}")
        
        return input_path
    
    def _extract_text_by_page(self, input_path: Path) -> List[TextBlock]:
        """Extract text page by page."""
        text_blocks = []

        try:
            # Extract text using pdftext plain_text_output
            all_text = extraction.plain_text_output(str(input_path))

            if all_text and all_text.strip():
                # Split into paragraphs (simple approach)
                paragraphs = all_text.split('\n\n')

                for para_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        block = TextBlock(
                            text=paragraph.strip(),
                            bbox=None,  # pdftext doesn't provide bounding boxes
                            confidence=1.0,
                            page_num=1,  # Can't determine page number easily
                            block_type='paragraph',
                            reading_order=para_idx
                        )
                        text_blocks.append(block)

        except Exception as e:
            logger.error(f"Error extracting text by page: {str(e)}")
            # Fallback to extracting all text at once
            return self._extract_text_all(input_path)

        return text_blocks
    
    def _extract_text_all(self, input_path: Path) -> List[TextBlock]:
        """Extract all text at once."""
        text_blocks = []
        
        try:
            # Extract all text from PDF using pdftext
            all_text = extraction.plain_text_output(str(input_path))
            
            if all_text and all_text.strip():
                # Split into paragraphs
                paragraphs = all_text.split('\n\n')
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        block = TextBlock(
                            text=paragraph.strip(),
                            bbox=None,
                            confidence=1.0,
                            page_num=1,  # Can't determine page number in bulk extraction
                            block_type='paragraph',
                            reading_order=para_idx
                        )
                        text_blocks.append(block)
                        
        except Exception as e:
            logger.error(f"Error extracting all text: {str(e)}")
            # Create single block with empty text as fallback
            block = TextBlock(
                text="",
                bbox=None,
                confidence=0.0,
                page_num=1,
                block_type='paragraph',
                reading_order=0
            )
            text_blocks.append(block)
        
        return text_blocks
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert pdftext output to standardized TextOutput format."""
        text_blocks = raw_output  # raw_output is already a list of TextBlocks
        
        # Sort blocks by page and reading order
        text_blocks.sort(key=lambda x: (x.page_num, x.reading_order or 0))
        
        # Combine all text
        full_text = '\n\n'.join(block.text for block in text_blocks if block.text.strip())
        
        # Get metadata
        metadata = {
            'engine': 'pdftext',
            'keep_layout': self.keep_layout,
            'physical_layout': self.physical_layout,
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
        """Extract text from PDF using pdftext.
        
        Args:
            input_path: Path to input PDF
            **kwargs: Additional parameters (ignored for pdftext)
            
        Returns:
            TextOutput containing extracted text
        """
        start_time = time.time()
        
        # Preprocess input
        input_path = self.preprocess_input(input_path)
        
        if self.show_log:
            logger.info(f"Extracting text from {input_path}")
        
        try:
            # Extract text blocks
            text_blocks = self._extract_text_by_page(input_path)
            
            # Create source info
            source_info = {
                'file_path': str(input_path),
                'file_name': input_path.name,
                'file_size': input_path.stat().st_size,
                'engine': 'pdftext'
            }
            
            # Post-process output
            output = self.postprocess_output(text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def extract_from_pages(
        self,
        input_path: Union[str, Path],
        page_range: Optional[tuple] = None,
        **kwargs
    ) -> TextOutput:
        """Extract text from specific pages.
        
        Args:
            input_path: Path to input PDF
            page_range: Optional tuple of (start_page, end_page) (1-based, inclusive)
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
            text_blocks = []
            
            if page_range is None:
                # Extract all pages
                text_blocks = self._extract_text_by_page(input_path)
            else:
                start_page, end_page = page_range
                
                for page_num in range(start_page, end_page + 1):
                    try:
                        page_text = pdftext.pdf_text(
                            str(input_path), 
                            page_num=page_num,
                            keep_layout=self.keep_layout,
                            physical_layout=self.physical_layout
                        )
                        
                        if page_text and page_text.strip():
                            paragraphs = page_text.split('\n\n')
                            
                            for para_idx, paragraph in enumerate(paragraphs):
                                if paragraph.strip():
                                    block = TextBlock(
                                        text=paragraph.strip(),
                                        bbox=None,
                                        confidence=1.0,
                                        page_num=page_num,
                                        block_type='paragraph',
                                        reading_order=para_idx
                                    )
                                    text_blocks.append(block)
                                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {str(e)}")
                        continue
            
            # Create source info
            source_info = {
                'file_path': str(input_path),
                'file_name': input_path.name,
                'file_size': input_path.stat().st_size,
                'engine': 'pdftext',
                'page_range': page_range
            }
            
            # Post-process output
            output = self.postprocess_output(text_blocks, source_info)
            output.processing_time = time.time() - start_time
            
            if self.show_log:
                logger.info(f"Extracted {len(output.text_blocks)} text blocks in {output.processing_time:.2f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting text from {input_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['.pdf']