from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from pydantic import BaseModel, Field
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)

class TextBlock(BaseModel):
    """
    Container for individual text block.
    
    Attributes:
        text: Text content
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Confidence score for text extraction
        page_num: Page number (for multi-page documents)
        block_type: Type of text block (paragraph, heading, list, etc.)
        font_info: Optional font information
        reading_order: Reading order index
        language: Detected language of the text
    """
    text: str = Field(..., description="Text content")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
    page_num: int = Field(1, description="Page number (1-based)")
    block_type: Optional[str] = Field(None, description="Type of text block")
    font_info: Optional[Dict[str, Any]] = Field(None, description="Font information")
    reading_order: Optional[int] = Field(None, description="Reading order index")
    language: Optional[str] = Field(None, description="Detected language code")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'page_num': self.page_num,
            'block_type': self.block_type,
            'font_info': self.font_info,
            'reading_order': self.reading_order,
            'language': self.language
        }

class TextOutput(BaseModel):
    """
    Container for text extraction results.
    
    Attributes:
        text_blocks: List of extracted text blocks
        full_text: Combined text from all blocks
        metadata: Additional metadata from extraction
        source_info: Information about the source document
        processing_time: Time taken for text extraction
        page_count: Number of pages in the document
    """
    text_blocks: List[TextBlock] = Field(..., description="List of extracted text blocks")
    full_text: str = Field(..., description="Combined text from all blocks")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional extraction metadata")
    source_info: Optional[Dict[str, Any]] = Field(None, description="Source document information")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    page_count: int = Field(1, description="Number of pages in the document")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'text_blocks': [block.to_dict() for block in self.text_blocks],
            'full_text': self.full_text,
            'metadata': self.metadata,
            'source_info': self.source_info,
            'processing_time': self.processing_time,
            'page_count': self.page_count
        }
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save output to JSON file."""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_text(self, output_path: Union[str, Path]) -> None:
        """Save full text to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.full_text)
    
    def save_markdown(self, output_path: Union[str, Path]) -> None:
        """Save text as markdown with basic formatting."""
        markdown_content = []
        
        for block in self.get_sorted_by_reading_order():
            if block.block_type == 'heading':
                # Convert to markdown heading
                markdown_content.append(f"# {block.text}\n")
            elif block.block_type == 'subheading':
                markdown_content.append(f"## {block.text}\n")
            elif block.block_type == 'list':
                # Convert to markdown list
                lines = block.text.split('\n')
                for line in lines:
                    if line.strip():
                        markdown_content.append(f"- {line.strip()}")
                markdown_content.append("")
            else:
                # Regular paragraph
                markdown_content.append(f"{block.text}\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
    
    def get_text_by_confidence(self, min_confidence: float = 0.5) -> List[TextBlock]:
        """Filter text blocks by minimum confidence threshold."""
        return [block for block in self.text_blocks if block.confidence is None or block.confidence >= min_confidence]
    
    def get_text_by_page(self, page_num: int) -> List[TextBlock]:
        """Get text blocks from a specific page."""
        return [block for block in self.text_blocks if block.page_num == page_num]
    
    def get_text_by_type(self, block_type: str) -> List[TextBlock]:
        """Get text blocks of a specific type."""
        return [block for block in self.text_blocks if block.block_type == block_type]
    
    def get_sorted_by_reading_order(self) -> List[TextBlock]:
        """Get text blocks sorted by reading order."""
        blocks_with_order = [block for block in self.text_blocks if block.reading_order is not None]
        blocks_without_order = [block for block in self.text_blocks if block.reading_order is None]
        
        # Sort blocks with reading order
        blocks_with_order.sort(key=lambda x: (x.page_num, x.reading_order))
        
        # Sort blocks without reading order by page and bbox
        if blocks_without_order:
            blocks_without_order.sort(key=lambda x: (
                x.page_num,
                x.bbox[1] if x.bbox else 0,  # Sort by y coordinate (top to bottom)
                x.bbox[0] if x.bbox else 0   # Then by x coordinate (left to right)
            ))
        
        return blocks_with_order + blocks_without_order

class BaseTextMapper:
    """Base class for mapping text extraction engine-specific outputs to standardized format."""
    
    def __init__(self, engine_name: str):
        """Initialize mapper for specific text extraction engine.
        
        Args:
            engine_name: Name of the text extraction engine
        """
        self.engine_name = engine_name.lower()
        self._block_type_mapping: Dict[str, str] = {}
        self._setup_block_type_mapping()
    
    def _setup_block_type_mapping(self):
        """Setup block type mapping for different engines."""
        # Common block type mappings
        self._block_type_mapping = {
            'title': 'heading',
            'header': 'heading',
            'h1': 'heading',
            'h2': 'subheading',
            'h3': 'subheading',
            'subtitle': 'subheading',
            'paragraph': 'paragraph',
            'text': 'paragraph',
            'body': 'paragraph',
            'list': 'list',
            'bullet': 'list',
            'table': 'table',
            'caption': 'caption',
            'footer': 'footer',
            'footnote': 'footnote'
        }
    
    def normalize_block_type(self, engine_type: str) -> str:
        """Convert engine-specific block type to standardized format."""
        return self._block_type_mapping.get(engine_type.lower(), engine_type)
    
    def normalize_bbox(self, bbox: List[float], page_width: int, page_height: int) -> List[float]:
        """Normalize bounding box coordinates to absolute values."""
        if all(0 <= coord <= 1 for coord in bbox):
            return [
                bbox[0] * page_width,
                bbox[1] * page_height,
                bbox[2] * page_width,
                bbox[3] * page_height
            ]
        return bbox
    
    def extract_font_info(self, raw_font_data: Any) -> Dict[str, Any]:
        """Extract and normalize font information."""
        font_info = {}
        
        if isinstance(raw_font_data, dict):
            font_info.update({
                'font_name': raw_font_data.get('name', raw_font_data.get('font_name')),
                'font_size': raw_font_data.get('size', raw_font_data.get('font_size')),
                'bold': raw_font_data.get('bold', raw_font_data.get('is_bold', False)),
                'italic': raw_font_data.get('italic', raw_font_data.get('is_italic', False)),
                'color': raw_font_data.get('color', raw_font_data.get('font_color'))
            })
        
        return {k: v for k, v in font_info.items() if v is not None}

class BaseTextExtractor(ABC):
    """Base class for text extraction models."""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 show_log: bool = False,
                 engine_name: Optional[str] = None,
                 extract_images: bool = False):
        """Initialize the text extractor.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            show_log: Whether to show detailed logs
            engine_name: Name of the text extraction engine
            extract_images: Whether to extract images alongside text
        """
        self.show_log = show_log
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine_name = engine_name or self.__class__.__name__.lower().replace('extractor', '')
        self.extract_images = extract_images
        self.model = None
        self.model_path = None
        self._label_mapper: Optional[BaseTextMapper] = None
        
        # Initialize mapper if engine name is provided
        if self.engine_name:
            self._label_mapper = BaseTextMapper(self.engine_name)
        
        if self.show_log:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Engine: {self.engine_name}")
            logger.info(f"Extract images: {self.extract_images}")
    
    @abstractmethod
    def _download_model(self) -> Path:
        """Download model from remote source."""
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load model into memory."""
        pass
    
    def preprocess_input(self, input_path: Union[str, Path]) -> Any:
        """Preprocess input document for text extraction.
        
        Args:
            input_path: Path to input document
            
        Returns:
            Preprocessed document object
        """
        # Default implementation - child classes should override for specific formats
        return input_path
    
    def postprocess_output(self, raw_output: Any, source_info: Optional[Dict] = None) -> TextOutput:
        """Convert raw text extraction output to standardized TextOutput format.
        
        Args:
            raw_output: Raw output from text extraction engine
            source_info: Optional source document information
            
        Returns:
            Standardized TextOutput object
        """
        raise NotImplementedError("Child classes must implement postprocess_output method")
    
    @abstractmethod
    def extract(
        self,
        input_path: Union[str, Path],
        **kwargs
    ) -> TextOutput:
        """Extract text from input document.
        
        Args:
            input_path: Path to input document
            **kwargs: Additional model-specific parameters
            
        Returns:
            TextOutput containing extracted text
        """
        pass
    
    def extract_all(
        self,
        input_paths: List[Union[str, Path]],
        **kwargs
    ) -> List[TextOutput]:
        """Extract text from multiple documents.
        
        Args:
            input_paths: List of document paths
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of TextOutput objects
        """
        results = []
        for input_path in input_paths:
            try:
                result = self.extract(input_path, **kwargs)
                results.append(result)
            except Exception as e:
                if self.show_log:
                    logger.error(f"Error processing {input_path}: {str(e)}")
                raise
        return results
    
    def extract_from_pages(
        self,
        input_path: Union[str, Path],
        page_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> TextOutput:
        """Extract text from specific pages of a document.
        
        Args:
            input_path: Path to input document
            page_range: Optional tuple of (start_page, end_page) (1-based, inclusive)
            **kwargs: Additional model-specific parameters
            
        Returns:
            TextOutput containing extracted text from specified pages
        """
        # Default implementation extracts all pages then filters
        # Child classes can override for more efficient page-specific extraction
        full_output = self.extract(input_path, **kwargs)
        
        if page_range is None:
            return full_output
        
        start_page, end_page = page_range
        filtered_blocks = [
            block for block in full_output.text_blocks
            if start_page <= block.page_num <= end_page
        ]
        
        # Rebuild full text from filtered blocks
        full_text = '\n'.join(block.text for block in filtered_blocks)
        
        return TextOutput(
            text_blocks=filtered_blocks,
            full_text=full_text,
            metadata=full_output.metadata,
            source_info=full_output.source_info,
            processing_time=full_output.processing_time,
            page_count=end_page - start_page + 1
        )
    
    def extract_with_layout(
        self,
        input_path: Union[str, Path],
        layout_regions: Optional[List[Dict]] = None,
        **kwargs
    ) -> TextOutput:
        """Extract text with optional layout information.
        
        Args:
            input_path: Path to input document
            layout_regions: Optional list of layout regions to focus extraction on
            **kwargs: Additional model-specific parameters
            
        Returns:
            TextOutput containing extracted text
        """
        # Default implementation just calls extract, can be overridden by child classes
        return self.extract(input_path, **kwargs)
    
    @property
    def label_mapper(self) -> BaseTextMapper:
        """Get the label mapper for this extractor."""
        if self._label_mapper is None:
            raise ValueError("Label mapper not initialized")
        return self._label_mapper
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        # Default formats - child classes should override
        return ['.txt', '.pdf']
