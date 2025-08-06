import os
from omnidocs.tasks.layout_analysis.extractors.rtdetr import RTDETRLayoutDetector
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFTextExtractor
from omnidocs.tasks.text_extraction.extractors.surya_text import SuryaTextExtractor
from omnidocs.tasks.table_extraction.extractors.table_transformer import TableTransformerExtractor
from omnidocs.tasks.math_expression_extraction.extractors.surya_math import SuryaMathExtractor

from omnidocs.tasks.layout_analysis.models import LayoutOutput
from omnidocs.tasks.text_extraction.base import TextOutput, TextBlock
from omnidocs.tasks.table_extraction.base import TableOutput
from omnidocs.tasks.math_expression_extraction.base import LatexOutput

from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple

class PDFtoMarkdownConverter:
    def __init__(self):
        """Initialize all the omnidocs extractors."""
        print("🔧 Initializing extractors...")
        self.layout_detector = RTDETRLayoutDetector()
        self.digital_text_extractor = PyMuPDFTextExtractor()
        self.ocr_text_extractor = SuryaTextExtractor()
        self.table_extractor = TableTransformerExtractor()
        self.math_extractor = SuryaMathExtractor()
        print("All extractors loaded!")

    def _is_digital_pdf(self, pdf_path: Path) -> Tuple[bool, TextOutput]:
        """Detect if PDF is digital (has selectable text) or scanned."""
        try:
            # Check if PDF file exists
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            digital_text_output = self.digital_text_extractor.extract(pdf_path)
            # If we get decent amount of text blocks, it's probably digital
            is_digital = len(digital_text_output.text_blocks) >= 5
            print(f"PDF type: {'Digital' if is_digital else 'Scanned'} ({len(digital_text_output.text_blocks)} text blocks found)")
            return is_digital, digital_text_output
        except Exception as e:
            print(f"Digital text extraction failed: {e}")
            # Create proper fallback TextOutput with required fields
            return False, TextOutput(text_blocks=[], full_text="")

    def _get_text_from_bbox(self, digital_blocks: List[TextBlock], target_bbox: Tuple[float, float, float, float], page_num: int) -> str:
        """Extract text that falls within a bounding box from digital PDF blocks."""
        if not digital_blocks:
            return ""
        
        # Filter blocks for current page
        page_blocks = [b for b in digital_blocks if b.page_num == page_num]
        
        if not page_blocks:
            return ""
        
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        matching_blocks = []
        
        for block in page_blocks:
            if not block.bbox or len(block.bbox) < 2:
                continue
                
            block_x, block_y = block.bbox[0], block.bbox[1]
            # Check if block center is within target bbox with some tolerance
            tolerance = 10  # pixels
            if (target_x1 - tolerance <= block_x <= target_x2 + tolerance and 
                target_y1 - tolerance <= block_y <= target_y2 + tolerance):
                matching_blocks.append(block)
        
        # Sort blocks by position (top to bottom, left to right)
        matching_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return " ".join(block.text for block in matching_blocks if block.text).strip()

    def _process_text_block(self, cropped_image: Image.Image, bbox: Tuple, page_num: int, 
                           is_digital: bool, digital_blocks: List[TextBlock]) -> str:
        """Process a text block - use digital extraction if available, otherwise OCR."""
        extracted_text = ""
        
        if is_digital and digital_blocks:
            # Try to get text from digital PDF first
            extracted_text = self._get_text_from_bbox(digital_blocks, bbox, page_num)
        
        if not extracted_text and cropped_image:
            # Fall back to OCR if no digital text found or if image is valid
            try:
                ocr_output = self.ocr_text_extractor.extract(cropped_image)
                if ocr_output.text_blocks:
                    extracted_text = " ".join(block.text for block in ocr_output.text_blocks).strip()
            except Exception as e:
                print(f"OCR extraction failed for text block: {e}")
        
        return extracted_text

    def _process_table_block(self, cropped_image: Image.Image) -> str:
        """Process a table block and convert to markdown table."""
        try:
            table_output = self.table_extractor.extract(cropped_image)
            
            if not table_output.tables:
                return ""
            
            markdown_tables = []
            for table_obj in table_output.tables:
                if not table_obj.cells:
                    continue
                    
                # Get table dimensions
                max_row = max((cell.row + cell.rowspan - 1 for cell in table_obj.cells), default=0)
                max_col = max((cell.col + cell.colspan - 1 for cell in table_obj.cells), default=0)
                
                if max_row < 0 or max_col < 0:
                    continue
                
                # Create grid
                grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
                
                # Fill grid with cell data
                for cell in table_obj.cells:
                    # Clean cell text for markdown
                    cell_text = cell.text.replace('\n', ' ').replace('|', '\\|').strip()
                    
                    # Place text in top-left cell of span
                    if cell.row <= max_row and cell.col <= max_col:
                        grid[cell.row][cell.col] = cell_text
                
                # Convert to markdown
                if grid:
                    # Header row
                    header_row = grid[0]
                    table_md = "| " + " | ".join(header_row) + " |\n"
                    table_md += "|" + "|".join(["---"] * len(header_row)) + "|\n"
                    
                    # Data rows
                    for row in grid[1:]:
                        table_md += "| " + " | ".join(row) + " |\n"
                    
                    markdown_tables.append(table_md)
            
            return "\n".join(markdown_tables) + "\n" if markdown_tables else ""
        
        except Exception as e:
            print(f"Table extraction failed: {e}")
            return "*[Table extraction failed]*\n\n"

    def _process_math_block(self, cropped_image: Image.Image) -> str:
        """Process a math block and convert to LaTeX."""
        try:
            math_output = self.math_extractor.extract(cropped_image)
            
            if not math_output.expressions:
                return ""
            
            math_blocks = []
            for latex_expr in math_output.expressions:
                # Clean up the LaTeX expression
                cleaned_expr = latex_expr.strip()
                if cleaned_expr:
                    math_blocks.append(f"$$\n{cleaned_expr}\n$$")
            
            return "\n".join(math_blocks) + "\n" if math_blocks else ""
        
        except Exception as e:
            print(f"Math extraction failed: {e}")
            return "*[Math expression extraction failed]*\n\n"

    def convert(self, pdf_path: Union[str, Path], output_md_path: Union[str, Path]):
        """Main conversion pipeline."""
        pdf_path = Path(pdf_path)
        output_md_path = Path(output_md_path)
        
        print(f"Starting conversion: {pdf_path}")
        
        # Validate input file
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Input file must be a PDF: {pdf_path}")
        
        # Step 1: Detect if PDF is digital or scanned
        is_digital, digital_text_output = self._is_digital_pdf(pdf_path)
        
        # Step 2: Run layout detection on all pages
        print("Running layout detection...")
        try:
            pages_layout = self.layout_detector.detect_all(pdf_path)
            print(f"Found {len(pages_layout)} pages")
        except Exception as e:
            print(f"Layout detection failed: {e}")
            raise
        
        if not pages_layout:
            raise ValueError("No pages found in PDF or layout detection failed")
        
        markdown_content = [f"# PDF Document: {pdf_path.name}\n\n"]
        
        # Step 3: Process each page
        for page_num, (page_image, layout_output) in enumerate(pages_layout):
            print(f"Processing page {page_num + 1}...")
            markdown_content.append(f"## Page {page_num + 1}\n\n")
            
            # Get digital text blocks for this page
            page_digital_blocks = [b for b in digital_text_output.text_blocks if b.page_num == page_num]
            
            # Sort layout blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(layout_output.bboxes, key=lambda b: (b.bbox[1], b.bbox[0]))
            
            # Step 4: Process each detected block based on its type
            for block in sorted_blocks:
                try:
                    # Crop the region from page image
                    cropped_image = page_image.crop(block.bbox)
                    
                    if block.label == "text":
                        text_content = self._process_text_block(
                            cropped_image, block.bbox, page_num, is_digital, page_digital_blocks
                        )
                        if text_content:
                            markdown_content.append(f"{text_content}\n\n")
                    
                    elif block.label == "table":
                        table_content = self._process_table_block(cropped_image)
                        if table_content:
                            markdown_content.append(f"{table_content}\n")
                    
                    elif block.label == "math":
                        math_content = self._process_math_block(cropped_image)
                        if math_content:
                            markdown_content.append(f"{math_content}\n")
                    
                    elif block.label == "figure":
                        markdown_content.append(f"![Figure on page {page_num + 1}]\n\n")
                    
                    else:
                        # Handle other block types generically
                        markdown_content.append(f"*[{block.label.title()} detected]*\n\n")
                
                except Exception as e:
                    print(f"Warning: Failed to process {block.label} block on page {page_num + 1}: {e}")
                    markdown_content.append(f"*[Failed to process {block.label} block]*\n\n")
            
            # Add page separator
            if page_num < len(pages_layout) - 1:
                markdown_content.append("---\n\n")
        
        # Step 5: Save the markdown file
        try:
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write("".join(markdown_content))
            
            print(f"Conversion complete! Output saved to: {output_md_path}")
        except Exception as e:
            print(f"Failed to save output file: {e}")
            raise

if __name__ == "__main__":
    converter = PDFtoMarkdownConverter()
    
    # Update these paths
    pdf_input = "omnidocs\\workflows\\assets\\formulas_pdf-3-5.pdf"
    markdown_output = "output_simple_doc.md"
    
    try:
        converter.convert(pdf_input, markdown_output)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure the PDF file exists in the current directory.")
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()