import os
from omnidocs.tasks.layout_analysis.extractors.surya import SuryaLayoutDetector
from omnidocs.tasks.text_extraction.extractors.surya_text import SuryaTextExtractor
from omnidocs.tasks.math_expression_extraction.extractors.surya_math import SuryaMathExtractor
from omnidocs.tasks.table_extraction.extractors.surya_table import SuryaTableExtractor

from omnidocs.tasks.layout_analysis.models import LayoutOutput
from omnidocs.tasks.text_extraction.base import TextOutput, TextBlock
from omnidocs.tasks.math_expression_extraction.base import LatexOutput
from omnidocs.tasks.table_extraction.base import TableOutput

from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple

class SuryaPDFtoMarkdownConverter:
    def __init__(self):
        """Initialize all Surya extractors including layout detector."""
        print("Initializing Surya pipeline...")
        self.layout_detector = SuryaLayoutDetector(show_log=True)
        self.text_extractor = SuryaTextExtractor()
        self.math_extractor = SuryaMathExtractor()
        self.table_extractor = SuryaTableExtractor()
        print("All Surya extractors loaded!")

    def _process_text_region(self, cropped_image: Image.Image) -> str:
        """Process a text region using Surya text extractor."""
        try:
            text_output = self.text_extractor.extract(cropped_image)
            if text_output.text_blocks:
                # Combine all text blocks from the region
                text_content = " ".join(block.text for block in text_output.text_blocks if block.text).strip()
                return text_content
            return ""
        except Exception as e:
            print(f"  Text extraction failed for region: {e}")
            return ""

    def _process_math_region(self, cropped_image: Image.Image) -> str:
        """Process a math region using Surya math extractor."""
        try:
            math_output = self.math_extractor.extract(cropped_image)
            if math_output.expressions:
                math_blocks = []
                for expr in math_output.expressions:
                    if expr.strip():
                        math_blocks.append(f"$$\n{expr.strip()}\n$$")
                return "\n".join(math_blocks) + "\n" if math_blocks else ""
            return ""
        except Exception as e:
            print(f"  Math extraction failed for region: {e}")
            return "*[Math expression extraction failed]*\n\n"

    def _process_table_region(self, cropped_image: Image.Image) -> str:
        """Process a table region using Surya table extractor."""
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
            print(f"  Table extraction failed for region: {e}")
            return "*[Table extraction failed]*\n\n"

    def convert(self, pdf_path: Union[str, Path], output_md_path: Union[str, Path]):
        """Proper pipeline: Layout detection → Region-specific extraction."""
        pdf_path = Path(pdf_path)
        output_md_path = Path(output_md_path)
        
        print(f"Starting Surya pipeline conversion: {pdf_path}")
        print("=" * 50)
        
        # Validate input file
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Input file must be a PDF: {pdf_path}")
        
        # Step 1: Run Surya layout detection on all pages
        print("🔍 Running Surya layout detection...")
        try:
            pages_layout = self.layout_detector.detect_all(pdf_path)
            print(f"Found {len(pages_layout)} pages")
        except Exception as e:
            print(f"  Layout detection failed: {e}")
            raise
        
        if not pages_layout:
            raise ValueError("No pages found in PDF or layout detection failed")
        
        # Start building markdown content
        markdown_content = [f"# PDF Document: {pdf_path.name}\n\n"]
        markdown_content.append(f"*Converted using Surya pipeline (Layout + Text + Math)*\n\n")
        
        total_text_regions = 0
        total_math_regions = 0
        total_table_regions = 0
        total_other_regions = 0
        
        # Step 2: Process each page with detected layout
        for page_num, (page_image, layout_output) in enumerate(pages_layout):
            print(f"Processing page {page_num + 1}...")
            markdown_content.append(f"## Page {page_num + 1}\n\n")
            
            # Sort layout blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(layout_output.bboxes, 
                                 key=lambda b: (b.bbox[1], b.bbox[0]))
            
            print(f"   Found {len(sorted_blocks)} regions on page {page_num + 1}")
            
            # Step 3: Process each detected region based on its type
            for i, block in enumerate(sorted_blocks):
                try:
                    # Crop the region from page image
                    cropped_image = page_image.crop(block.bbox)
                    
                    if block.label == "text":
                        total_text_regions += 1
                        text_content = self._process_text_region(cropped_image)
                        if text_content:
                            # Simple formatting heuristics
                            if len(text_content) < 100 and text_content.isupper():
                                markdown_content.append(f"### {text_content}\n\n")
                            elif text_content.endswith(':') and len(text_content) < 50:
                                markdown_content.append(f"**{text_content}**\n\n")
                            else:
                                markdown_content.append(f"{text_content}\n\n")
                    
                    elif block.label == "formula" or block.label == "equation":
                        total_math_regions += 1
                        math_content = self._process_math_region(cropped_image)
                        if math_content:
                            markdown_content.append(f"{math_content}\n")
                    
                    elif block.label == "table":
                        total_table_regions += 1
                        table_content = self._process_table_region(cropped_image)
                        if table_content:
                            markdown_content.append(f"{table_content}\n")
                        else:
                            markdown_content.append(f"*[Table detected but extraction failed - region {i+1}]*\n\n")
                    
                    elif block.label == "figure":
                        total_other_regions += 1
                        markdown_content.append(f"*[Figure detected - region {i+1}]*\n\n")
                    
                    elif block.label == "title":
                        total_text_regions += 1
                        text_content = self._process_text_region(cropped_image)
                        if text_content:
                            markdown_content.append(f"# {text_content}\n\n")
                    
                    elif block.label == "list":
                        total_text_regions += 1
                        text_content = self._process_text_region(cropped_image)
                        if text_content:
                            # Simple list formatting
                            lines = text_content.split('\n')
                            for line in lines:
                                if line.strip():
                                    markdown_content.append(f"- {line.strip()}\n")
                            markdown_content.append("\n")
                    
                    else:
                        total_other_regions += 1
                        # Handle other block types generically
                        markdown_content.append(f"*[{block.label.title()} detected - region {i+1}]*\n\n")
                
                except Exception as e:
                    print(f"Warning: Failed to process {block.label} region {i+1} on page {page_num + 1}: {e}")
                    markdown_content.append(f"*[Failed to process {block.label} region]*\n\n")
            
            # Add page separator (except for last page)
            if page_num < len(pages_layout) - 1:
                markdown_content.append("---\n\n")
        
        # Add extraction summary
        markdown_content.append("---\n\n")
        markdown_content.append("## Extraction Summary\n\n")
        markdown_content.append(f"- **Pages processed**: {len(pages_layout)}\n")
        markdown_content.append(f"- **Text regions extracted**: {total_text_regions}\n")
        markdown_content.append(f"- **Math regions found**: {total_math_regions}\n")
        markdown_content.append(f"- **Table regions found**: {total_table_regions}\n")
        markdown_content.append(f"- **Other regions**: {total_other_regions}\n")
        markdown_content.append(f"- **Pipeline used**: Surya Layout → Surya Text/Math/Table\n\n")
        
        # Save the markdown file
        try:
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write("".join(markdown_content))
            
            print("=" * 50)
            print(f"Conversion complete! Output saved to: {output_md_path}")
            print(f"Pages: {len(pages_layout)}")
            print(f"Text regions: {total_text_regions}")
            print(f"Math regions: {total_math_regions}")
            print(f"Table regions: {total_table_regions}")
            print(f"Other regions: {total_other_regions}")
        except Exception as e:
            print(f"  Failed to save output file: {e}")
            raise

if __name__ == "__main__":
    converter = SuryaPDFtoMarkdownConverter()
    
    # Update these paths
    pdf_input = "omnidocs\\workflows\\assets\\llama.pdf"
    markdown_output = "output_surya_doc.md"
    
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
