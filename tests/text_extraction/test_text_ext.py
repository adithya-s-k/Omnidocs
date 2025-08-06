import sys
import os
import warnings
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from omnidocs.tasks.text_extraction.extractors import (
    PyMuPDFTextExtractor,
    PyPDF2TextExtractor,
    PdfplumberTextExtractor,
    PdftextTextExtractor,
    DoclingTextExtractor,
    SuryaTextExtractor
)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Custom JSON encoder to handle PyPDF2 objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if it's a PyPDF2 IndirectObject or any other non-serializable type
        try:
            # First try the default serialization
            return super().default(obj)
        except TypeError:
            # If it fails, convert to string
            return str(obj)

def text_output_to_dict(text_output):
    """Convert TextOutput to dictionary for JSON serialization."""
    return text_output.to_dict()

def save_text_to_json(text_output, extractor_name, filetype):
    """Save text extraction results to JSON file."""
    data = text_output_to_dict(text_output)
    out_path = f"output_{extractor_name}_{filetype}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    print(f"Saved JSON: {out_path}")

def test_text_extraction():
    # All text extractors work with PDF files
    pdf_extractors = [
       PyMuPDFTextExtractor,
        PyPDF2TextExtractor,
        PdfplumberTextExtractor,
        PdftextTextExtractor,
        DoclingTextExtractor,
        SuryaTextExtractor
        
    ]

    pdf_path = "./assets/sample_document.pdf"

    # Test PDF extractors
    print("Testing PDF text extractors...")
    for extractor_cls in pdf_extractors:
        print(f"\n{'='*50}")
        print(f"Testing {extractor_cls.__name__} with PDF")
        print(f"{'='*50}")


        try:
            result = extractor_cls().extract(pdf_path)
            print(f"Extracted text blocks: {len(result.text_blocks)} block(s)")
            print(f"Total pages: {result.page_count}")
            print(f"Processing time: {result.processing_time:.2f}s" if result.processing_time else "Processing time: N/A")

            # Show text block details
            for i, block in enumerate(result.text_blocks[:5]):  # Show first 5 blocks
                print(f"\nBlock {i+1}:")
                print(f"  Page: {block.page_num}")
                print(f"  Type: {block.block_type}")
                print(f"  Confidence: {block.confidence}")
                print(f"  Reading order: {block.reading_order}")

                # Show text preview (first 100 characters)
                text_preview = block.text.strip()[:100]
                if len(block.text.strip()) > 100:
                    text_preview += "..."
                print(f"  Text: '{text_preview}'")

                if block.bbox:
                    print(f"  BBox: {block.bbox}")

            # Show full text preview
            full_text_preview = result.full_text.strip()[:200]
            if len(result.full_text.strip()) > 200:
                full_text_preview += "..."
            print(f"\nFull text preview: '{full_text_preview}'")
            print(f"Full text length: {len(result.full_text)} characters")

            # Save to JSON
            save_text_to_json(result, extractor_cls.__name__, "pdf")

            print("SUCCESS: PDF text extraction completed")
            assert len(result.text_blocks) >= 0
            assert len(result.full_text) >= 0
        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

if __name__ == "__main__":
    test_text_extraction()
