import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from omnidocs.tasks.text_extraction.extractors import (
    PyMuPDFExtractor, #works
    PyPDF2Extractor, #works
    PDFPlumberTextExtractor, #works
    PDFTextExtractor, #works
    DoclingExtractor #works
)

def test_text_extraction():
    extractors = [
        PyMuPDFExtractor,
        PyPDF2Extractor,
        PDFPlumberTextExtractor,
        PDFTextExtractor,
        DoclingExtractor
    ]

    # All extractors should be available now

    pdf_path = "tests/text_extraction/assets/sample_document.pdf"

    for extractor_cls in extractors:
        print(f"\n{'='*50}")
        print(f"Testing {extractor_cls.__name__}")
        print(f"{'='*50}")

        try:
            result = extractor_cls().extract(pdf_path)
            print(f"Extracted text length: {len(result.full_text)} characters")
            print(f"First 200 characters:")
            print(f"'{result.full_text[:200]}...'")
            assert len(result.full_text) > 0
            print("SUCCESS: Text extraction completed")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

if __name__ == "__main__":
    test_text_extraction()
