import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

warnings.filterwarnings("ignore")

from omnidocs.tasks.ocr_extraction.extractors import (
    PaddleOCRExtractor,
    TesseractOCRExtractor,
    EasyOCRExtractor,
    SuryaOCRExtractor
)

def test_ocr_extraction():
    extractors = [
        PaddleOCRExtractor,
        TesseractOCRExtractor,
        EasyOCRExtractor,
        SuryaOCRExtractor
    ]

    image_path = "omnidocs/tests/ocr_extraction/assets/invoice.jpg"

    for extractor_cls in extractors:
        print(f"\n{'='*50}")
        print(f"Testing {extractor_cls.__name__}")
        print(f"{'='*50}")

        try:
            result = extractor_cls().extract(image_path)
            print(f"Extracted text length: {len(result.full_text)} characters")
            print(f"First 200 characters:")
            print(f"'{result.full_text[:200]}...'")
            assert len(result.full_text) > 0
            print("SUCCESS: OCR extraction completed")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

if __name__ == "__main__":
    test_ocr_extraction()
