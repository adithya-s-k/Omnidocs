import sys
import os
import warnings
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def ocr_result_to_dict(result):
    texts = []
    if hasattr(result, "texts"):
        for text_obj in result.texts:
            texts.append({
                "text": getattr(text_obj, "text", ""),
                "bbox": getattr(text_obj, "bbox", None),
                "polygon": getattr(text_obj, "polygon", None),
                "confidence": getattr(text_obj, "confidence", None),
                "language": getattr(text_obj, "language", None),
                "reading_order": getattr(text_obj, "reading_order", None)
            })
    return {
        "full_text": getattr(result, "full_text", ""),
        "texts": texts if texts else None,
        "source_img_size": getattr(result, "source_img_size", None),
        "processing_time": getattr(result, "processing_time", None)
    }

def save_ocr_to_json(result, extractor_name, filetype):
    data = ocr_result_to_dict(result)
    out_path = f"output_{extractor_name}_{filetype}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {out_path}")

def test_ocr_extraction():
    # Import all available extractors
    from omnidocs.tasks.ocr_extraction.extractors.paddle import PaddleOCRExtractor
    from omnidocs.tasks.ocr_extraction.extractors.tesseract_ocr import TesseractOCRExtractor
    from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor
    
    extractors = [
        PaddleOCRExtractor,
        TesseractOCRExtractor, 
        EasyOCRExtractor,
    ]
    
    # Try to import SuryaOCR if available
    try:
        from omnidocs.tasks.ocr_extraction.extractors.surya_ocr import SuryaOCRExtractor
        # Test if dependencies are actually available by trying to instantiate
        try:
            _ = SuryaOCRExtractor()
            extractors.append(SuryaOCRExtractor)
            print("SuryaOCR available and dependencies satisfied")
        except ImportError as dep_error:
            print(f"SuryaOCR import successful but dependencies missing: {dep_error}")
        except Exception as init_error:
            print(f"SuryaOCR dependencies check failed: {init_error}")
    except ImportError as e:
        print(f"SuryaOCR not available: {e}")

    print(f"Total working extractors: {len(extractors)}")

    image_path = "tests/ocr_extraction/assets/invoice.jpg"

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
            
            print(result)
            print(hasattr(result, "texts"))
            if hasattr(result, "texts"):
                print(result.texts)
            # Save to JSON
            save_ocr_to_json(result, extractor_cls.__name__, "image")

            print("SUCCESS: OCR extraction completed")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

if __name__ == "__main__":
    test_ocr_extraction()
