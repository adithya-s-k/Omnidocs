from pathlib import Path
from omnidocs.tasks.math_expression_extraction.extractors import SuryaMathExtractor
def test_surya_math_extractor():
    """Test SuryaMathExtractor functionality."""
    image_path = Path("tests/math_expression_extraction/assets/math_equation.png")
    
    # Verify test asset exists
    assert image_path.exists(), f"Test asset not found: {image_path}"

    extractor = SuryaMathExtractor(device='cpu', show_log=False)
    print("Surya Extractor initialized!")
    result = extractor.extract(str(image_path))
    print(f"Surya: Found {len(result.expressions)} expressions")
    
    # Basic assertions
    assert hasattr(result, 'expressions')
    assert isinstance(result.expressions, list)

    if result.expressions:
        expr = result.expressions[0]
        print(f"LaTeX: {expr[:80]}...")
        assert isinstance(expr, str)
        assert len(expr) > 0

if __name__ == "__main__":
    test_surya_math_extractor()