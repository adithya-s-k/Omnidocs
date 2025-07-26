"""
Simple NougatExtractor test - just pass an image and print the full output.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

def main():
    """Simple test - just pass an image and print the output."""

    print("Simple NougatExtractor Test")
    print("=" * 50)

    try:
        # Initialize extractor
        print("Initializing NougatExtractor...")
        extractor = NougatExtractor(
            model_type="small",
            device='cpu',
            show_log=True
        )
        print("NougatExtractor initialized!")

        # Test with the math equation image
        image_path = "tests/math_expression_extraction/assets/math_equation.png"
        print(f"Processing image: {image_path}")

        # Check if image exists
        if not Path(image_path).exists():
            print(f"Error: Image file not found at {image_path}")
            return

        result = extractor.extract(image_path)

        print("\n" + "=" * 80)
        print("NOUGAT OUTPUT:")
        print("=" * 80)
        print(f"Result type: {type(result)}")
        print(f"Result object: {result}")

        if hasattr(result, 'expressions'):
            print(f"Number of expressions: {len(result.expressions)}")
            for i, expr in enumerate(result.expressions):
                print(f"\nExpression {i+1}:")
                print(f"  Raw: {repr(expr)}")
                print(f"  Display: {expr}")
                print(f"  Length: {len(expr)} characters")

        if hasattr(result, 'source_img_size'):
            print(f"Source image size: {result.source_img_size}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()