"""
Simple TexifyExtractor test - just pass an image and print the full output.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from omnidocs.tasks.math_expression_extraction.extractors.texify import TexifyExtractor

def main():
    """Simple test - just pass an image and print output."""

    print("🧪 Simple TexifyExtractor Test")
    print("=" * 40)

    try:
        # Initialize extractor
        print("🔧 Initializing TexifyExtractor...")
        extractor = TexifyExtractor(device='cpu', show_log=True)
        print("✅ TexifyExtractor initialized!")

        # Test with the math equation image
        image_path = "omnidocs/tests/math_expression_extraction/assets/math_equation.png"
        print(f"📸 Processing image: {image_path}")

        result = extractor.extract(image_path)

        print("\n" + "=" * 60)
        print("🎯 FULL TEXIFY OUTPUT:")
        print("=" * 60)
        print(f"Result type: {type(result)}")
        print(f"Result object: {result}")

        if hasattr(result, 'expressions'):
            print(f"Number of expressions: {len(result.expressions)}")
            for i, expr in enumerate(result.expressions):
                print(f"\nExpression {i+1}:")
                print(f"  Raw: {repr(expr)}")
                print(f"  Display: {expr}")

        if hasattr(result, 'source_img_size'):
            print(f"Source image size: {result.source_img_size}")

        if hasattr(result, 'confidences'):
            print(f"Confidences: {result.confidences}")

        print("\n" + "=" * 60)
        print("✅ TexifyExtractor test completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
