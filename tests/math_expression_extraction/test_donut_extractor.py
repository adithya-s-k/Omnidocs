"""
Simple DonutExtractor test - just pass an image and print the full output.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

def main():
    """Simple test - just pass an image and print the output."""

    print("Simple DonutExtractor Test")
    print("=" * 50)

    # Initialize extractor
    extractor = DonutExtractor(device='cpu', show_log=True)

    # Test with the math equation image
    image_path = "tests/math_expression_extraction/assets/math_equation.png"

    result = extractor.extract(image_path)
    result.save_json()
    result.save_txt()
    result.save_md()

    
if __name__ == "__main__":
    main()