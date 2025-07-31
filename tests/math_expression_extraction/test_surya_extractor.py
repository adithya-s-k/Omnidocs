from omnidocs.tasks.math_expression_extraction.extractors import SuryaMathExtractor

image_path = "tests/math_expression_extraction/assets/math_equation.png"


extractor = SuryaMathExtractor(device='cpu', show_log=False)
print("Surya Extractor initialized!")

result = extractor.extract(image_path)
print(f"Sruya : Found {len(result.expressions)} expressions")

if result.expressions:
    expr = result.expressions[0]
    print(f"LaTeX: {expr[:80]}...")