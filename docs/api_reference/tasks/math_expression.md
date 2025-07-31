# 🔢 Math Expression Extraction

This section documents the API for mathematical expression extraction tasks, providing various extractors to recognize and retrieve LaTeX math from documents.

## Overview

Math expression extraction in OmniDocs focuses on converting mathematical formulas and equations found in documents (e.g., academic papers, textbooks) into a machine-readable format, typically LaTeX. This enables further processing, rendering, or indexing of mathematical content.

## Available Extractors

### DonutExtractor

NAVER CLOVA Donut model for math/LaTeX extraction.

::: omnidocs.tasks.math_expression_extraction.extractors.donut.DonutExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

extractor = DonutExtractor()
result = extractor.extract("math_document.pdf")
print(f"Extracted LaTeX: {result.full_text[:200]}...")
```

### NougatExtractor

Facebook's Nougat model for LaTeX extraction from academic documents.

::: omnidocs.tasks.math_expression_extraction.extractors.nougat.NougatExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

extractor = NougatExtractor()
result = extractor.extract("academic_paper.pdf")
print(f"Extracted LaTeX: {result.full_text[:200]}...")
```

### SuryaMathExtractor

Surya-based mathematical expression extraction.

::: omnidocs.tasks.math_expression_extraction.extractors.surya_math.SuryaMathExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.math_expression_extraction.extractors.surya_math import SuryaMathExtractor

extractor = SuryaMathExtractor()
result = extractor.extract("math_image.png")
print(f"Extracted LaTeX: {result.full_text[:200]}...")
```

### UniMERNetExtractor

Universal Mathematical Expression Recognition Network.

::: omnidocs.tasks.math_expression_extraction.extractors.unimernet.UniMERNetExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.math_expression_extraction.extractors.unimernet import UniMERNetExtractor

extractor = UniMERNetExtractor()
result = extractor.extract("math_equation.png")
print(f"Extracted LaTeX: {result.full_text[:200]}...")
```

## MathOutput

The standardized output format for mathematical expression extraction results.

::: omnidocs.tasks.math_expression_extraction.base.MathOutput
    options:
      show_root_heading: true
      show_source: false

#### Key Properties

- `expressions` (List[MathExpression]): List of detected mathematical expressions.
- `full_text` (str): Combined LaTeX string of all expressions.
- `source_file` (str): Path to the processed file.

#### Key Methods

- `save_json(output_path)`: Save results to a JSON file.

## MathExpression

Represents a single extracted mathematical expression.

::: omnidocs.tasks.math_expression_extraction.base.MathExpression
    options:
      show_root_heading: true
      show_source: false

#### Attributes

- `latex` (str): The extracted LaTeX string.
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2].
- `confidence` (Optional[float]): Confidence score of the extraction.
- `page_number` (int): The page number where the expression is found.

## BaseMathExtractor

The abstract base class for all mathematical expression extractors.

::: omnidocs.tasks.math_expression_extraction.base.BaseMathExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - preprocess_input
        - postprocess_output

## Related Resources

- [Math Expression Overview](../tasks/math_extraction/overview.md)
- [Donut Tutorial](../../tasks/math_extraction/tutorials/donut.ipynb)
- [Nougat Tutorial](../../tasks/math_extraction/tutorials/nougat.ipynb)
- [SuryaMath Tutorial](../../tasks/math_extraction/tutorials/suryamath.ipynb)
- [UniMERNet Tutorial](../../tasks/math_extraction/tutorials/unimernet.ipynb)
- [Core Classes](../core.md)
- [Utilities](../utils.md)
