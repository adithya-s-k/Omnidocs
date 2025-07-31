# Mathematical LaTeX Expression Extraction with OmniDocs

## Overview

OmniDocs provides multiple specialized extractors for LaTeX mathematical expression recognition. Each extractor is implemented as a standalone class that can be imported and used independently, following a consistent interface pattern.

## Available LaTeX Extractors

### 1. DonutExtractor
Document Understanding Transformer for mathematical expressions.

```python
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

# Initialize extractor
extractor = DonutExtractor(device='cuda', show_log=True)

# Extract from image
result = extractor.extract("path/to/math_equation_image.png")
print(result.expressions[0])  # Prints LaTeX code
```


### 4. NougatExtractor
Neural Optical Understanding for Academic Documents.

```python
from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

# Initialize Nougat extractor
extractor = NougatExtractor(device='cuda', show_log=True)

# Extract from academic paper image
result = extractor.extract("paper_with_equations.png")
print(result.expressions[0])  # Mathematical expressions in LaTeX
```

## Common Usage Patterns

### Basic Extraction Workflow
```python
# Choose your preferred extractor
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

# Initialize
extractor = DonutExtractor(device='cuda', show_log=True)

# Extract expressions
result = extractor.extract("equation_image.png")

# Access results
for i, expr in enumerate(result.expressions):
    print(f"Expression {i+1}: {expr}")
```

### Batch Processing Multiple Images
```python
import os
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

# Initialize extractor once (choose Donut, Nougat, SuryaMath, or UniMERNet)
extractor = DonutExtractor(device='cuda', show_log=True)

# Process multiple images in a folder
image_folder = "path/to/math_images/"
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        try:
            result = extractor.extract(image_path)
            results.append({
                'filename': filename,
                'expressions': result.expressions,
                'image_size': result.source_img_size
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Print all results
for item in results:
    print(f"\n{item['filename']}:")
    for expr in item['expressions']:
        print(f"  {expr}")
```

### Working with Different Input Types
```python
from PIL import Image
import numpy as np
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

extractor = DonutExtractor(device='cuda', show_log=True)

# From file path
result1 = extractor.extract("equation.png")

# From PIL Image
img = Image.open("equation.png")
result2 = extractor.extract(img)

# From numpy array (if supported)
img_array = np.array(img)
result3 = extractor.extract(img_array)

print("All methods produce LaTeX:", result1.expressions[0])
```
#!/usr/bin/env python3

from omnidocs.tasks.math_expression_extraction.extractors.unimernet_ocr import UniMERNetExtractor
import torch

print('Testing UniMERNet functionality...')
try:
    extractor = UniMERNetExtractor()
    print('✓ UniMERNet extractor initialized successfully')
    print(f'✓ Device: {extractor.device}')
    print(f'✓ Model: {type(extractor.model)}')
    print('UniMERNet is ready to use!')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()

## Extractor Comparison

| Extractor | Accuracy | Speed | Memory Usage | Best Use Case |
|-----------|----------|-------|--------------|---------------|
| **DonutExtractor** | High | Medium | Medium | General mathematical expressions |
| **NougatExtractor** | Very High | Slow | High | Academic papers, complex layouts |
| **SuryaMathExtractor** | High | Fast | Low | Lightweight, fast math extraction |
| **UniMERNetExtractor** | High | Fast | Medium | Universal math recognition |

## Configuration Options

### Device Selection
```python
# GPU processing (recommended)
extractor = DonutExtractor(device='cuda', show_log=True)

# CPU processing (slower but works without GPU)
extractor = DonutExtractor(device='cpu', show_log=True)

# Auto-detect best device
extractor = DonutExtractor(device=None, show_log=True)  # Auto-selects best available
```

### Logging and Debugging
```python
# Enable detailed logging
extractor = DonutExtractor(device='cuda', show_log=True)

# Disable logging for production
extractor = DonutExtractor(device='cuda', show_log=False)
```

### Model-Specific Parameters
```python
# Nougat with specific checkpoint
extractor = NougatExtractor(
    device='cuda',
    show_log=True,
    model_checkpoint='facebook/nougat-base'  # Specific model version
)
```

## Advanced Usage Examples

### Error Handling and Validation
```python
from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor

def safe_extract_latex(image_path):
    """Safely extract LaTeX with error handling."""
    try:
        extractor = DonutExtractor(device='cuda', show_log=False)
        result = extractor.extract(image_path)

        if result.expressions:
            return result.expressions[0]
        else:
            print(f"No expressions found in {image_path}")
            return None

    except ImportError as e:
        print(f"Model not available: {e}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Usage
latex_code = safe_extract_latex("my_equation.png")
if latex_code:
    print(f"Extracted: {latex_code}")
```

### Comparing Multiple Extractors
```python
from omnidocs.tasks.math_expression_extraction.extractors import (
    DonutExtractor, NougatExtractor, SuryaMathExtractor, UniMERNetExtractor
)

def compare_extractors(image_path):
    """Compare results from different extractors."""
    extractors = {
        'Donut': DonutExtractor,
        'Nougat': NougatExtractor,
        'SuryaMath': SuryaMathExtractor,
        'UniMERNet': UniMERNetExtractor
    }

    results = {}
    for name, ExtractorClass in extractors.items():
        try:
            extractor = ExtractorClass(device='cuda', show_log=False)
            result = extractor.extract(image_path)
            results[name] = result.expressions[0] if result.expressions else "No result"
        except Exception as e:
            results[name] = f"Error: {e}"

    return results

# Compare results
image_path = "complex_equation.png"
comparison = compare_extractors(image_path)

for extractor_name, result in comparison.items():
    print(f"{extractor_name}: {result}")
```

### Processing PDF Pages with Math
```python
import fitz  # PyMuPDF
from PIL import Image
from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

def extract_math_from_pdf(pdf_path, page_numbers=None):
    """Extract mathematical expressions from PDF pages."""
    extractor = NougatExtractor(device='cuda', show_log=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    results = []

    pages_to_process = page_numbers or range(len(doc))

    for page_num in pages_to_process:
        try:
            # Get page as image
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))

            # Extract expressions
            result = extractor.extract(img)

            results.append({
                'page': page_num + 1,
                'expressions': result.expressions,
                'image_size': result.source_img_size
            })

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")

    doc.close()
    return results

# Usage
pdf_results = extract_math_from_pdf("research_paper.pdf", [0, 1, 2])  # First 3 pages
for page_result in pdf_results:
    print(f"\nPage {page_result['page']}:")
    for expr in page_result['expressions']:
        print(f"  {expr}")
```

## Output Format

All extractors return a `LatexOutput` object with the following structure:

```python
class LatexOutput:
    expressions: List[str]        # List of LaTeX expressions
    source_img_size: Tuple[int, int]  # Original image dimensions (width, height)
```

### Accessing Results
```python
result = extractor.extract("equation.png")

# Get all expressions
all_expressions = result.expressions

# Get first expression
first_expr = result.expressions[0]

# Get image dimensions
width, height = result.source_img_size

# Check if any expressions were found
if result.expressions:
    print(f"Found {len(result.expressions)} expressions")
    for i, expr in enumerate(result.expressions):
        print(f"Expression {i+1}: {expr}")
else:
    print("No mathematical expressions detected")
```

## Installation Requirements

Each extractor has specific dependencies:

### DonutExtractor
```bash
pip install transformers torch pillow
```

### NougatExtractor
```bash
pip install nougat-ocr
```

### SuryaMathExtractor
```bash
pip install surya-ocr torch pillow
```

### UniMERNetExtractor
```bash
pip install unimer-net torch pillow
```

## Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory**
```python
# Use CPU instead
extractor = DonutExtractor(device='cpu', show_log=True)

# Or process smaller batches
```

**2. Model Download Issues**
```python
# Ensure internet connection for first-time model download
# Models are cached locally after first download
extractor = DonutExtractor(device='cuda', show_log=True)  # Will download if needed
```

**3. Import Errors**
```python
try:
    from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor
    extractor = DonutExtractor(device='cuda', show_log=True)
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install required packages")
```

**4. No Expressions Detected**
```python
result = extractor.extract("image.png")
if not result.expressions:
    print("Try:")
    print("1. Check image quality and resolution")
    print("2. Ensure mathematical content is clearly visible")
    print("3. Try a different extractor model")
```

## Best Practices

1. **Choose the Right Extractor**: 
   - Use Donut for general-purpose extraction
   - Use Nougat for academic papers
   - Use SuryaMath for lightweight, fast math extraction
   - Use UniMERNet for universal math recognition

2. **Optimize Performance**:
   - Use GPU when available (`device='cuda'`)
   - Initialize extractor once for batch processing
   - Process images in appropriate resolution

3. **Handle Errors Gracefully**:
   - Always wrap extraction in try-catch blocks
   - Check if expressions list is not empty
   - Validate LaTeX output if needed

4. **Image Quality**:
   - Use high-resolution images when possible
   - Ensure good contrast between text and background
   - Crop to focus on mathematical content when feasible