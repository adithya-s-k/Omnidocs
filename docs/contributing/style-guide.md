# Style Guide

OmniDocs follows consistent code and documentation standards to maintain quality and clarity.

## Code Style

### Type Hints

All public APIs must have complete type hints:

```python
# ✅ GOOD
def extract(self, image: Image.Image, output_format: str = "markdown") -> TextOutput:
    """Extract text from image."""
    pass

# ❌ BAD
def extract(self, image, output_format="markdown"):
    """Extract text from image."""
    pass
```

### Docstrings

Use Google-style docstrings for all public classes and methods:

```python
def extract(
    self,
    image: Image.Image,
    output_format: str = "markdown",
    include_layout: bool = False,
) -> TextOutput:
    """Extract text from document image.

    Converts document images to formatted text using the configured model
    and backend.

    Args:
        image: PIL Image to extract text from.
        output_format: Output format ("markdown" or "html").
        include_layout: Include layout information in output.

    Returns:
        TextOutput containing extracted content.

    Raises:
        ValueError: If output_format is invalid.
        RuntimeError: If model is not loaded.

    Example:
        >>> extractor = QwenTextExtractor(backend=config)
        >>> result = extractor.extract(image, output_format="markdown")
        >>> print(result.content)
    """
```

### Pydantic Configs

All config classes must follow these rules:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class MyConfig(BaseModel):
    """Clear docstring explaining purpose."""

    # Required parameters
    required_param: str = Field(
        ...,  # Indicates required
        description="Description of parameter"
    )

    # Optional with defaults
    optional_param: str = Field(
        default="default_value",
        description="Description"
    )

    # Enum/Literal for fixed choices
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Data type"
    )

    # Numeric with bounds
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )

    # Optional nullable
    cache_dir: Optional[str] = Field(
        default=None,
        description="Optional cache directory"
    )

    class Config:
        extra = "forbid"  # CRITICAL: Catch typos
```

**Rules**:
- ✅ All parameters use `Field(...)`
- ✅ All parameters have descriptions
- ✅ Type hints for everything
- ✅ Validation rules (ge, le, Literal)
- ✅ `extra = "forbid"` to catch mistakes
- ✅ Class-level docstring

### Error Handling

Provide informative error messages with installation instructions:

```python
# ✅ GOOD
if isinstance(self.backend_config, QwenPyTorchConfig):
    try:
        from omnidocs.inference.pytorch import PyTorchInference
    except ImportError:
        raise ImportError(
            "PyTorch backend requires torch and transformers. "
            "Install with: pip install omnidocs[pytorch]"
        )
    return PyTorchInference(self.backend_config)

# ❌ BAD
from omnidocs.inference.pytorch import PyTorchInference  # Hard requirement
```

---

## Testing Standards

### Test Structure

Use pytest fixtures and clear test names:

```python
import pytest
from unittest.mock import Mock, patch

class TestMyExtractor:
    """Test suite for MyExtractor."""

    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        from PIL import Image
        return Image.new("RGB", (800, 600), color="white")

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MyConfig(model="test-model", device="cpu")

    def test_valid_config(self, config):
        """Test that valid config initializes."""
        assert config.device == "cpu"
        assert config.model == "test-model"

    def test_invalid_param_raises(self):
        """Test that invalid params raise ValidationError."""
        with pytest.raises(ValueError):
            MyConfig(model="test", invalid_param="value")

    def test_extract_returns_output(self, sample_image, config):
        """Test that extract method returns Output."""
        extractor = MyExtractor(config=config)
        result = extractor.extract(sample_image)

        assert isinstance(result, MyOutput)
        assert len(result.content) > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU"
    )
    def test_extract_on_gpu(self, sample_image, config):
        """Test GPU extraction."""
        config.device = "cuda"
        extractor = MyExtractor(config=config)
        result = extractor.extract(sample_image)
        assert result is not None
```

### Test Coverage

Target >80% coverage for all code:

```bash
# Run tests with coverage
uv run pytest tests/ --cov=omnidocs --cov-report=html

# Check coverage report
open htmlcov/index.html
```

### Assertion Best Practices

```python
# ✅ GOOD - Specific assertions
assert result.format.value == "markdown"
assert len(result.content) > 100
assert result.content_length == len(result.content)

# ❌ BAD - Generic assertions
assert result is not None
assert bool(result)
```

---

## Documentation Standards

### Markdown Style

```markdown
# Main Title (H1)

Brief introductory paragraph explaining purpose.

## Section (H2)

Subsection content.

### Subsection (H3)

Use code blocks for examples:

\`\`\`python
# Python code example
result = extractor.extract(image)
\`\`\`

### Lists

- Bullet point 1
- Bullet point 2
  - Nested point
  - Another nested

1. Numbered item 1
2. Numbered item 2

### Tables

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

### Links

[Link text](../path/to/file.md)
```

### Code Examples

All code examples must be:
1. **Complete** - Can be copy-pasted and run
2. **Runnable** - All imports included
3. **Correct** - Verified against actual API
4. **Documented** - Comments explaining key parts

```markdown
### Example: Extract Text

Here's a complete example:

\`\`\`python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Load document
doc = Document.from_pdf("document.pdf")

# Create extractor
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
)

# Extract text from first page
result = extractor.extract(
    doc.get_page(0),
    output_format="markdown"
)

# Print result
print(result.content)
\`\`\`
```

### Documentation Requirements

Every user-facing documentation file must include:

1. **Title** (H1) - Clear, descriptive
2. **Summary** (1-2 paragraphs) - What is this doc about?
3. **Table of Contents** - For long docs (>2000 words)
4. **Main Content** - Progressive disclosure (simple → complex)
5. **Code Examples** - 3-5 complete, runnable examples
6. **Comparison Table** - When/what/why decisions
7. **Troubleshooting** - Common issues and solutions
8. **See Also** - Links to related docs
9. **YAML Frontmatter** - For AI-friendly parsing

```markdown
---
title: "Task Name"
description: "Short description"
category: "guides"
difficulty: "intermediate"
time_estimate: "15 minutes"
keywords: ["text extraction", "markdown", "pdf"]
---

# Task Name

Brief intro (1-2 sentences).

## When to Use This Guide

## Quick Example

## Main Content

## Advanced Features

## Troubleshooting

## Next Steps
```

---

## Git & Commit Standards

### Branch Naming

```
feature/add-qwen-support
bugfix/fix-memory-leak
docs/update-installation-guide
refactor/simplify-backend-system
```

### Commit Messages

Use imperative mood, no capitals, no periods:

```
# ✅ GOOD
feat: add Qwen text extraction support
fix: prevent GPU memory leak in batch processing
docs: update installation guide
refactor: simplify backend selection logic

# ❌ BAD
Added Qwen support
Fix memory leak
Update docs
Refactored backend code
```

### Commit Content

- One logical change per commit
- All tests passing
- Code properly formatted
- No debugging code or comments

---

## Python Code Formatting

### Use Black

```bash
uv run black omnidocs/ tests/
```

Black settings (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]
```

### Use MyPy for Type Checking

```bash
uv run mypy omnidocs/
```

MyPy settings (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
disallow_incomplete_defs = true
```

### Import Ordering

Use isort (configured in pyproject.toml):

```python
# Standard library
import os
import sys
from pathlib import Path

# Third party
import numpy as np
import torch
from transformers import AutoModel
from pydantic import BaseModel

# Local
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
```

---

## Quality Checklist

Before submitting code:

- [ ] All tests pass (`uv run pytest`)
- [ ] Coverage >80% (`uv run pytest --cov`)
- [ ] Code formatted (`uv run black omnidocs/`)
- [ ] Type checked (`uv run mypy omnidocs/`)
- [ ] No hard-coded paths or secrets
- [ ] Docstrings complete
- [ ] Error messages helpful
- [ ] Examples runnable
- [ ] No debug print statements
- [ ] Git history clean

---

## Common Patterns

### Backend Selection Pattern

```python
def _create_backend(self):
    """Create appropriate backend."""
    if isinstance(self.backend_config, MyModelPyTorchConfig):
        from omnidocs.inference.pytorch import PyTorchInference
        return PyTorchInference(self.backend_config)
    # ... other backends
    else:
        raise TypeError(f"Unknown backend: {type(self.backend_config)}")
```

### Config Validation Pattern

```python
class MyConfig(BaseModel):
    param: str = Field(..., description="...")

    @validator("param")
    def validate_param(cls, v):
        if v not in ["valid1", "valid2"]:
            raise ValueError(f"Invalid param: {v}")
        return v

    class Config:
        extra = "forbid"
```

### Test Fixture Pattern

```python
@pytest.fixture
def config(self):
    """Create test config."""
    return MyConfig(
        model="test-model",
        device="cpu",  # Use CPU for tests
    )

@pytest.fixture
def extractor(self, config):
    """Create test extractor."""
    return MyExtractor(config=config)
```

---

## When in Doubt

- Check existing code in `/omnidocs/` - follow established patterns
- Read [Concepts](../concepts/) - understand architecture
- Check tests in `/tests/` - see testing patterns
- Run `black` and `mypy` - follow their output

---

**Questions?** Open an issue or check [CLAUDE.md](../../CLAUDE.md) for development guide.
