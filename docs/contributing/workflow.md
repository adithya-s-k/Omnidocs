# Implementation Workflow

This guide walks through the complete 6-phase workflow for adding new features to OmniDocs.

## Overview

```
Issue & Planning
    ↓
Experimentation (scripts/)
    ↓
Integration (Omnidocs/)
    ↓
Testing & Validation
    ↓
Pull Request & Review
    ↓
Version Release
```

---

## Phase 1: Issue & Planning

### Create GitHub Issue

Create a new issue with this template:

```markdown
**Title**: Add [Model/Task Name] Support

**Description**:
- **Task Type**: Text Extraction / Layout Analysis / OCR / etc.
- **Model**: [Model name]
- **Backends**: PyTorch / VLLM / MLX / API
- **Use Case**: [Brief description]

**References**:
- Model Card: [HuggingFace link]
- Paper: [arXiv link if applicable]

**Checklist**:
- [ ] Create implementation plan
- [ ] Experiment in scripts/
- [ ] Integrate into Omnidocs/
- [ ] Write tests
- [ ] Pass lint checks
- [ ] Create PR
```

### Read Design Documents

Before implementing ANYTHING, read these:

1. `IMPLEMENTATION_PLAN/BACKEND_ARCHITECTURE.md` - Backend system design
2. `IMPLEMENTATION_PLAN/DEVEX.md` - API design principles
3. `CLAUDE.md` - Development guide and standards

### Write Implementation Plan

Add a comment to your issue with:

```markdown
## Implementation Plan

### Architecture
- Single-backend or multi-backend?
- Which backends to support?
- Config class names?

### File Structure
```
omnidocs/tasks/text_extraction/
├── mymodel.py           # Main extractor
└── mymodel/             # Configs (if multi-backend)
    ├── pytorch.py
    ├── vllm.py
    └── api.py
```

### Dependencies
List all new packages to add

### Timeline
Estimate effort for each phase
```

---

## Phase 2: Experimentation (scripts/)

### Create Experiment Scripts

For GPU models, create `scripts/text_extract/modal_mymodel_pytorch.py`:

```python
import modal

app = modal.App("omnidocs-mymodel-test")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Base image (cached)
BASE_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install("torch", "transformers", "pillow")
    .env({"HF_HOME": "/data/.cache"})
)

# Model-specific layer
IMAGE = BASE_IMAGE.uv_pip_install("mymodel-package")

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

@app.function(image=IMAGE, gpu="A10G:1", volumes={"/data": volume}, secrets=[secret])
def test_inference():
    # Test code here
    pass

@app.local_entrypoint()
def main():
    test_inference.remote()
```

**Naming Convention**:
- `modal_{model}_{backend}.py` - GPU-based (PyTorch/VLLM)
- `litellm_{model}_{task}.py` - API-based (local)
- `mlx_{model}_{task}.py` - Apple Silicon (local)

### Run and Validate

```bash
cd scripts/text_extract/
modal run modal_mymodel_pytorch.py
```

**Validation Checklist**:
- [ ] Model loads successfully
- [ ] Inference produces expected output
- [ ] Performance is acceptable
- [ ] Error handling works
- [ ] Different input types work

### Document Findings

In the GitHub issue, comment with:
- Model size and memory requirements
- Inference speed (tokens/sec or pages/sec)
- Quality observations
- Recommended configurations

---

## Phase 3: Integration (Omnidocs/)

### Step 1: Determine Backend Support

Is this single-backend or multi-backend?

**Single-Backend** (e.g., DocLayoutYOLO):
- One config class in same file as extractor
- Use `config=` parameter

**Multi-Backend** (e.g., Qwen):
- Multiple config classes in subfolder
- Use `backend=` parameter with Union type

### Step 2: Create Config Classes

**Single-Backend** (in `omnidocs/tasks/text_extraction/mymodel.py`):

```python
from pydantic import BaseModel, Field

class MyModelConfig(BaseModel):
    """Configuration for MyModel."""

    model: str = Field(default="vendor/mymodel", description="Model ID")
    device: str = Field(default="cuda", description="Device")
    dtype: str = Field(default="bfloat16", description="Data type")

    class Config:
        extra = "forbid"
```

**Multi-Backend** (in `omnidocs/tasks/text_extraction/mymodel/pytorch.py`):

```python
class MyModelPyTorchConfig(BaseModel):
    """PyTorch backend config."""
    model: str = Field(..., description="Model ID")
    device: str = Field(default="cuda")
    torch_dtype: str = Field(default="bfloat16")
    class Config:
        extra = "forbid"
```

**Config Rules**:
- ✅ All parameters with `Field(...)`
- ✅ All parameters have descriptions
- ✅ Type hints for everything
- ✅ Validation rules (ge, le, Literal)
- ✅ `extra = "forbid"` to catch typos
- ✅ Pydantic docstring for class

### Step 3: Create Extractor Class

```python
from typing import Union, Optional
from .base import BaseTextExtractor
from .models import TextOutput

class MyModelTextExtractor(BaseTextExtractor):
    """MyModel text extractor."""

    def __init__(self, backend: MyModelBackendConfig):
        self.backend_config = backend
        self._backend = self._create_backend()

    def _create_backend(self):
        """Create backend based on config type."""
        if isinstance(self.backend_config, MyModelPyTorchConfig):
            from omnidocs.inference.pytorch import PyTorchInference
            return PyTorchInference(self.backend_config)
        # ... other backends

    def extract(self, image, output_format="markdown", **kwargs) -> TextOutput:
        """Extract text from image."""
        # Implementation
        pass
```

### Step 4: Update Exports

Edit `omnidocs/tasks/text_extraction/__init__.py`:

```python
from .mymodel import MyModelTextExtractor
from .mymodel import (
    MyModelPyTorchConfig,
    MyModelVLLMConfig,
    # ...
)

__all__ = [
    "MyModelTextExtractor",
    "MyModelPyTorchConfig",
    # ...
]
```

### Step 5: Add Dependencies

```bash
cd Omnidocs/
uv add --group pytorch mymodel-package
uv sync
```

---

## Phase 4: Testing & Validation

### Write Unit Tests

Create `Omnidocs/tests/tasks/text_extraction/test_mymodel.py`:

```python
import pytest
from omnidocs.tasks.text_extraction import MyModelTextExtractor
from omnidocs.tasks.text_extraction.mymodel import MyModelPyTorchConfig

class TestMyModelConfig:
    """Test config validation."""

    def test_valid_config(self):
        config = MyModelPyTorchConfig(model="vendor/mymodel")
        assert config.device == "cuda"

    def test_invalid_param(self):
        with pytest.raises(ValueError):
            MyModelPyTorchConfig(
                model="vendor/mymodel",
                invalid_param="value"  # Should raise error
            )

class TestMyModelExtractor:
    """Test extractor functionality."""

    @pytest.fixture
    def sample_image(self):
        from PIL import Image
        return Image.new("RGB", (800, 600))

    def test_extract_markdown(self, sample_image):
        config = MyModelPyTorchConfig(model="vendor/mymodel", device="cuda")
        extractor = MyModelTextExtractor(backend=config)
        result = extractor.extract(sample_image, output_format="markdown")

        assert result.format.value == "markdown"
        assert len(result.content) > 0
```

**Coverage Target**: >80%

### Integration Test

Create `scripts/text_extract_omnidocs/modal_mymodel_text_hf.py`:

```python
"""Test MyModel through Omnidocs package on Modal."""

@app.function(image=OMNIDOCS_IMAGE, gpu="A10G:1", ...)
def test_omnidocs_mymodel():
    from omnidocs.tasks.text_extraction import MyModelTextExtractor
    from omnidocs.tasks.text_extraction.mymodel import MyModelPyTorchConfig

    extractor = MyModelTextExtractor(
        backend=MyModelPyTorchConfig(model="vendor/mymodel", device="cuda")
    )

    result = extractor.extract(image, output_format="markdown")
    assert result.format.value == "markdown"
    return {"success": True, "length": len(result.content)}
```

### Lint Checks

```bash
cd Omnidocs/

# Format code
uv run black omnidocs/ tests/

# Type checking
uv run mypy omnidocs/

# Fix any issues before proceeding
```

---

## Phase 5: Pull Request

### Create Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/add-mymodel-support
```

### Commit Changes

```bash
git add Omnidocs/omnidocs/tasks/text_extraction/mymodel.py
git add Omnidocs/omnidocs/tasks/text_extraction/mymodel/
git add Omnidocs/tests/tasks/text_extraction/test_mymodel.py
git add Omnidocs/pyproject.toml
git add scripts/text_extract/modal_mymodel_pytorch.py
git add scripts/text_extract_omnidocs/test_mymodel.py

git commit -m "$(cat <<'EOF'
feat: add MyModel text extraction support

Adds MyModelTextExtractor with PyTorch backend:
- Complete text extraction to Markdown/HTML
- Pydantic config validation
- Integration with Document class
- Unit tests (>80% coverage)
- Modal deployment scripts

Testing:
- Unit tests passing
- Integration tests on Modal verified
- Black formatting applied
- Mypy type checks passing
EOF
)"
```

**Important**:
- NO `Co-Authored-By` attribution
- NO AI/Claude mentions
- Commits attributed to repository owner only

### Push and Create PR

```bash
git push origin feature/add-mymodel-support

gh pr create \
  --title "Add MyModel Text Extraction" \
  --body "Adds MyModel support with PyTorch backend.

## Changes
- MyModelTextExtractor class
- PyTorch configuration
- Unit and integration tests
- Modal deployment scripts

## Testing
- [x] Unit tests passing (>80%)
- [x] Integration tests on Modal
- [x] Lint checks (black, mypy)
- [x] Documentation updated"
```

### Iterate on Review

1. Address reviewer feedback
2. Push updates to same branch
3. Re-run lint and tests
4. Request re-review

---

## Phase 6: Version & Release

### Update Version

Edit `Omnidocs/pyproject.toml`:

```toml
[project]
name = "omnidocs"
version = "2.2.0"  # Increment MINOR for new feature
```

### Update Changelog

Edit `Omnidocs/CHANGELOG.md`:

```markdown
## [2.2.0] - 2026-02-15

### Added
- **MyModel Text Extraction**: PyTorch backend for efficient text extraction
  - Markdown and HTML output formats
  - Integration with Document class
  - Unit tests with >80% coverage
```

### Create Git Tag

```bash
git checkout main
git pull origin main
git tag -a v2.2.0 -m "Release v2.2.0: Add MyModel text extraction"
git push origin v2.2.0
```

### Build and Publish

```bash
cd Omnidocs/

# Build distribution
uv build

# Publish to PyPI
uv publish
```

### Create GitHub Release

```bash
gh release create v2.2.0 \
  --title "v2.2.0: MyModel Text Extraction" \
  --notes "Added MyModel support for efficient text extraction"
```

---

## Summary Checklist

### Planning
- [ ] GitHub issue created
- [ ] Design docs read
- [ ] Implementation plan written

### Experimentation
- [ ] Experiment script in scripts/
- [ ] Modal/local execution successful
- [ ] Findings documented

### Integration
- [ ] Config classes created
- [ ] Extractor class implemented
- [ ] __init__.py exports updated
- [ ] Dependencies added (uv add)

### Testing
- [ ] Unit tests >80% coverage
- [ ] Integration test in *_omnidocs/
- [ ] Modal test successful
- [ ] Lint checks passing (black, mypy)

### PR
- [ ] Feature branch created
- [ ] Changes committed
- [ ] PR created with description
- [ ] CI/CD checks pass
- [ ] Feedback addressed

### Release
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Git tag created
- [ ] Package published to PyPI
- [ ] GitHub release created

---

## Next: See [Adding Models](adding-models.md) for step-by-step model integration.
