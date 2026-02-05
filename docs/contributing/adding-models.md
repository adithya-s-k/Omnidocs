# Adding New Models to OmniDocs

This guide walks through the complete end-to-end process of adding a new model to OmniDocs, from creating an issue to merging a PR. We use **MinerU VL** as a real-world example throughout.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Issue & Planning                                      │
│  - Create GitHub issue                                          │
│  - Read design docs                                             │
│  - Write implementation plan                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Experimentation (scripts/)                            │
│  - Create standalone test script                                │
│  - Run on Modal (GPU) or locally (MLX/API)                      │
│  - Validate model works and document findings                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Integration (omnidocs/)                               │
│  - Create config classes                                        │
│  - Implement extractor class                                    │
│  - Update exports                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Testing                                               │
│  - Write unit tests                                             │
│  - Create integration test runners                              │
│  - Run tests on Modal                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: Lint & CI                                             │
│  - Run ruff format                                              │
│  - Run ruff check                                               │
│  - Verify CI workflows pass                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: Pull Request                                          │
│  - Create feature branch                                        │
│  - Commit changes                                               │
│  - Create PR and iterate on review                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Issue & Planning

### 1.1 Create GitHub Issue

Create a new issue using this template:

```markdown
## Add [Model Name] Support

### Description
- **Task Type**: Text Extraction / Layout Analysis / OCR
- **Model**: [Model name and HuggingFace link]
- **Backends**: PyTorch, VLLM, MLX, API
- **Model Size**: [Parameters and VRAM requirements]

### Use Case
[Why is this model useful? What does it do well?]

### References
- Model Card: [HuggingFace link]
- Paper: [arXiv link if applicable]
- Original Repo: [GitHub link]

### Implementation Checklist
- [ ] Create experiment scripts in `scripts/`
- [ ] Integrate into `omnidocs/tasks/`
- [ ] Write unit tests
- [ ] Write integration test runners
- [ ] Pass lint checks
- [ ] Create PR
```

??? example "Real Example: MinerU VL Issue #42"
    ```markdown
    ## Add MinerU VL Support

    ### Description
    - **Task Type**: Text Extraction, Layout Analysis
    - **Model**: opendatalab/MinerU2.5-2509-1.2B
    - **Backends**: PyTorch, VLLM, MLX, API
    - **Model Size**: 1.2B params, 3-4GB VRAM

    ### Use Case
    MinerU VL excels at layout-aware document extraction with specialized
    table (OTSL format) and equation (LaTeX) recognition.

    ### References
    - Model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
    - Utils: https://github.com/opendatalab/mineru-vl-utils
    ```

### 1.2 Read Design Documents

Before implementing, read these files:

1. **`CLAUDE.md`** - Development standards and patterns
2. **`IMPLEMENTATION_PLAN/BACKEND_ARCHITECTURE.md`** - Backend system design
3. **Existing model implementations** - Look at similar models in `omnidocs/tasks/`

### 1.3 Write Implementation Plan

Add a comment to your issue with your plan:

```markdown
## Implementation Plan

### Architecture Decision
- **Single-backend** vs **Multi-backend**: Multi-backend (PyTorch, VLLM, MLX, API)
- **Reason**: Model supports multiple inference engines

### File Structure
```
omnidocs/tasks/text_extraction/mineruvl/
├── __init__.py          # Exports
├── extractor.py         # Main MinerUVLTextExtractor
├── pytorch.py           # MinerUVLTextPyTorchConfig
├── vllm.py              # MinerUVLTextVLLMConfig
├── mlx.py               # MinerUVLTextMLXConfig
├── api.py               # MinerUVLTextAPIConfig
└── utils.py             # Shared utilities
```

### Dependencies
- `qwen-vl-utils` (for image processing)
- Adapted code from `mineru-vl-utils` (AGPL-3.0 licensed)
```

---

## Phase 2: Experimentation

Create standalone scripts to validate the model works before integrating.

### 2.1 Create Experiment Script

Scripts go in `scripts/` organized by task:

```
scripts/
├── text_extract/              # Raw model experiments
│   ├── modal_mineruvl_pytorch.py
│   ├── modal_mineruvl_vllm.py
│   └── mlx_mineruvl_text.py
└── text_extract_omnidocs/     # Integration test runners
    └── modal_mineruvl_text_hf.py
```

#### PyTorch/VLLM Script (Modal)

```python
"""
Experiment: MinerU VL Text Extraction with PyTorch on Modal

Usage:
    modal run scripts/text_extract/modal_mineruvl_pytorch.py
"""

import modal
from pathlib import Path

# CUDA configuration (use these exact versions)
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Build image in layers for caching
IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    # Base dependencies (cached across scripts)
    .uv_pip_install(
        "torch",
        "transformers==4.57.6",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "accelerate",
    )
    # Model-specific dependencies
    .uv_pip_install("qwen-vl-utils")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)

app = modal.App("experiment-mineruvl-pytorch")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")


@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def test_mineruvl_pytorch():
    """Test MinerU VL with PyTorch backend."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    print("Creating test image...")
    image = Image.new("RGB", (800, 600), "white")

    print("Running inference...")
    messages = [
        {"role": "system", "content": "You are a document parser."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "[layout]"}
        ]}
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = inputs.to(device=model.device, dtype=model.dtype)

    output_ids = model.generate(**inputs, max_new_tokens=4096)
    result = processor.decode(output_ids[0], skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result[:500])

    return {"success": True, "length": len(result)}


@app.local_entrypoint()
def main():
    result = test_mineruvl_pytorch.remote()
    print(f"\nTest completed: {result}")
```

#### MLX Script (Local)

```python
"""
Experiment: MinerU VL Text Extraction with MLX

Usage:
    uv run python scripts/text_extract/mlx_mineruvl_text.py
"""

from PIL import Image
from mlx_vlm import load, generate

MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"

print("Loading model...")
model, processor = load(MODEL_NAME)

print("Creating test image...")
image = Image.new("RGB", (800, 600), "white")

print("Running inference...")
messages = [
    {"role": "system", "content": "You are a document parser."},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "[layout]"}
    ]}
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model=model, processor=processor, prompt=prompt, image=image, max_tokens=4096)

print("\n" + "=" * 60)
print("RESULT:")
print("=" * 60)
print(response.text[:500])
```

### 2.2 Run and Validate

```bash
# GPU models (Modal)
modal run scripts/text_extract/modal_mineruvl_pytorch.py
modal run scripts/text_extract/modal_mineruvl_vllm.py

# MLX (local Apple Silicon)
uv run python scripts/text_extract/mlx_mineruvl_text.py
```

**Validation Checklist:**

- [ ] Model loads successfully
- [ ] Inference produces reasonable output
- [ ] Memory usage is acceptable
- [ ] Different input types work (images, PDFs)
- [ ] Error handling is graceful

### 2.3 Document Findings

Comment on your GitHub issue with results:

```markdown
## Experiment Results

### Performance
- Load time: ~4s (PyTorch), ~15s (VLLM)
- Inference: 3-6s per page
- VRAM: 3-4GB

### Observations
- Excellent table extraction (OTSL format)
- Good equation recognition (LaTeX output)
- Two-step extraction: layout detection → content recognition

### Recommended Config
- PyTorch: `float16`, `sdpa` attention
- VLLM: `gpu_memory_utilization=0.85`, `enforce_eager=True`
```

---

## Phase 3: Integration

### 3.1 Decide: Single vs Multi-Backend

| Type | When to Use | Example |
|------|-------------|---------|
| **Single-backend** | Model only works with one backend | DocLayoutYOLO (PyTorch only) |
| **Multi-backend** | Model supports multiple engines | MinerU VL, Qwen |

**Multi-backend structure:**

```
omnidocs/tasks/text_extraction/mineruvl/
├── __init__.py          # Exports all configs and extractor
├── extractor.py         # Main MinerUVLTextExtractor class
├── pytorch.py           # MinerUVLTextPyTorchConfig
├── vllm.py              # MinerUVLTextVLLMConfig
├── mlx.py               # MinerUVLTextMLXConfig
├── api.py               # MinerUVLTextAPIConfig
└── utils.py             # Shared utilities (prompts, parsing, etc.)
```

### 3.2 Create Config Classes

Each backend gets its own config file with Pydantic validation.

```python
# omnidocs/tasks/text_extraction/mineruvl/pytorch.py

from pydantic import BaseModel, Field
from typing import Literal, Optional

class MinerUVLTextPyTorchConfig(BaseModel):
    """PyTorch backend configuration for MinerU VL text extraction."""

    model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="HuggingFace model identifier",
    )
    device: str = Field(
        default="cuda",
        description="Device to run on (cuda, cpu, auto)",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="float16",
        description="Torch data type for model weights",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention 2 (requires flash-attn). Uses SDPA by default.",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map for model parallelism",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code from HuggingFace",
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum tokens to generate",
    )
    layout_image_size: tuple = Field(
        default=(1036, 1036),
        description="Image size for layout detection",
    )

    class Config:
        extra = "forbid"  # CRITICAL: Catch typos in config
```

**Config Rules:**

- ✅ Use `Field()` for all parameters with descriptions
- ✅ Add type hints for everything
- ✅ Use `Literal` for constrained choices
- ✅ Add validation (`ge`, `le`, etc.)
- ✅ Set `extra = "forbid"` to catch typos
- ✅ Provide sensible defaults

### 3.3 Create Extractor Class

```python
# omnidocs/tasks/text_extraction/mineruvl/extractor.py

from typing import TYPE_CHECKING, List, Literal, Union
from PIL import Image

from ..base import BaseTextExtractor
from ..models import TextOutput, OutputFormat

if TYPE_CHECKING:
    from .pytorch import MinerUVLTextPyTorchConfig
    from .vllm import MinerUVLTextVLLMConfig
    from .mlx import MinerUVLTextMLXConfig
    from .api import MinerUVLTextAPIConfig

MinerUVLTextBackendConfig = Union[
    "MinerUVLTextPyTorchConfig",
    "MinerUVLTextVLLMConfig",
    "MinerUVLTextMLXConfig",
    "MinerUVLTextAPIConfig",
]


class MinerUVLTextExtractor(BaseTextExtractor):
    """
    MinerU VL text extractor with layout-aware extraction.

    Supports multiple backends:
    - PyTorch (HuggingFace Transformers)
    - VLLM (high-throughput GPU)
    - MLX (Apple Silicon)
    - API (VLLM OpenAI-compatible server)

    Example:
        ```python
        from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        extractor = MinerUVLTextExtractor(
            backend=MinerUVLTextPyTorchConfig(device="cuda")
        )
        result = extractor.extract(image)
        print(result.content)
        ```
    """

    def __init__(self, backend: MinerUVLTextBackendConfig):
        """Initialize with backend configuration."""
        self.backend_config = backend
        self._client = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load model based on backend config type."""
        config_type = type(self.backend_config).__name__

        if config_type == "MinerUVLTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "MinerUVLTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "MinerUVLTextMLXConfig":
            self._load_mlx_backend()
        elif config_type == "MinerUVLTextAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(f"Unknown backend config: {config_type}")

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        config = self.backend_config

        # Determine device and dtype
        device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.float16 if device == "cuda" else torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # Load model
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": dtype,
        }
        if device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2" if config.use_flash_attention else "sdpa"
        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        model = Qwen2VLForConditionalGeneration.from_pretrained(config.model, **model_kwargs)
        if not config.device_map:
            model = model.to(device)
        model = model.eval()

        processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)

        self._client = _TransformersClient(model, processor, config.max_new_tokens)
        self._layout_size = config.layout_image_size

    # ... other backend loaders ...

    def extract(
        self,
        image: Union[Image.Image, str],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text from image.

        Args:
            image: Input image (PIL Image or file path)
            output_format: Output format ('html' or 'markdown')

        Returns:
            TextOutput with extracted content
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        # Implementation...
        pass
```

### 3.4 Update Exports

```python
# omnidocs/tasks/text_extraction/mineruvl/__init__.py

from .api import MinerUVLTextAPIConfig
from .extractor import MinerUVLTextExtractor
from .mlx import MinerUVLTextMLXConfig
from .pytorch import MinerUVLTextPyTorchConfig
from .vllm import MinerUVLTextVLLMConfig

__all__ = [
    "MinerUVLTextExtractor",
    "MinerUVLTextPyTorchConfig",
    "MinerUVLTextVLLMConfig",
    "MinerUVLTextMLXConfig",
    "MinerUVLTextAPIConfig",
]
```

```python
# omnidocs/tasks/text_extraction/__init__.py

# Add new imports
from .mineruvl import MinerUVLTextExtractor

__all__ = [
    # ... existing exports ...
    "MinerUVLTextExtractor",
]
```

---

## Phase 4: Testing

### 4.1 Write Unit Tests

Create `tests/tasks/text_extraction/test_mineruvl.py`:

```python
"""Unit tests for MinerU VL text extraction."""

import pytest
from PIL import Image


class TestMinerUVLTextPyTorchConfig:
    """Test PyTorch config validation."""

    def test_default_config(self):
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        config = MinerUVLTextPyTorchConfig()
        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.device == "cuda"
        assert config.torch_dtype == "float16"
        assert config.use_flash_attention is False

    def test_custom_config(self):
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        config = MinerUVLTextPyTorchConfig(
            device="cpu",
            torch_dtype="float32",
            max_new_tokens=2048,
        )
        assert config.device == "cpu"
        assert config.torch_dtype == "float32"
        assert config.max_new_tokens == 2048

    def test_extra_fields_forbidden(self):
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValueError):
            MinerUVLTextPyTorchConfig(invalid_param="value")

    def test_invalid_dtype(self):
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValueError):
            MinerUVLTextPyTorchConfig(torch_dtype="invalid")


class TestMinerUVLTextMLXConfig:
    """Test MLX config validation."""

    def test_default_config(self):
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        config = MinerUVLTextMLXConfig()
        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.max_tokens == 4096


class TestMinerUVLLayoutPyTorchConfig:
    """Test layout detector config."""

    def test_default_config(self):
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

        config = MinerUVLLayoutPyTorchConfig()
        assert config.device == "cuda"
        assert config.use_flash_attention is False
```

### 4.2 Create Integration Test Runners

Create `tests/runners/modal_runner.py`:

```python
"""Modal integration test runner for MinerU VL."""

import modal
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent

# CUDA configuration
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)

app = modal.App("test-mineruvl-omnidocs")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")


@app.function(
    image=IMAGE,
    gpu="A10G:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_mineruvl_text_pytorch():
    """Test MinerU VL text extraction through Omnidocs."""
    from PIL import Image
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

    # Create test image
    image = Image.new("RGB", (800, 600), "white")

    # Initialize extractor
    extractor = MinerUVLTextExtractor(
        backend=MinerUVLTextPyTorchConfig(
            device="cuda",
            torch_dtype="float16",
            use_flash_attention=False,
        )
    )

    # Run extraction
    result = extractor.extract(image, output_format="markdown")

    # Validate
    assert result.format.value == "markdown"
    assert isinstance(result.content, str)

    print(f"✅ Test passed! Content length: {len(result.content)}")
    return {"success": True, "length": len(result.content)}


@app.local_entrypoint()
def main():
    result = test_mineruvl_text_pytorch.remote()
    print(f"\nTest result: {result}")
```

### 4.3 Run Tests

```bash
cd Omnidocs/

# Run unit tests (fast, no GPU needed)
uv run pytest tests/tasks/text_extraction/test_mineruvl.py -v

# Run all unit tests
uv run pytest tests/ -v -m "not slow"

# Run integration tests on Modal
modal run tests/runners/modal_runner.py

# Run MLX tests locally
uv run python tests/runners/local_runner.py
```

---

## Phase 5: Lint & CI

OmniDocs uses GitHub Actions for CI/CD. Before creating a PR, ensure your code passes all checks.

### 5.1 Lint Checks

The CI runs these checks on every PR:

```yaml
# .github/workflows/lint.yml
- name: Run Ruff check
  run: ruff check --output-format=github .

- name: Run Ruff format check
  run: ruff format --check .
```

**Run locally:**

```bash
cd Omnidocs/

# Check for lint errors
uv run ruff check .

# Auto-fix lint errors
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Auto-format code
uv run ruff format .
```

### 5.2 Test Checks

The CI runs tests on Python 3.10 and 3.11:

```yaml
# .github/workflows/test.yml
- name: Run tests (non-slow)
  run: uv run pytest tests/ -v -m "not slow"
```

**Run locally:**

```bash
# Run same tests as CI
uv run pytest tests/ -v -m "not slow"

# Run with coverage
uv run pytest tests/ -v --cov=omnidocs --cov-report=term-missing
```

### 5.3 Common Lint Issues

| Issue | Fix |
|-------|-----|
| `F401: imported but unused` | Remove unused import or add to `__all__` |
| `E501: line too long` | Break line or configure max-line-length |
| `I001: import order` | Run `ruff check --fix` |
| `F821: undefined name` | Add missing import |

**Example fix:**

```bash
# Before
$ uv run ruff check omnidocs/tasks/text_extraction/mineruvl/
omnidocs/tasks/text_extraction/mineruvl/extractor.py:15:1: F401 `typing.Optional` imported but unused

# Fix
$ uv run ruff check --fix omnidocs/tasks/text_extraction/mineruvl/
Found 1 error (1 fixed, 0 remaining).
```

---

## Phase 6: Pull Request

### 6.1 Create Feature Branch

```bash
git checkout master
git pull origin master
git checkout -b feature/add-mineruvl-support
```

### 6.2 Stage and Commit

```bash
# Stage specific files (avoid staging unnecessary files)
git add omnidocs/tasks/text_extraction/mineruvl/
git add omnidocs/tasks/layout_extraction/mineruvl/
git add omnidocs/tasks/text_extraction/__init__.py
git add omnidocs/tasks/layout_extraction/__init__.py
git add tests/tasks/text_extraction/test_mineruvl.py
git add tests/runners/

# Check what's staged
git status

# Commit with descriptive message
git commit -m "$(cat <<'EOF'
Add MinerU VL text extraction and layout detection

- MinerUVLTextExtractor with PyTorch, VLLM, MLX, API backends
- MinerUVLLayoutDetector with same backend support
- Two-step extraction: layout detection → content recognition
- OTSL table format and LaTeX equation support
- Unit tests for config validation
- Modal integration test runners

Closes #42
EOF
)"
```

**Important:**

- ❌ NO `Co-Authored-By` attribution
- ❌ NO AI/Claude mentions in commits
- ✅ Reference the issue number (`Closes #42`)

### 6.3 Push and Create PR

```bash
# Push branch
git push origin feature/add-mineruvl-support

# Create PR
gh pr create \
  --title "Add MinerU VL text extraction and layout detection" \
  --body "$(cat <<'EOF'
## Summary
Adds MinerU VL support for layout-aware document extraction.

## Changes
- `MinerUVLTextExtractor` - Text extraction with 4 backends
- `MinerUVLLayoutDetector` - Layout detection with 4 backends
- Two-step extraction pipeline (layout → content)
- Specialized table (OTSL) and equation (LaTeX) recognition

## Testing
- [x] Unit tests passing
- [x] Modal integration tests passing
- [x] MLX local tests passing
- [x] Ruff lint checks passing

## Checklist
- [x] Code follows project style guide
- [x] Tests added for new functionality
- [x] Documentation updated

Closes #42
EOF
)"
```

### 6.4 Monitor CI

After creating the PR, GitHub Actions will run:

1. **Lint** - Ruff check and format
2. **Test** - pytest on Python 3.10 and 3.11

If any checks fail:

```bash
# Fix lint issues
uv run ruff check --fix .
uv run ruff format .

# Run tests locally
uv run pytest tests/ -v -m "not slow"

# Push fixes
git add .
git commit -m "Fix lint issues"
git push
```

### 6.5 Address Review Feedback

1. Read reviewer comments
2. Make requested changes
3. Push to the same branch
4. Request re-review

---

## Summary Checklist

### Phase 1: Planning
- [ ] GitHub issue created with template
- [ ] Design docs read (CLAUDE.md, BACKEND_ARCHITECTURE.md)
- [ ] Implementation plan written and commented

### Phase 2: Experimentation
- [ ] Experiment scripts in `scripts/`
- [ ] Modal tests passing (GPU backends)
- [ ] Local tests passing (MLX/API)
- [ ] Findings documented in issue

### Phase 3: Integration
- [ ] Config classes with Pydantic validation
- [ ] Extractor class with multi-backend support
- [ ] `__init__.py` exports updated
- [ ] Shared utilities extracted

### Phase 4: Testing
- [ ] Unit tests for config validation
- [ ] Modal integration test runner
- [ ] Local test runner (MLX)
- [ ] All tests passing

### Phase 5: Lint & CI
- [ ] `ruff check` passes
- [ ] `ruff format --check` passes
- [ ] `pytest` passes locally

### Phase 6: PR
- [ ] Feature branch created
- [ ] Changes committed (no AI attribution)
- [ ] PR created with description
- [ ] CI checks passing
- [ ] Review feedback addressed

---

## Next Steps

After PR is merged:

1. **Update documentation** - Add model to `docs/usage/models/`
2. **Update mkdocs.yml** - Add to navigation
3. **Version bump** - Update `pyproject.toml` version
4. **Release** - Create git tag and publish to PyPI

See [Workflow](workflow.md) for full release process.
