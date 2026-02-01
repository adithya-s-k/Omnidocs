# Adding New Models to OmniDocs

This guide provides step-by-step instructions for integrating a new model into OmniDocs.

## Before You Start

1. Read [Workflow](workflow.md) - Understand the 6-phase process
2. Read [IMPLEMENTATION_PLAN/BACKEND_ARCHITECTURE.md](../../IMPLEMENTATION_PLAN/BACKEND_ARCHITECTURE.md)
3. Verify the model you want to add doesn't already exist

## Step 1: Create Experiment Script

Create a standalone test script in `scripts/` to verify the model works.

### For GPU Models (PyTorch/VLLM)

Create `scripts/text_extract/modal_mymodel_pytorch.py`:

```python
import modal
from pathlib import Path

app = modal.App("test-mymodel")

# Standard CUDA configuration
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Base image (cached across scripts)
BASE_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision",
        "torchaudio",
        "transformers",
        "pillow",
        "numpy",
        "pydantic",
        "huggingface_hub",
        "accelerate",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)

# Model-specific dependencies
IMAGE = BASE_IMAGE.uv_pip_install("mymodel-package")

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def test_model():
    """Test model loading and inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    import requests
    from io import BytesIO

    MODEL_NAME = "vendor/mymodel"

    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # Load test image
    print("Loading test image...")
    response = requests.get("https://example.com/test.png")
    image = Image.open(BytesIO(response.content))

    # Run inference
    print("Running inference...")
    inputs = processor(text="Extract text", images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4096)

    result = processor.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(result)

    return {"success": True, "length": len(result)}

@app.local_entrypoint()
def main():
    result = test_model.remote()
    print(f"\nTest completed: {result['success']}")
```

### For API Models (Local)

Create `scripts/text_extract/litellm_mymodel_text.py`:

```python
import os
from PIL import Image
from openai import OpenAI
import base64
from io import BytesIO

def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"

# Initialize client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Test image
image = Image.open("test.png")

# Run inference
response = client.chat.completions.create(
    model="vendor/mymodel",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image)}},
                {"type": "text", "text": "Extract text in markdown format"}
            ]
        }
    ],
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

### Run and Validate

```bash
cd scripts/text_extract/

# GPU models
modal run modal_mymodel_pytorch.py

# API models (local)
python litellm_mymodel_text.py
```

**Checklist**:
- [ ] Model loads
- [ ] Inference runs
- [ ] Output is reasonable
- [ ] Error handling works

---

## Step 2: Decide: Single or Multi-Backend?

### Single-Backend Model

**When**: Model only works with one inference backend (e.g., YOLO-based models)

**File Structure**:
```
omnidocs/tasks/text_extraction/
├── mymodel.py           # Config + Extractor in same file
```

**Example**:
```python
from pydantic import BaseModel, Field

class MyModelConfig(BaseModel):
    """Config for MyModel (PyTorch only)."""
    device: str = Field(default="cuda")
    model_name: str = Field(default="vendor/mymodel")
    class Config:
        extra = "forbid"

class MyModelTextExtractor:
    def __init__(self, config: MyModelConfig):
        self.config = config
        # Load model

    def extract(self, image, output_format="markdown"):
        # Extraction logic
```

### Multi-Backend Model

**When**: Model can use multiple backends (PyTorch, VLLM, MLX, API)

**File Structure**:
```
omnidocs/tasks/text_extraction/
├── mymodel.py           # Main extractor class
└── mymodel/             # Backend configurations
    ├── __init__.py
    ├── pytorch.py       # MyModelPyTorchConfig
    ├── vllm.py          # MyModelVLLMConfig
    ├── mlx.py           # MyModelMLXConfig
    └── api.py           # MyModelAPIConfig
```

**Example**:
```python
from typing import Union

QwenBackendConfig = Union[
    MyModelPyTorchConfig,
    MyModelVLLMConfig,
    MyModelMLXConfig,
    MyModelAPIConfig,
]

class MyModelTextExtractor:
    def __init__(self, backend: MyModelBackendConfig):
        self.backend_config = backend
        self._backend = self._create_backend()

    def _create_backend(self):
        if isinstance(self.backend_config, MyModelPyTorchConfig):
            # Create PyTorch backend
        elif isinstance(self.backend_config, MyModelVLLMConfig):
            # Create VLLM backend
        # ...
```

---

## Step 3: Create Config Classes

### Single-Backend Config

```python
# omnidocs/tasks/text_extraction/mymodel.py

from pydantic import BaseModel, Field
from typing import Literal, Optional

class MyModelConfig(BaseModel):
    """Configuration for MyModel (PyTorch only)."""

    # Required
    model: str = Field(
        default="vendor/mymodel-8b",
        description="Model identifier"
    )

    # Optional with defaults
    device: str = Field(
        default="cuda",
        description="Device to run on (cuda/cpu)"
    )

    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Torch data type"
    )

    # Numeric with validation
    max_new_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum tokens to generate"
    )

    class Config:
        extra = "forbid"  # CRITICAL: Catch typos
```

### Multi-Backend Configs

```python
# omnidocs/tasks/text_extraction/mymodel/pytorch.py

class MyModelPyTorchConfig(BaseModel):
    """PyTorch backend for MyModel."""

    model: str = Field(..., description="Model ID")
    device: str = Field(default="cuda")
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="bfloat16")
    device_map: Optional[str] = Field(default="auto")

    class Config:
        extra = "forbid"

# omnidocs/tasks/text_extraction/mymodel/vllm.py

class MyModelVLLMConfig(BaseModel):
    """VLLM backend for MyModel."""

    model: str = Field(..., description="Model ID")
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int = Field(default=32768)

    class Config:
        extra = "forbid"

# omnidocs/tasks/text_extraction/mymodel/api.py

class MyModelAPIConfig(BaseModel):
    """API backend for MyModel."""

    model: str = Field(..., description="API model name")
    api_key: str = Field(..., description="API key")
    base_url: Optional[str] = Field(None, description="API base URL")

    class Config:
        extra = "forbid"
```

---

## Step 4: Create Extractor Class

### Single-Backend Extractor

```python
# omnidocs/tasks/text_extraction/mymodel.py

from .base import BaseTextExtractor
from .models import TextOutput
from PIL import Image

class MyModelTextExtractor(BaseTextExtractor):
    """MyModel text extractor (PyTorch only)."""

    def __init__(self, config: MyModelConfig):
        self.config = config
        self._load_model()

    def _load_model(self):
        """Load model with PyTorch."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float16,
            device_map=self.config.device,
        ).eval()

    def extract(
        self,
        image: Image.Image,
        output_format: str = "markdown",
        **kwargs
    ) -> TextOutput:
        """Extract text from image."""
        import torch

        prompt = self._get_prompt(output_format)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )

        raw_output = self.processor.decode(outputs[0], skip_special_tokens=True)

        return TextOutput(
            content=raw_output,
            format=output_format,
            raw_output=raw_output,
        )

    def _get_prompt(self, output_format: str) -> str:
        """Get extraction prompt."""
        if output_format == "markdown":
            return "Extract text in markdown format..."
        elif output_format == "html":
            return "Extract text in HTML format..."
        else:
            return "Extract all text..."
```

### Multi-Backend Extractor

```python
# omnidocs/tasks/text_extraction/mymodel.py

from typing import Union, Optional
from .base import BaseTextExtractor
from .models import TextOutput
from omnidocs.tasks.text_extraction.mymodel import (
    MyModelPyTorchConfig,
    MyModelVLLMConfig,
    MyModelAPIConfig,
)

MyModelBackendConfig = Union[
    MyModelPyTorchConfig,
    MyModelVLLMConfig,
    MyModelAPIConfig,
]

class MyModelTextExtractor(BaseTextExtractor):
    """MyModel text extractor with multi-backend support."""

    def __init__(self, backend: MyModelBackendConfig):
        self.backend_config = backend
        self._backend = self._create_backend()

    def _create_backend(self):
        """Create appropriate backend based on config type."""
        if isinstance(self.backend_config, MyModelPyTorchConfig):
            try:
                from omnidocs.inference.pytorch import PyTorchInference
            except ImportError:
                raise ImportError(
                    "PyTorch backend requires torch and transformers. "
                    "Install with: pip install omnidocs[pytorch]"
                )
            return PyTorchInference(self.backend_config)

        elif isinstance(self.backend_config, MyModelVLLMConfig):
            try:
                from omnidocs.inference.vllm import VLLMInference
            except ImportError:
                raise ImportError(
                    "VLLM backend requires vllm. "
                    "Install with: pip install omnidocs[vllm]"
                )
            return VLLMInference(self.backend_config)

        elif isinstance(self.backend_config, MyModelAPIConfig):
            try:
                from omnidocs.inference.api import APIInference
            except ImportError:
                raise ImportError(
                    "API backend requires openai. "
                    "Install with: pip install omnidocs[api]"
                )
            return APIInference(self.backend_config)

        else:
            raise TypeError(f"Unknown backend: {type(self.backend_config)}")

    def extract(
        self,
        image: Image.Image,
        output_format: str = "markdown",
        **kwargs
    ) -> TextOutput:
        """Extract text from image."""
        prompt = self._get_prompt(output_format)
        raw_output = self._backend.infer(image, prompt)

        return TextOutput(
            content=raw_output,
            format=output_format,
            raw_output=raw_output,
        )
```

---

## Step 5: Update Package Exports

Edit `omnidocs/tasks/text_extraction/__init__.py`:

```python
# Existing imports
from .base import BaseTextExtractor
from .models import TextOutput

# New imports - single backend
from .mymodel import MyModelTextExtractor, MyModelConfig

# OR - multi-backend
from .mymodel import MyModelTextExtractor
from .mymodel import (
    MyModelPyTorchConfig,
    MyModelVLLMConfig,
    MyModelAPIConfig,
)

__all__ = [
    # Base
    "BaseTextExtractor",
    "TextOutput",

    # Existing
    "QwenTextExtractor",
    "DotsOCRTextExtractor",

    # New
    "MyModelTextExtractor",
    "MyModelConfig",  # or MyModelPyTorchConfig, MyModelVLLMConfig, etc.
]
```

---

## Step 6: Add Dependencies

```bash
cd Omnidocs/

# Add model-specific package
uv add --group pytorch mymodel-package

# Sync virtual environment
uv sync
```

---

## Step 7: Write Tests

Create `Omnidocs/tests/tasks/text_extraction/test_mymodel.py`:

```python
import pytest
from omnidocs.tasks.text_extraction import MyModelTextExtractor, MyModelConfig

class TestMyModelConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        config = MyModelConfig(model="vendor/mymodel")
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"

    def test_invalid_dtype(self):
        with pytest.raises(ValueError):
            MyModelConfig(model="vendor/mymodel", torch_dtype="invalid")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            MyModelConfig(model="vendor/mymodel", invalid_param="value")

class TestMyModelExtractor:
    """Test extractor functionality."""

    @pytest.fixture
    def sample_image(self):
        from PIL import Image
        return Image.new("RGB", (800, 600), color="white")

    def test_extract_markdown(self, sample_image):
        config = MyModelConfig(model="vendor/mymodel", device="cpu")
        extractor = MyModelTextExtractor(config=config)

        result = extractor.extract(sample_image, output_format="markdown")

        assert result.format.value == "markdown"
        assert isinstance(result.content, str)
```

**Run tests**:
```bash
cd Omnidocs/
uv run pytest tests/tasks/text_extraction/test_mymodel.py -v
```

---

## Step 8: Integration Test

Create `scripts/text_extract_omnidocs/modal_mymodel_text_hf.py`:

```python
"""Test MyModel through Omnidocs package on Modal."""

import modal

app = modal.App("test-mymodel-omnidocs")

# Standard image with Omnidocs
OMNIDOCS_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install("torch", "transformers", "pillow")
    .run_commands("uv pip install -e /pkg/Omnidocs --system")
    .env({"HF_HOME": "/data/.cache"})
)

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")
pkg_mount = modal.Mount.from_local_dir(
    Path(__file__).parent.parent.parent / "Omnidocs",
    remote_path="/pkg/Omnidocs"
)

@app.function(
    image=OMNIDOCS_IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    mounts=[pkg_mount],
    timeout=600,
)
def test_omnidocs_mymodel():
    """Test MyModel through Omnidocs."""
    from omnidocs.tasks.text_extraction import MyModelTextExtractor
    from omnidocs.tasks.text_extraction.mymodel import MyModelPyTorchConfig
    from PIL import Image

    extractor = MyModelTextExtractor(
        config=MyModelPyTorchConfig(
            model="vendor/mymodel",
            device="cuda",
        )
    )

    test_image = Image.new("RGB", (800, 600), color="white")
    result = extractor.extract(test_image, output_format="markdown")

    assert result.format.value == "markdown"
    assert len(result.content) > 0

    return {"success": True, "length": len(result.content)}

@app.local_entrypoint()
def main():
    result = test_omnidocs_mymodel.remote()
    print(f"Test passed: {result}")
```

---

## Step 9: Lint Checks

```bash
cd Omnidocs/

# Format code
uv run black omnidocs/

# Type checking
uv run mypy omnidocs/
```

---

## Step 10: Submit PR

Follow [Workflow - Phase 5: Pull Request](workflow.md#phase-5-pull-request).

---

## Checklist

- [ ] Experiment script created and tested
- [ ] Backend type chosen (single or multi)
- [ ] Config classes written with validation
- [ ] Extractor class implemented
- [ ] Package exports updated
- [ ] Dependencies added (uv add)
- [ ] Unit tests >80% coverage
- [ ] Integration test passing on Modal
- [ ] Black formatting applied
- [ ] Mypy checks passing
- [ ] PR created and reviewed

---

## Next Steps

After PR approval and merge, follow [Workflow - Phase 6: Release](workflow.md#phase-6-version--release) to publish the new version.
