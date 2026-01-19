# OmniDocs - Backend Architecture

> **Status**: ✅ Design Complete
> **Last Updated**: January 20, 2026
> **Version**: 2.0.0

---

## Overview

OmniDocs supports 4 inference backends:

| Backend | Use Case | Platform | Key Dependencies |
|---------|----------|----------|------------------|
| **PyTorch** | Default local inference | CPU/GPU | torch, transformers |
| **VLLM** | High-throughput serving | GPU only | vllm |
| **MLX** | Apple Silicon optimization | macOS M1/M2/M3+ | mlx, mlx-lm |
| **API** | Hosted models | Cloud | litellm |

---

## Core Architecture Principles

### Separation of Concerns: `__init__` vs `.extract()`

OmniDocs maintains a clear separation between model initialization and runtime parameters:

**`__init__` (via config)** - Model Setup & Verification
- Which model to use
- Which backend (PyTorch/VLLM/MLX/API)
- Model loading settings (device, dtype, quantization)
- Download and cache paths
- Model verification and validation

**`.extract()` - Runtime Task Parameters**
- Output format (markdown/html)
- Custom prompts
- Task-specific options (include_layout, custom_labels)
- Per-call inference settings

**Example**:
```python
# Init: Model setup (happens once)
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen2-VL-7B",  # Which model
        device="cuda",              # Where to run
        torch_dtype="bfloat16",     # How to load
    )
)

# Extract: Runtime params (can vary per call)
result1 = extractor.extract(image1, output_format="markdown")
result2 = extractor.extract(image2, output_format="html", custom_prompt="...")
```

---

## Design Principle: Model-Specific Configs

Each model defines its own config classes for supported backends. This provides:

1. **IDE Autocomplete** - Only relevant parameters shown
2. **Type Safety** - Pydantic validation at config creation
3. **Clear Discoverability** - Config exists = backend supported
4. **No Abstraction Leakage** - Each backend can have unique parameters

---

## Config Class Structure

### Single-Backend Model

Models with only one backend (e.g., DocLayoutYOLO = PyTorch only):

```python
# omnidocs/tasks/layout_analysis/doc_layout_yolo.py

from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image

class DocLayoutYOLOConfig(BaseModel):
    """Configuration for DocLayoutYOLO model."""

    device: str = Field(default="cuda", description="Device to run on")
    model_path: Optional[str] = Field(default=None, description="Custom model weights")
    img_size: int = Field(default=1024, description="Input image size")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    class Config:
        extra = "forbid"  # Raise error on unknown params


class DocLayoutYOLO:
    """DocLayout-YOLO layout detector. PyTorch only."""

    def __init__(self, config: DocLayoutYOLOConfig):
        self.config = config
        self._load_model()

    def _load_model(self):
        """Load model with PyTorch."""
        import torch
        # Load model...

    def extract(self, image: Image.Image) -> LayoutOutput:
        """Run layout detection."""
        # Inference...
        pass
```

### Multi-Backend Model

Models with multiple backends (e.g., Qwen = PyTorch, VLLM, MLX, API):

```python
# omnidocs/tasks/text_extraction/qwen.py

from typing import Union
from PIL import Image

# Import all backend configs
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
)

# Union type for all supported backends
QwenBackendConfig = Union[
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
]


class QwenTextExtractor:
    """Qwen VLM text extractor. Supports PyTorch, VLLM, MLX, API backends."""

    def __init__(self, backend: QwenBackendConfig):
        self.backend_config = backend
        self._backend = self._create_backend()

    def _create_backend(self):
        """Create appropriate backend based on config type."""
        if isinstance(self.backend_config, QwenPyTorchConfig):
            from omnidocs.inference.pytorch import PyTorchInference
            return PyTorchInference(self.backend_config)

        elif isinstance(self.backend_config, QwenVLLMConfig):
            from omnidocs.inference.vllm import VLLMInference
            return VLLMInference(self.backend_config)

        elif isinstance(self.backend_config, QwenMLXConfig):
            from omnidocs.inference.mlx import MLXInference
            return MLXInference(self.backend_config)

        elif isinstance(self.backend_config, QwenAPIConfig):
            from omnidocs.inference.api import APIInference
            return APIInference(self.backend_config)

        else:
            raise TypeError(f"Unknown backend config: {type(self.backend_config)}")

    def extract(
        self,
        image: Image.Image,
        output_format: str = "markdown",
        include_layout: bool = False,
        custom_prompt: Optional[str] = None,
    ) -> TextOutput:
        """
        Extract text from image.

        Args:
            image: PIL Image
            output_format: "markdown" or "html"
            include_layout: Include layout information
            custom_prompt: Override default prompt

        Returns:
            TextOutput with extracted content
        """
        prompt = custom_prompt or self._get_default_prompt(output_format, include_layout)
        raw_output = self._backend.infer(image, prompt)
        return self._postprocess(raw_output, output_format)
```

---

## Backend Config Definitions

### PyTorch Config

```python
# omnidocs/tasks/text_extraction/qwen/pytorch.py

from pydantic import BaseModel, Field
from typing import Optional, Literal

class QwenPyTorchConfig(BaseModel):
    """PyTorch/HuggingFace backend configuration for Qwen."""

    model: str = Field(..., description="HuggingFace model ID")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Torch dtype for model"
    )
    trust_remote_code: bool = Field(default=True)
    device_map: Optional[str] = Field(default="auto")
    max_memory: Optional[dict] = Field(default=None)
    quantization: Optional[Literal["4bit", "8bit"]] = Field(default=None)

    class Config:
        extra = "forbid"
```

### VLLM Config

```python
# omnidocs/tasks/text_extraction/qwen/vllm.py

from pydantic import BaseModel, Field
from typing import Optional

class QwenVLLMConfig(BaseModel):
    """VLLM backend configuration for Qwen."""

    model: str = Field(..., description="HuggingFace model ID")
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: Optional[int] = Field(default=None)
    enforce_eager: bool = Field(default=False)
    trust_remote_code: bool = Field(default=True)
    dtype: str = Field(default="bfloat16")

    # VLLM-specific features
    enable_prefix_caching: bool = Field(default=False)
    enable_chunked_prefill: bool = Field(default=False)

    class Config:
        extra = "forbid"
```

### MLX Config

```python
# omnidocs/tasks/text_extraction/qwen/mlx.py

from pydantic import BaseModel, Field
from typing import Optional, Literal

class QwenMLXConfig(BaseModel):
    """MLX backend configuration for Qwen (Apple Silicon)."""

    model: str = Field(..., description="MLX model path or HuggingFace ID")
    quantization: Optional[Literal["4bit", "8bit"]] = Field(default=None)
    max_tokens: int = Field(default=4096)

    class Config:
        extra = "forbid"
```

### API Config

```python
# omnidocs/tasks/text_extraction/qwen/api.py

from pydantic import BaseModel, Field
from typing import Optional, Dict

class QwenAPIConfig(BaseModel):
    """API backend configuration for Qwen (hosted or proxy)."""

    model: str = Field(..., description="API model identifier")
    api_key: str = Field(..., description="API key")
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API endpoint (for proxies)"
    )
    rate_limit: int = Field(default=10, ge=1, description="Requests per minute")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0)
    custom_headers: Optional[Dict[str, str]] = Field(default=None)

    class Config:
        extra = "forbid"
```

---

## Inference Utilities

The `omnidocs/inference/` module contains shared utilities for each backend:

```
omnidocs/inference/
├── __init__.py
├── base.py          # Base inference class
├── pytorch.py       # PyTorch utilities
├── vllm.py          # VLLM utilities
├── mlx.py           # MLX utilities
└── api.py           # LiteLLM/API utilities
```

### Base Inference Class

```python
# omnidocs/inference/base.py

from abc import ABC, abstractmethod
from typing import Any
from PIL import Image

class BaseInference(ABC):
    """Base class for inference backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def infer(self, image: Image.Image, prompt: str) -> Any:
        """Run inference."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Free resources."""
        pass
```

### PyTorch Inference

```python
# omnidocs/inference/pytorch.py

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from .base import BaseInference

class PyTorchInference(BaseInference):
    """PyTorch/HuggingFace inference backend."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        self.processor = AutoProcessor.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=dtype_map[self.config.torch_dtype],
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.model.eval()

    def infer(self, image: Image.Image, prompt: str):
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
            )

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def unload(self):
        if self.model:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
```

### VLLM Inference

```python
# omnidocs/inference/vllm.py

from PIL import Image
from .base import BaseInference

class VLLMInference(BaseInference):
    """VLLM inference backend."""

    def __init__(self, config):
        self.config = config
        self.llm = None
        self.load_model()

    def load_model(self):
        from vllm import LLM

        self.llm = LLM(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enforce_eager=self.config.enforce_eager,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
        )

    def infer(self, image: Image.Image, prompt: str):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
        )

        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    def unload(self):
        if self.llm:
            del self.llm
```

### API Inference

```python
# omnidocs/inference/api.py

import base64
from io import BytesIO
from PIL import Image
from .base import BaseInference

class APIInference(BaseInference):
    """LiteLLM/API inference backend."""

    def __init__(self, config):
        self.config = config
        self.load_model()

    def load_model(self):
        """Validate API configuration."""
        import litellm

        # Configure LiteLLM
        if self.config.base_url:
            litellm.api_base = self.config.base_url

    def infer(self, image: Image.Image, prompt: str):
        import litellm

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = litellm.completion(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_base64}"
                        }
                    ]
                }
            ],
            timeout=self.config.timeout,
            num_retries=self.config.max_retries,
        )

        return response.choices[0].message.content

    def unload(self):
        pass  # Nothing to unload
```

---

## Dependency Management

### pyproject.toml Structure

```toml
[project]
name = "omnidocs"
dependencies = [
    "pydantic>=2.0",
    "pillow>=10.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
# Individual backends
pytorch = [
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.40",
]

vllm = [
    "vllm>=0.4.0",
    "torch>=2.0",
]

mlx = [
    "mlx>=0.10",
    "mlx-lm>=0.10",
]

api = [
    "litellm>=1.30",
    "openai>=1.0",
]

# Convenience groups
local = ["omnidocs[pytorch]"]
all-local = ["omnidocs[pytorch,vllm,mlx]"]
all = ["omnidocs[pytorch,vllm,mlx,api]"]

# Development
dev = ["omnidocs[all]", "pytest", "black", "mypy"]
```

### Installation Examples

```bash
# Minimal (no inference)
pip install omnidocs

# PyTorch only (most common)
pip install omnidocs[pytorch]

# High-throughput serving
pip install omnidocs[vllm]

# Apple Silicon
pip install omnidocs[mlx]

# API only (no local inference)
pip install omnidocs[api]

# Everything
pip install omnidocs[all]
```

---

## Lazy Import Pattern

To avoid import errors when backends aren't installed:

```python
# omnidocs/tasks/text_extraction/qwen.py

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from omnidocs.tasks.text_extraction.qwen import (
        QwenPyTorchConfig,
        QwenVLLMConfig,
        QwenMLXConfig,
        QwenAPIConfig,
    )


class QwenTextExtractor:
    def __init__(self, backend):
        self.backend_config = backend
        self._backend = None
        self._load_backend()

    def _load_backend(self):
        """Lazy load backend based on config type."""
        config = self.backend_config
        config_type = type(config).__name__

        if config_type == "QwenPyTorchConfig":
            try:
                from omnidocs.inference.pytorch import PyTorchInference
            except ImportError:
                raise ImportError(
                    "PyTorch backend requires torch and transformers. "
                    "Install with: pip install omnidocs[pytorch]"
                )
            self._backend = PyTorchInference(config)

        elif config_type == "QwenVLLMConfig":
            try:
                from omnidocs.inference.vllm import VLLMInference
            except ImportError:
                raise ImportError(
                    "VLLM backend requires vllm. "
                    "Install with: pip install omnidocs[vllm]"
                )
            self._backend = VLLMInference(config)

        # ... etc
```

---

## Error Handling

### Config Validation

```python
from pydantic import ValidationError

try:
    config = QwenVLLMConfig(
        model="Qwen/Qwen2-VL-7B",
        tensor_parallel_size=-1,  # Invalid!
    )
except ValidationError as e:
    print(e)
    # tensor_parallel_size: Input should be greater than or equal to 1
```

### Backend Not Installed

```python
try:
    extractor = QwenTextExtractor(
        backend=QwenVLLMConfig(model="Qwen/Qwen2-VL-7B")
    )
except ImportError as e:
    print(e)
    # VLLM backend requires vllm. Install with: pip install omnidocs[vllm]
```

### Invalid Backend for Model

```python
# DotsOCR doesn't support API
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor

# This import would fail because DotsOCRAPIConfig doesn't exist
# from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

# User naturally discovers DotsOCR doesn't support API
# because there's no config class to import
```

---

## Layout Detection: Fixed vs Flexible Models

### Overview

Layout detection models in OmniDocs fall into two categories based on label flexibility:

| Category | Examples | Label Support | Implementation |
|----------|----------|---------------|----------------|
| **Fixed Labels** | DocLayoutYOLO, RT-DETR | Predefined only | Trained model classes |
| **Flexible VLM** | Qwen, Florence-2 | Custom via prompting | Vision-language models |

### Fixed Label Models

**Models**: DocLayoutYOLO, RTDETRLayoutDetector, SuryaLayoutDetector

These models are trained on specific label sets (title, text, table, figure, etc.) and **cannot detect custom elements**.

```python
# omnidocs/tasks/layout_analysis/doc_layout_yolo.py

class DocLayoutYOLO:
    """Fixed label layout detector. PyTorch only."""

    FIXED_LABELS = ["title", "text", "list", "table", "figure", "caption", "formula"]

    def __init__(self, config: DocLayoutYOLOConfig):
        self.config = config
        self._load_model()

    def extract(self, image: Image.Image) -> LayoutOutput:
        """
        Extract layout with predefined labels only.

        Args:
            image: PIL Image

        Returns:
            LayoutOutput with bboxes using FIXED_LABELS
        """
        # Run YOLO detection
        detections = self.model(image)

        # Map to fixed labels
        bboxes = []
        for det in detections:
            label = self.FIXED_LABELS[det.class_id]
            bboxes.append(LayoutBox(label=label, bbox=det.bbox, confidence=det.conf))

        return LayoutOutput(bboxes=bboxes)
```

### Flexible VLM Models

**Models**: QwenLayoutDetector, Florence2LayoutDetector, VLMLayoutDetector

These models use vision-language prompting and **can detect ANY custom layout elements**.

```python
# omnidocs/tasks/layout_analysis/qwen.py

from typing import Union, List, Optional
from omnidocs.tasks.layout_analysis.models import CustomLabel, LayoutOutput

class QwenLayoutDetector:
    """Flexible VLM layout detector. Supports custom labels."""

    DEFAULT_LABELS = ["title", "text", "list", "table", "figure", "caption", "formula"]

    def __init__(self, backend: QwenBackendConfig):
        self.backend_config = backend
        self._backend = self._create_backend()

    def extract(
        self,
        image: Image.Image,
        custom_labels: Optional[Union[List[str], List[CustomLabel]]] = None,
    ) -> LayoutOutput:
        """
        Extract layout with flexible label support.

        Args:
            image: PIL Image
            custom_labels:
                - None: Use DEFAULT_LABELS
                - List[str]: Simple custom label names
                - List[CustomLabel]: Structured labels with metadata

        Returns:
            LayoutOutput with detected elements
        """
        # Normalize labels
        if custom_labels is None:
            labels = [CustomLabel(name=name) for name in self.DEFAULT_LABELS]
        else:
            labels = self._normalize_labels(custom_labels)

        # Build detection prompt
        prompt = self._build_prompt(labels)

        # Run VLM inference
        raw_output = self._backend.infer(image, prompt)

        # Parse results
        return self._parse_detections(raw_output, labels)

    def _normalize_labels(
        self,
        labels: Union[List[str], List[CustomLabel]]
    ) -> List[CustomLabel]:
        """Convert string labels to CustomLabel objects."""
        normalized = []
        for label in labels:
            if isinstance(label, str):
                normalized.append(CustomLabel(name=label))
            elif isinstance(label, CustomLabel):
                normalized.append(label)
        return normalized

    def _build_prompt(self, labels: List[CustomLabel]) -> str:
        """Build detection prompt from labels."""
        label_descriptions = []

        for label in labels:
            if label.detection_prompt:
                # Use custom detection prompt
                label_descriptions.append(
                    f"- {label.name}: {label.detection_prompt}"
                )
            else:
                # Use label name only
                label_descriptions.append(f"- {label.name}")

        prompt = f"""Detect the following layout elements in this document image:

{chr(10).join(label_descriptions)}

Return bounding boxes [x1, y1, x2, y2] for each detected element."""

        return prompt
```

### CustomLabel Definition

```python
# omnidocs/tasks/layout_analysis/models.py

from pydantic import BaseModel, Field
from typing import Optional

class CustomLabel(BaseModel):
    """Custom layout label definition for flexible VLM models."""

    name: str = Field(..., description="Label identifier (e.g., 'code_block')")

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description"
    )

    detection_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt hint for model to use during detection"
    )

    color: Optional[str] = Field(
        default=None,
        description="Visualization color (hex or name)"
    )

    class Config:
        extra = "allow"  # Users can add custom fields
```

### Usage Examples

**Fixed Model** (Simple, Fast):
```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)
# Returns: title, text, table, figure (fixed set)
```

**Flexible VLM** (Simple Strings):
```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
from omnidocs.tasks.layout_analysis.qwen import QwenPyTorchConfig

layout = QwenLayoutDetector(
    backend=QwenPyTorchConfig(model="Qwen/Qwen2-VL-7B")
)

# Detect custom elements
result = layout.extract(
    image,
    custom_labels=["code_block", "sidebar", "pull_quote"]
)
```

**Flexible VLM** (Structured Labels):
```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector, CustomLabel
from omnidocs.tasks.layout_analysis.qwen import QwenPyTorchConfig

layout = QwenLayoutDetector(
    backend=QwenPyTorchConfig(model="Qwen/Qwen2-VL-7B")
)

result = layout.extract(
    image,
    custom_labels=[
        CustomLabel(
            name="code_block",
            description="Source code listings",
            detection_prompt="Regions with monospace text and syntax highlighting",
            color="#2ecc71",
        ),
        CustomLabel(
            name="sidebar",
            description="Supplementary content boxes",
            detection_prompt="Boxed regions with background color or borders",
            color="#3498db",
        ),
    ]
)

# Access metadata
for box in result.bboxes:
    print(f"{box.label.name}: {box.label.description}")
```

### Benefits

| Feature | Fixed Models | Flexible VLMs |
|---------|--------------|---------------|
| **Speed** | ⚡ Fast | 🐢 Slower (VLM inference) |
| **Accuracy** | ⭐⭐⭐ High (trained) | ⭐⭐ Good (prompted) |
| **Custom Labels** | ❌ No | ✅ Yes |
| **Label Metadata** | ❌ No | ✅ Yes (CustomLabel) |
| **Detection Prompts** | ❌ No | ✅ Yes |
| **Extensibility** | ❌ No | ✅ Yes (extra fields) |
| **Use Case** | Standard documents | Any document type |

---

## Summary

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Config Pattern** | Model-specific classes | IDE support, type safety |
| **Backend Discovery** | Import exists = supported | Obvious, no guessing |
| **Lazy Imports** | Load on use | Avoid dependency errors |
| **Validation** | Pydantic | Early error detection |
| **Error Messages** | Clear install instructions | Good UX |

### Config Naming Convention

| Model Type | Config Location | Naming |
|------------|-----------------|--------|
| Single-backend | Same file as model | `{Model}Config` |
| Multi-backend | Subfolder | `{Model}{Backend}Config` |

### Parameter Naming

| Model Type | Parameter |
|------------|-----------|
| Single-backend | `config=` |
| Multi-backend | `backend=` |

---

**Last Updated**: January 20, 2026
**Status**: ✅ Design Complete
