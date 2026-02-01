# Backend System

> **Core Principle**: The same model can run on different backends. Backend selection is explicit via config classes and detected at runtime.

## Table of Contents

1. [Four Backends](#four-backends)
2. [How Backend Selection Works](#how-backend-selection-works)
3. [Config Drives Backend](#config-drives-backend)
4. [Lazy Imports](#lazy-imports)
5. [Adding New Backends](#adding-new-backends)
6. [Backend Trade-Offs](#backend-trade-offs)
7. [Real Code Examples](#real-code-examples)

---

## Four Backends

OmniDocs supports four inference backends, each optimized for different scenarios:

### 1. PyTorch Backend

**Use When**: Local GPU inference, development, small-to-medium workloads

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        torch_dtype="bfloat16",
        device_map="auto",
    )
)
```

**Dependencies**: `torch`, `transformers`, `accelerate`

**Best For**:
- Development and prototyping
- Single GPU inference
- When you control the environment
- Interactive notebooks

**Limitations**:
- Single GPU only (use tensor parallelism for multi-GPU)
- Must manage memory explicitly
- No built-in batching optimization

### 2. VLLM Backend

**Use When**: Production serving, high throughput, multi-GPU

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

extractor = QwenTextExtractor(
    backend=QwenTextVLLMConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=4,  # 4 GPUs
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
)
```

**Dependencies**: `vllm`, `torch`

**Best For**:
- Production inference servers
- High throughput requirements
- Multiple requests in parallel
- Multi-GPU deployments

**Limitations**:
- Requires CUDA (GPU only)
- More complex setup
- Higher memory overhead

### 3. MLX Backend

**Use When**: Apple Silicon (M1, M2, M3+), local inference without GPU cost

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

extractor = QwenTextExtractor(
    backend=QwenTextMLXConfig(
        model="Qwen/Qwen3-VL-8B-Instruct-MLX",
        quantization="4bit",
    )
)
```

**Dependencies**: `mlx`, `mlx-lm`

**Best For**:
- Development on Apple Silicon Macs
- Local inference without cloud costs
- Battery efficiency (can use GPU without NVIDIA)
- Privacy-focused applications

**Limitations**:
- Apple Silicon only (M1, M2, M3+)
- Fewer models available in MLX format
- Potentially slower than GPU for large models

### 4. API Backend

**Use When**: Hosted models, no local GPU needed, managed inference

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

extractor = QwenTextExtractor(
    backend=QwenTextAPIConfig(
        model="qwen-vision-max",
        api_key="sk-...",
        base_url="https://api.openrouter.co/openai/v1",
        rate_limit=10,  # requests per second
    )
)
```

**Dependencies**: `litellm`, `requests`

**Best For**:
- Minimal setup (no local GPU)
- Managed infrastructure
- Cost-per-request billing
- Using cloud-hosted models

**Limitations**:
- Network latency
- API rate limits
- Dependency on external service
- Cost per request

---

## How Backend Selection Works

### The Detection Mechanism

When you initialize an extractor with a config, the `_load_model()` method detects which backend to use by checking the config class name:

```python
class QwenTextExtractor(BaseTextExtractor):
    def __init__(self, backend: QwenTextBackendConfig):
        self.backend_config = backend
        self._load_model()

    def _load_model(self) -> None:
        """Load appropriate backend based on config type."""
        config_type = type(self.backend_config).__name__

        if config_type == "QwenTextPyTorchConfig":
            self._load_pytorch()
        elif config_type == "QwenTextVLLMConfig":
            self._load_vllm()
        elif config_type == "QwenTextMLXConfig":
            self._load_mlx()
        elif config_type == "QwenTextAPIConfig":
            self._load_api()
        else:
            raise ValueError(f"Unknown backend: {config_type}")
```

### Why Type Checking?

Using `isinstance()` or type name checking has important advantages:

```python
# Using isinstance() - Pythonic
if isinstance(self.backend_config, QwenTextPyTorchConfig):
    self._load_pytorch()

# Using type name - More robust to import issues
config_type = type(self.backend_config).__name__
if config_type == "QwenTextPyTorchConfig":
    self._load_pytorch()
```

**Type name approach benefits**:
- Works even if you only import the config type-checked (via `TYPE_CHECKING`)
- No circular imports
- Cleaner code structure

### Example: Full Backend Detection

```python
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .pytorch import QwenTextPyTorchConfig
    from .vllm import QwenTextVLLMConfig

class QwenTextExtractor(BaseTextExtractor):
    def __init__(self, backend: Union["QwenTextPyTorchConfig", "QwenTextVLLMConfig"]):
        self.backend_config = backend
        self._load_model()

    def _load_model(self) -> None:
        config_type = type(self.backend_config).__name__

        if config_type == "QwenTextPyTorchConfig":
            # Import only if actually using this backend
            from transformers import AutoModel
            # ... actual loading code

        elif config_type == "QwenTextVLLMConfig":
            # Import only if actually using this backend
            from vllm import LLM
            # ... actual loading code
```

---

## Config Drives Backend

### Single vs Multi-Backend Models

The presence of config classes determines backend support:

#### Single-Backend Model: DocLayoutYOLO

```python
# Only one config class = only one backend
from omnidocs.tasks.layout_extraction import DocLayoutYOLOConfig

# Can only be initialized one way
extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(...))

# Other config types don't exist (ImportError)
from omnidocs.tasks.layout_extraction import DocLayoutYOLOVLLMConfig  # ImportError!
```

**File structure**:
```
omnidocs/tasks/layout_extraction/
├── doc_layout_yolo.py
│   └── DocLayoutYOLOConfig  ← Single config
│   └── DocLayoutYOLO        ← Single backend (PyTorch)
```

#### Multi-Backend Model: QwenTextExtractor

```python
# Four config classes = four backends
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,    # ← PyTorch backend
    QwenTextVLLMConfig,       # ← VLLM backend
    QwenTextMLXConfig,        # ← MLX backend
    QwenTextAPIConfig,        # ← API backend
)

# Can be initialized multiple ways
extractor1 = QwenTextExtractor(backend=QwenTextPyTorchConfig(...))
extractor2 = QwenTextExtractor(backend=QwenTextVLLMConfig(...))
extractor3 = QwenTextExtractor(backend=QwenTextMLXConfig(...))
extractor4 = QwenTextExtractor(backend=QwenTextAPIConfig(...))
```

**File structure**:
```
omnidocs/tasks/text_extraction/
├── qwen/
│   ├── pytorch.py
│   │   └── QwenTextPyTorchConfig
│   ├── vllm.py
│   │   └── QwenTextVLLMConfig
│   ├── mlx.py
│   │   └── QwenTextMLXConfig
│   ├── api.py
│   │   └── QwenTextAPIConfig
│   └── extractor.py
│       └── QwenTextExtractor ← Uses all configs
```

### How Config Determines Behavior

Each config contains parameters specific to its backend:

```python
# PyTorch config has torch-specific parameters
class QwenTextPyTorchConfig(BaseModel):
    device: str = "cuda"
    torch_dtype: Literal["float16", "bfloat16", "float32"]
    device_map: Optional[str] = "auto"
    use_flash_attention: bool = False

# VLLM config has VLLM-specific parameters
class QwenTextVLLMConfig(BaseModel):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False

# MLX config has MLX-specific parameters
class QwenTextMLXConfig(BaseModel):
    quantization: Literal["4bit", "8bit", None] = None
    max_tokens: int = 2048

# API config has API-specific parameters
class QwenTextAPIConfig(BaseModel):
    api_key: str
    base_url: Optional[str] = None
    rate_limit: int = 10
    timeout: int = 60
```

When the extractor loads the model, it uses these backend-specific parameters:

```python
def _load_pytorch(self):
    from transformers import AutoModel
    config = self.backend_config  # QwenTextPyTorchConfig

    model = AutoModel.from_pretrained(
        config.model,
        device_map=config.device_map,  # From config
        torch_dtype=config.torch_dtype,  # From config
    )

def _load_vllm(self):
    from vllm import LLM
    config = self.backend_config  # QwenTextVLLMConfig

    model = LLM(
        model=config.model,
        tensor_parallel_size=config.tensor_parallel_size,  # From config
        gpu_memory_utilization=config.gpu_memory_utilization,  # From config
    )
```

---

## Lazy Imports

OmniDocs uses **lazy imports** to avoid requiring all dependencies upfront.

### The Problem with Eager Imports

```python
# ❌ BAD: Requires all dependencies at startup
from omnidocs.tasks.text_extraction.qwen import QwenTextExtractor
# Tries to import:
# - transformers (for PyTorch)
# - vllm (for VLLM)
# - mlx (for MLX)
# - litellm (for API)
# Even if user only wants PyTorch!
# ImportError if any dependency is missing!
```

### Solution: Lazy Imports

```python
# ✅ GOOD: Import extractor without dependencies
from omnidocs.tasks.text_extraction import QwenTextExtractor

# Dependencies imported only when loading that backend
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(...)  # Only torch imported now
)
```

### How It's Implemented

```python
class QwenTextExtractor(BaseTextExtractor):
    def _load_model(self) -> None:
        config_type = type(self.backend_config).__name__

        if config_type == "QwenTextPyTorchConfig":
            try:
                from transformers import AutoModel
                # Import only happens here
            except ImportError:
                raise ImportError(
                    "PyTorch backend requires transformers. "
                    "Install with: pip install omnidocs[pytorch]"
                )

        elif config_type == "QwenTextVLLMConfig":
            try:
                from vllm import LLM
                # Import only happens here
            except ImportError:
                raise ImportError(
                    "VLLM backend requires vllm. "
                    "Install with: pip install omnidocs[vllm]"
                )
        # ... other backends
```

### Benefits

```python
# User only has PyTorch installed
pip install omnidocs[pytorch]

# Can still import extractor (no error)
from omnidocs.tasks.text_extraction import QwenTextExtractor

# Can use PyTorch backend (works)
extractor = QwenTextExtractor(backend=QwenTextPyTorchConfig(...))

# Can't use VLLM (helpful error message)
try:
    extractor = QwenTextExtractor(backend=QwenTextVLLMConfig(...))
except ImportError as e:
    print(e)  # "VLLM backend requires vllm. Install with: pip install omnidocs[vllm]"
```

---

## Adding New Backends

To add a new backend to an existing model, follow this pattern:

### Step 1: Create Config Class

```python
# omnidocs/tasks/text_extraction/qwen/mynew_backend.py

from pydantic import BaseModel, Field, ConfigDict

class QwenTextMyNewBackendConfig(BaseModel):
    """Configuration for MyNewBackend text extraction."""

    model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="Model identifier"
    )
    param1: str = Field(default="value", description="Backend-specific param")
    param2: int = Field(default=10, ge=1, description="Another param")

    model_config = ConfigDict(extra="forbid")
```

### Step 2: Update Extractor

```python
# omnidocs/tasks/text_extraction/qwen/extractor.py

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .mynew_backend import QwenTextMyNewBackendConfig

QwenTextBackendConfig = Union[
    "QwenTextPyTorchConfig",
    "QwenTextVLLMConfig",
    "QwenTextMyNewBackendConfig",  # Add here
]

class QwenTextExtractor(BaseTextExtractor):
    def _load_model(self) -> None:
        config_type = type(self.backend_config).__name__

        if config_type == "QwenTextPyTorchConfig":
            self._load_pytorch()
        elif config_type == "QwenTextVLLMConfig":
            self._load_vllm()
        elif config_type == "QwenTextMyNewBackendConfig":  # Add handler
            self._load_mynew_backend()
        else:
            raise ValueError(f"Unknown backend: {config_type}")

    def _load_mynew_backend(self):
        """Load MyNewBackend."""
        try:
            import mynew_backend_lib
        except ImportError:
            raise ImportError(
                "MyNewBackend requires mynew_backend_lib. "
                "Install with: pip install mynew_backend_lib"
            )

        config = self.backend_config
        # Load model with mynew_backend_lib...
```

### Step 3: Export Config

```python
# omnidocs/tasks/text_extraction/qwen/__init__.py

from .pytorch import QwenTextPyTorchConfig
from .vllm import QwenTextVLLMConfig
from .mynew_backend import QwenTextMyNewBackendConfig  # Add export
from .extractor import QwenTextExtractor

__all__ = [
    "QwenTextPyTorchConfig",
    "QwenTextVLLMConfig",
    "QwenTextMyNewBackendConfig",  # Add to public API
    "QwenTextExtractor",
]
```

### Step 4: Update Dependencies

```bash
# Add backend dependency to optional group
cd Omnidocs/
uv add --group mynew_backend mynew_backend_lib
```

---

## Backend Trade-Offs

| Aspect | PyTorch | VLLM | MLX | API |
|--------|---------|------|-----|-----|
| **Setup Complexity** | Low | High | Low | Very Low |
| **Latency** | 500-2000ms | 200-500ms | 1000-3000ms | 1000-5000ms |
| **Throughput** | 1 req/s | 10-100 req/s | 1-5 req/s | 0.5-5 req/s |
| **Memory (8B model)** | 16GB VRAM | 8GB VRAM | 6GB RAM | - |
| **Cost** | Infra + GPU | Infra + GPU | None | Pay/request |
| **Privacy** | Local | Local | Local | Cloud |
| **Hardware** | Any NVIDIA GPU | Multi-GPU | Apple Silicon | None (API) |
| **Scaling** | Single GPU | Multi-GPU | Single device | Unlimited |

### Decision Matrix

**Choose PyTorch if**:
- Developing locally
- Single GPU available
- Want simplicity
- Cost not a concern

**Choose VLLM if**:
- Production serving
- High throughput needed
- Multiple GPUs available
- Can handle complexity

**Choose MLX if**:
- Developing on Apple Silicon
- Want local inference
- Battery efficiency matters
- Minimal setup

**Choose API if**:
- No GPU available
- Want managed infrastructure
- Pay-per-request pricing OK
- Minimal setup needed

---

## Real Code Examples

### Example 1: Multi-Backend Same Code

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
)
from omnidocs import Document

doc = Document.from_pdf("paper.pdf")
page = doc.get_page(0)

# Backend 1: PyTorch
print("=== PyTorch Backend ===")
extractor1 = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)
result1 = extractor1.extract(page, output_format="markdown")
print(result1.content[:200])

# Backend 2: VLLM (same extractor, different backend)
print("\n=== VLLM Backend ===")
extractor2 = QwenTextExtractor(
    backend=QwenTextVLLMConfig(tensor_parallel_size=2)
)
result2 = extractor2.extract(page, output_format="markdown")
print(result2.content[:200])

# Results should be identical (or very similar)
# Code is identical except for config!
```

### Example 2: Detecting Available Backends

```python
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
    QwenTextMLXConfig,
    QwenTextAPIConfig,
)

backends = [
    ("PyTorch", QwenTextPyTorchConfig),
    ("VLLM", QwenTextVLLMConfig),
    ("MLX", QwenTextMLXConfig),
    ("API", QwenTextAPIConfig),
]

print("Available backends:")
for name, config_class in backends:
    print(f"  - {name}: {config_class.__doc__.split(chr(10))[0]}")

# Output:
# Available backends:
#   - PyTorch: PyTorch/HuggingFace backend configuration for Qwen text extraction.
#   - VLLM: VLLM backend configuration for Qwen text extraction.
#   - MLX: MLX backend configuration for Qwen text extraction.
#   - API: API backend configuration for Qwen text extraction.
```

### Example 3: Conditional Backend Selection

```python
import os
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
    QwenTextMLXConfig,
    QwenTextAPIConfig,
)

def get_extractor():
    """Select best available backend."""
    backend_choice = os.environ.get("OMNIDOCS_BACKEND", "auto")

    if backend_choice == "pytorch":
        config = QwenTextPyTorchConfig(device="cuda")
    elif backend_choice == "vllm":
        config = QwenTextVLLMConfig(tensor_parallel_size=2)
    elif backend_choice == "mlx":
        config = QwenTextMLXConfig()
    elif backend_choice == "api":
        config = QwenTextAPIConfig(api_key=os.environ["OPENROUTER_API_KEY"])
    else:  # auto
        # Try PyTorch first, fall back to others
        try:
            config = QwenTextPyTorchConfig(device="cuda")
        except ImportError:
            try:
                config = QwenTextVLLMConfig()
            except ImportError:
                config = QwenTextAPIConfig(
                    api_key=os.environ.get("OPENROUTER_API_KEY")
                )

    return QwenTextExtractor(backend=config)

extractor = get_extractor()
```

### Example 4: Single-Backend Model

```python
from omnidocs.tasks.layout_extraction import (
    DocLayoutYOLO,
    DocLayoutYOLOConfig,
)

# This model only has one backend (PyTorch)
# Config type is obvious
config = DocLayoutYOLOConfig(device="cuda", confidence=0.3)
extractor = DocLayoutYOLO(config=config)

# These don't exist (it's single-backend)
# from omnidocs.tasks.layout_extraction import (
#     DocLayoutYOLOVLLMConfig,  # ❌ ImportError
#     DocLayoutYOLOAPIConfig,   # ❌ ImportError
# )
```

---

## Summary

| Concept | Key Points |
|---------|-----------|
| **4 Backends** | PyTorch, VLLM, MLX, API - different use cases |
| **Config-Driven** | Config class type determines which backend loads |
| **Detection** | `_load_model()` checks `type(backend_config).__name__` |
| **Lazy Imports** | Dependencies only imported when backend used |
| **Backend Discoverability** | If config exists, backend is available |
| **Trade-Offs** | Speed vs complexity, cost vs control, privacy vs convenience |

---

## Next Steps

- See [Config Pattern](./config-pattern.md) for how configs are structured
- See [Architecture Overview](./architecture-overview.md) for system design
- Read backend-specific documentation for detailed setup instructions

