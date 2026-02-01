# Config Pattern

> **Core Principle**: Pydantic configs drive both backend selection AND model initialization. Separation of concerns: configs for setup, method parameters for task options.

## Table of Contents

1. [Single vs Multi-Backend](#single-vs-multi-backend)
2. [How to Tell Which Is Which](#how-to-tell-which-is-which)
3. [Pydantic Config Structure](#pydantic-config-structure)
4. [Init vs Extract Separation](#init-vs-extract-separation)
5. [What Goes Where](#what-goes-where)
6. [Real Examples](#real-examples)
7. [Extending Configs](#extending-configs)

---

## Single vs Multi-Backend

OmniDocs models fall into two categories based on backend support:

### Single-Backend Models

A model that works with ONLY ONE backend has a single config class.

```python
# Example: DocLayout-YOLO (PyTorch only)
from omnidocs.tasks.layout_extraction import (
    DocLayoutYOLO,
    DocLayoutYOLOConfig,  # ← Single config
)

# Parameter name is always 'config='
extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(...))
```

**Characteristics**:
- One config class named `{Model}Config`
- Parameter name: `config=`
- Model optimized for specific backend
- Can't switch backends

**When This Happens**:
- Model only supports one framework (e.g., YOLO models are PyTorch)
- Backend has unique requirements
- Model implementation already optimized for one backend

### Multi-Backend Models

A model that works with MULTIPLE backends has multiple config classes.

```python
# Example: Qwen (PyTorch, VLLM, MLX, API)
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,    # ← Config 1
    QwenTextVLLMConfig,       # ← Config 2
    QwenTextMLXConfig,        # ← Config 3
    QwenTextAPIConfig,        # ← Config 4
)
from omnidocs.tasks.text_extraction import QwenTextExtractor

# Parameter name is always 'backend='
extractor = QwenTextExtractor(backend=QwenTextPyTorchConfig(...))
extractor = QwenTextExtractor(backend=QwenTextVLLMConfig(...))
extractor = QwenTextExtractor(backend=QwenTextMLXConfig(...))
extractor = QwenTextExtractor(backend=QwenTextAPIConfig(...))
```

**Characteristics**:
- Multiple config classes: `{Model}{Backend}Config`
- Parameter name: `backend=`
- One extractor works with all configs
- Can switch backends by changing config type
- Each backend has optimized implementation

**When This Happens**:
- Model available on multiple frameworks (Qwen on PyTorch, VLLM, MLX)
- Want to support different hardware (GPU, Apple Silicon, API)
- Need flexibility for different deployment scenarios

---

## How to Tell Which Is Which

### Method 1: Look at Imports

```python
# Single-backend: One config class
from omnidocs.tasks.layout_extraction import DocLayoutYOLOConfig

# Multi-backend: Multiple config classes
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
    QwenTextMLXConfig,
    QwenTextAPIConfig,
)
```

### Method 2: Check Parameter Name

```python
# If parameter is 'config=' → Single-backend
extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(...))

# If parameter is 'backend=' → Multi-backend
extractor = QwenTextExtractor(backend=QwenTextPyTorchConfig(...))
```

### Method 3: Try to Import Other Backends

```python
# Single-backend: ImportError for other backends
try:
    from omnidocs.tasks.layout_extraction import DocLayoutYOLOVLLMConfig
except ImportError:
    print("Single-backend model (PyTorch only)")

# Multi-backend: All imports succeed
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
    QwenTextMLXConfig,
    QwenTextAPIConfig,
)
print("Multi-backend model (4 backends available)")
```

### Method 4: Check Constructor Signature

```python
# Single-backend: config parameter
import inspect
sig = inspect.signature(DocLayoutYOLO.__init__)
print(sig)  # (__self, config: DocLayoutYOLOConfig)

# Multi-backend: backend parameter
sig = inspect.signature(QwenTextExtractor.__init__)
print(sig)  # (__self, backend: QwenTextBackendConfig)
```

---

## Pydantic Config Structure

All configs are **Pydantic BaseModel** classes. This provides:
- Type validation
- IDE autocomplete
- Documentation
- Default values
- Custom validation

### Basic Structure

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional

class MyConfig(BaseModel):
    """
    Clear docstring explaining what this config does.

    This config is for [purpose/backend].
    """

    # Required parameter (no default)
    param1: str = Field(
        ...,  # ... means required
        description="What this parameter does"
    )

    # Optional parameter with default
    param2: str = Field(
        default="default_value",
        description="Optional parameter"
    )

    # Numeric with bounds
    param3: int = Field(
        default=10,
        ge=1,         # greater than or equal
        le=100,       # less than or equal
        description="Must be between 1 and 100"
    )

    # Enumerated values
    param4: Literal["option1", "option2"] = Field(
        default="option1",
        description="Choose one option"
    )

    # Nullable/Optional
    param5: Optional[str] = Field(
        default=None,
        description="Can be None or a string"
    )

    # Pydantic model config
    model_config = ConfigDict(
        extra="forbid",  # Raise error on unknown fields
    )
```

### Real Example: QwenTextPyTorchConfig

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional

class QwenTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for Qwen text extraction.

    This backend uses the transformers library with PyTorch for local GPU inference.
    Requires: torch, transformers, accelerate, qwen-vl-utils

    Example:
        ```python
        config = QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-8B-Instruct",
            device="cuda",
            torch_dtype="bfloat16",
        )
        ```
    """

    model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="HuggingFace model ID (e.g., Qwen/Qwen3-VL-2B-Instruct, "
        "Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct)",
    )

    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'mps', 'cpu'. "
        "Auto-detects best available if specified device is unavailable.",
    )

    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="auto",
        description="Torch dtype for model weights. 'auto' lets the model decide.",
    )

    device_map: Optional[str] = Field(
        default="auto",
        description="Device map strategy for model parallelism. "
        "Options: 'auto', 'balanced', 'sequential', or None.",
    )

    max_new_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum number of tokens to generate.",
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )

    model_config = ConfigDict(extra="forbid")
```

### Field Validation Options

```python
from pydantic import BaseModel, Field, field_validator

class ConfigExample(BaseModel):
    # String constraints
    device: str = Field(default="cuda", min_length=1, max_length=10)

    # Numeric constraints
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=1024)

    # Pattern matching
    api_key: str = Field(default="", pattern=r"^sk-[a-zA-Z0-9]+$")

    # Enumerated
    format: Literal["json", "xml", "csv"] = Field(default="json")

    # Custom validation
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"Invalid device: {v}")
        return v

    # Ensure exclusive options
    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, v, info):
        if info.data.get("use_api") and v > 8:
            raise ValueError("API backend doesn't support models > 8B")
        return v
```

---

## Init vs Extract Separation

### Clear Separation of Concerns

OmniDocs strictly separates **initialization parameters** from **task parameters**.

#### Initialization (goes in config)

Config parameters are determined ONCE when you create the extractor.

```python
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",  # ← Model choice (init)
    device="cuda",                      # ← Hardware (init)
    torch_dtype="bfloat16",             # ← Quantization (init)
)

extractor = QwenTextExtractor(backend=config)
# Model loaded ONCE here
```

**Characteristics**:
- Set once at extractor creation
- Don't change during processing
- Affect model loading
- Backend-specific

#### Task Parameters (goes in extract())

Extract parameters change per-call based on task requirements.

```python
# Same extractor, different task parameters
result1 = extractor.extract(
    image1,
    output_format="markdown",   # ← Task parameter
    include_layout=True,        # ← Task parameter
)

result2 = extractor.extract(
    image2,
    output_format="html",       # ← Different task parameter
    include_layout=False,       # ← Different task parameter
)
```

**Characteristics**:
- Set per-extraction call
- Can vary between calls
- Affect task output
- Backend-agnostic

### Why This Matters

```python
# ❌ BAD: Task parameters in config
config = ExtractorConfig(
    model="qwen",
    output_format="markdown",  # ← Should be in extract()!
)
extractor = MyExtractor(config=config)
result = extractor.extract(image)  # Always markdown!

# Can't change output format without recreating extractor

# ✅ GOOD: Task parameters in extract()
config = QwenTextPyTorchConfig(model="qwen")
extractor = QwenTextExtractor(backend=config)

result1 = extractor.extract(image, output_format="markdown")
result2 = extractor.extract(image, output_format="html")
# Same extractor, different outputs!
```

---

## What Goes Where

### In Config ✅

```python
class QwenTextPyTorchConfig(BaseModel):
    # Model choice
    model: str  # Which model to load?

    # Hardware configuration
    device: str  # GPU or CPU?
    torch_dtype: Literal[...]  # What precision?
    device_map: Optional[str]  # How to parallelize?

    # Loading options
    trust_remote_code: bool  # Custom model code?
    use_flash_attention: bool  # Acceleration method?

    # Resource limits
    max_new_tokens: int  # Max generation length
    temperature: float  # Sampling randomness
```

**Rule**: If it determines HOW THE MODEL LOADS, it goes in config.

### In extract() ✅

```python
def extract(
    self,
    image: Image.Image,
    output_format: Literal["markdown", "html"],  # Output format
    include_layout: bool = False,  # Include layout info?
    custom_prompt: Optional[str] = None,  # Custom instruction?
) -> TextOutput:
    """..."""
    pass
```

**Rule**: If it affects WHAT THE MODEL OUTPUTS, it goes in extract().

### Decision Tree

```
Question: Where does this parameter go?

├─ Does it determine which/how the model loads?
│  └─ YES → Goes in Config (init time)
│
├─ Does it change between calls for same image?
│  └─ YES → Goes in extract() (call time)
│
├─ Is it backend-specific?
│  └─ YES → Goes in Config (different for each backend)
│
└─ Is it task-specific but not backend-specific?
   └─ YES → Goes in extract()
```

### Examples

```python
# device: "cuda", "cpu", "mps"
# ✅ Config (affects model loading, backend-specific)

# output_format: "markdown", "html"
# ✅ extract() (task output, not model loading)

# model: "Qwen/Qwen3-VL-8B"
# ✅ Config (determines which model to load)

# include_layout: True/False
# ✅ extract() (task option, same model, different output)

# max_new_tokens: 8192
# ✅ Config (generation limit, affects model inference)

# custom_prompt: str
# ✅ extract() (task-specific instruction)

# quantization: "4bit", "8bit"
# ✅ Config (affects model loading/memory)

# batch_size: 32
# ✅ Config (affects GPU memory, set once)
```

---

## Real Examples

### Example 1: Single-Backend Model (DocLayoutYOLO)

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

# Configuration (one time)
config = DocLayoutYOLOConfig(
    device="cuda",          # Where to run (config)
    img_size=1024,          # Model input size (config)
    confidence=0.3,         # Detection threshold (config)
)

extractor = DocLayoutYOLO(config=config)
# Model loads with these settings

# Usage (can vary per call)
result1 = extractor.extract(image1)
result2 = extractor.extract(image2)
# Same config, same model, different inputs
```

**Analysis**:
- Single config class: `DocLayoutYOLOConfig`
- Parameter name: `config=`
- Config contains: device, input size, threshold
- extract() only takes image (no other params)

### Example 2: Multi-Backend Model (QwenTextExtractor)

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
)

# ────────────────────────────────
# SCENARIO 1: PyTorch Backend
# ────────────────────────────────
config_pytorch = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",  # Model to load
    device="cuda",                      # Hardware
    torch_dtype="bfloat16",             # Precision
)

extractor_pytorch = QwenTextExtractor(backend=config_pytorch)

# Same extract() call works the same way
result = extractor_pytorch.extract(
    image,
    output_format="markdown",  # Task parameter
)

# ────────────────────────────────
# SCENARIO 2: VLLM Backend
# ────────────────────────────────
config_vllm = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",  # Same model
    tensor_parallel_size=2,             # GPU parallelism (VLLM-specific)
    gpu_memory_utilization=0.9,         # Memory limit (VLLM-specific)
)

extractor_vllm = QwenTextExtractor(backend=config_vllm)

# Same extract() call works the same way
result = extractor_vllm.extract(
    image,
    output_format="markdown",  # Task parameter
)

# ────────────────────────────────
# KEY INSIGHT
# ────────────────────────────────
# - Different configs: PyTorchConfig vs VLLMConfig
# - Different backend initialization
# - SAME extract() interface!
# - Backend differences hidden from user
```

### Example 3: Custom Validation

```python
from pydantic import BaseModel, Field, field_validator

class CustomConfig(BaseModel):
    """Config with custom validation."""

    model_size: Literal["small", "medium", "large"] = Field(
        default="medium",
        description="Model size category"
    )

    max_batch_size: int = Field(default=32, ge=1, le=128)

    # Custom validation: batch size depends on model size
    @field_validator("max_batch_size")
    @classmethod
    def validate_batch_size(cls, v, info):
        model_size = info.data.get("model_size")

        if model_size == "small" and v > 128:
            raise ValueError("Small model max batch size is 128")
        elif model_size == "large" and v < 8:
            raise ValueError("Large model min batch size is 8")

        return v

# Valid
config = CustomConfig(model_size="small", max_batch_size=64)

# Invalid
config = CustomConfig(model_size="large", max_batch_size=4)
# ValidationError: Large model min batch size is 8
```

### Example 4: Config with Enum

```python
from enum import Enum
from pydantic import BaseModel, Field

class QuantizationType(str, Enum):
    """Quantization options."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"

class MLXConfig(BaseModel):
    """MLX backend config."""

    quantization: QuantizationType = Field(
        default=QuantizationType.INT4,
        description="Quantization type for MLX"
    )

# Usage
config = MLXConfig(quantization=QuantizationType.INT4)
# Or with string (auto-converted)
config = MLXConfig(quantization="int4")
```

---

## Extending Configs

### Adding New Parameters

```python
# Original config
class MyConfig(BaseModel):
    model: str
    device: str

# Extended config
class MyConfigV2(BaseModel):
    model: str
    device: str
    # New parameter
    memory_limit: Optional[int] = Field(
        default=None,
        description="GPU memory limit in MB"
    )
```

### Config Inheritance

```python
from pydantic import BaseModel, Field

class BaseInferenceConfig(BaseModel):
    """Common inference parameters."""
    device: str = Field(default="cuda")
    max_tokens: int = Field(default=2048)

    model_config = ConfigDict(extra="forbid")

class SpecificModelConfig(BaseInferenceConfig):
    """Model-specific parameters."""
    model: str = Field(...)  # Required
    # Inherits: device, max_tokens

config = SpecificModelConfig(
    model="my-model",
    device="cpu",  # From base
    max_tokens=4096,  # From base
)
```

### Frozen Configs

```python
from pydantic import ConfigDict

class ImmutableConfig(BaseModel):
    """Config that can't be changed after creation."""

    model: str
    device: str

    model_config = ConfigDict(
        frozen=True,  # ← Prevents modification
        extra="forbid",
    )

config = ImmutableConfig(model="qwen", device="cuda")
config.device = "cpu"  # Error: Config is frozen
```

---

## Summary

| Aspect | Single-Backend | Multi-Backend |
|--------|----------------|---------------|
| **Config Classes** | One (`{Model}Config`) | Multiple (`{Model}{Backend}Config`) |
| **Parameter Name** | `config=` | `backend=` |
| **Backend Selection** | Fixed | Type of config passed |
| **Use Case** | Model designed for one backend | Model supports multiple backends |
| **Example** | `DocLayoutYOLOConfig` | `QwenTextPyTorchConfig`, `QwenTextVLLMConfig`, ... |

**Key Rules**:
1. **Config is for init** - Model setup, happens once
2. **extract() is for tasks** - Task parameters, change per-call
3. **Pydantic validates** - Errors caught at creation time
4. **Types matter** - IDE shows all options, typos caught

---

## Next Steps

- See [Backend System](./backend-system.md) for how backends work
- See [Architecture Overview](./architecture-overview.md) for system design
- Read model-specific documentation for exact parameters

