# Model Registry System

This document explains the unified model configuration and registry system in Omnidocs.

## Overview

The Model Registry provides a centralized, standardized way to manage all models across Omnidocs. It supports:

- **HuggingFace Transformers models** (e.g., Nougat, Donut, TableFormer)
- **YOLO-based models** (e.g., DocLayout-YOLO, RT-DETR)
- **Library-managed models** (e.g., Surya, EasyOCR, PaddleOCR)
- **Vision Language Models (VLMs)** (e.g., GPT-4 Vision, Claude 3, LLaVA)
- **Custom models**

## Architecture

The system consists of three main components:

1. **Model Configurations** (`omnidocs/core/model_registry.py`)
   - Pydantic models defining model metadata and settings
   - Type-safe configuration with validation

2. **Model Catalog** (`omnidocs/core/models_catalog.py`)
   - Registry of all available models
   - Auto-registration on import

3. **Model Loader** (`omnidocs/core/model_loader.py`)
   - Utilities for downloading and loading models
   - Consistent handling across model types

## Model Configuration Types

### Base ModelConfig

All model configurations inherit from `ModelConfig`:

```python
from omnidocs.core import ModelConfig, TaskType, ModelType

config = ModelConfig(
    id="unique-model-id",
    name="Human Readable Name",
    model_type=ModelType.CUSTOM,
    task_type=TaskType.LAYOUT_ANALYSIS,
    source="source-identifier",
    version="1.0",
    confidence_threshold=0.5,
    description="What this model does",
)
```

### HuggingFaceModelConfig

For Transformer-based models from HuggingFace Hub:

```python
from omnidocs.core import HuggingFaceModelConfig, TaskType

NOUGAT_BASE = HuggingFaceModelConfig(
    id="nougat-base",
    name="Nougat Base",
    task_type=TaskType.MATH_EXPRESSION,
    hf_model_id="facebook/nougat-base",
    requires_processor=True,
    local_dir="nougat_ckpt",
    description="Nougat base model for LaTeX extraction",
)
```

### YOLOModelConfig

For YOLO-based object detection models:

```python
from omnidocs.core import YOLOModelConfig, TaskType

DOCLAYOUT_YOLO = YOLOModelConfig(
    id="doclayout-yolo-docstructbench",
    name="DocLayout-YOLO DocStructBench",
    task_type=TaskType.LAYOUT_ANALYSIS,
    model_repo="juliozhao/DocLayout-YOLO-DocStructBench",
    model_filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    yolo_version="v10",
    imgsz=1024,
    confidence_threshold=0.25,
    local_dir="DocLayout-YOLO-DocStructBench",
)
```

### LibraryManagedModelConfig

For models managed by external libraries (Surya, EasyOCR, etc.):

```python
from omnidocs.core import LibraryManagedModelConfig, TaskType

SURYA_LAYOUT = LibraryManagedModelConfig(
    id="surya-layout",
    name="Surya Layout Analysis",
    task_type=TaskType.LAYOUT_ANALYSIS,
    library_name="surya",
    predictor_class="LayoutPredictor",
    description="Fast and accurate layout detection",
    local_dir="surya",
)
```

### VLMModelConfig

For Vision Language Models:

```python
from omnidocs.core import VLMModelConfig, TaskType

LLAVA_1_6 = VLMModelConfig(
    id="llava-1.6-vicuna-7b",
    name="LLaVA 1.6 Vicuna 7B",
    task_type=TaskType.VLM_PROCESSING,
    hf_model_id="liuhaotian/llava-v1.6-vicuna-7b",
    supports_multimodal=True,
    max_tokens=2048,
    image_size=336,
    description="LLaVA 1.6 vision-language model",
)
```

## Adding a New Model

### Step 1: Define the Model Configuration

Add your model configuration to `omnidocs/core/models_catalog.py`:

```python
# Example: Adding a new VLM
QWEN_VL = VLMModelConfig(
    id="qwen-vl-chat",
    name="Qwen-VL Chat",
    task_type=TaskType.VLM_PROCESSING,
    hf_model_id="Qwen/Qwen-VL-Chat",
    supports_multimodal=True,
    max_tokens=2048,
    supports_batch=True,
    description="Qwen Vision-Language model for document understanding",
    extra_config={
        "supports_grounding": True,
        "supports_ocr": True,
    }
)
```

### Step 2: Register the Model

Add it to the `register_all_models()` function:

```python
def register_all_models():
    """Register all models with the ModelRegistry."""
    models = [
        # ... existing models ...
        QWEN_VL,  # Add your model here
    ]

    for model in models:
        ModelRegistry.register(model)
```

### Step 3: Create an Extractor (if needed)

Create an extractor class that uses the model:

```python
from omnidocs.core import ModelRegistry, ModelLoader

class QwenVLExtractor(BaseExtractor):
    """Qwen-VL based document understanding."""

    MODEL_ID = "qwen-vl-chat"

    def __init__(self, device: Optional[str] = None, show_log: bool = False):
        super().__init__(show_log=show_log)

        # Get model configuration from registry
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        if not self.model_config:
            raise ValueError(f"Model '{self.MODEL_ID}' not found in registry")

        # Load model using ModelLoader
        model_data = ModelLoader.load_from_registry(
            self.MODEL_ID,
            device=device,
            download_if_needed=True
        )

        self.model = model_data["model"]
        self.processor = model_data.get("processor")

        if show_log:
            logger.success(f"Loaded model: {self.model_config.description}")
```

## Using the Model Registry

### Getting a Model Configuration

```python
from omnidocs.core import ModelRegistry, get_model_config

# Method 1: Using the registry directly
config = ModelRegistry.get("nougat-base")

# Method 2: Using the helper function
config = get_model_config("nougat-base")
```

### Listing Models by Task

```python
from omnidocs.core import ModelRegistry, TaskType, list_models_for_task

# Get all layout analysis models
layout_models = ModelRegistry.get_by_task(TaskType.LAYOUT_ANALYSIS)

# Or use the helper function
layout_models = list_models_for_task("layout_analysis")

for model in layout_models:
    print(f"{model.id}: {model.name} - {model.description}")
```

### Listing Models by Type

```python
from omnidocs.core import ModelRegistry, ModelType

# Get all YOLO models
yolo_models = ModelRegistry.get_by_type(ModelType.YOLO)

# Get all VLMs
vlms = ModelRegistry.get_by_type(ModelType.VLM)
```

### Loading a Model

```python
from omnidocs.core import ModelLoader

# Load any model by ID
model_data = ModelLoader.load_from_registry(
    "nougat-base",
    device="cuda",
    download_if_needed=True
)

model = model_data["model"]
processor = model_data.get("processor")
config = model_data["config"]
```

## Adding VLM Support

Vision Language Models require special handling for multimodal inputs. Here's a complete example:

### 1. Define the VLM Configuration

```python
# In omnidocs/core/models_catalog.py

GPT4_VISION = VLMModelConfig(
    id="gpt4-vision",
    name="GPT-4 Vision",
    task_type=TaskType.VLM_PROCESSING,
    source="openai",
    supports_multimodal=True,
    max_tokens=4096,
    description="OpenAI GPT-4 with vision capabilities",
    extra_config={
        "api_based": True,
        "requires_api_key": True,
        "api_endpoint": "https://api.openai.com/v1/chat/completions",
    },
)
```

### 2. Create a VLM Base Class

```python
# In omnidocs/tasks/vlm_processing/base.py

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from pathlib import Path
from PIL import Image

class BaseVLMProcessor(ABC):
    """Base class for VLM processors."""

    def __init__(self, show_log: bool = False):
        self.show_log = show_log
        self.model_config = None

    @abstractmethod
    def process(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process image with text prompt."""
        pass

    @abstractmethod
    def batch_process(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple images with prompts."""
        pass
```

### 3. Implement a VLM Extractor

```python
# In omnidocs/tasks/vlm_processing/extractors/qwen_vl.py

from omnidocs.core import ModelRegistry, VLMModelConfig
from omnidocs.tasks.vlm_processing.base import BaseVLMProcessor

class QwenVLProcessor(BaseVLMProcessor):
    """Qwen-VL vision-language model processor."""

    MODEL_ID = "qwen-vl-chat"

    def __init__(self, device: Optional[str] = None, show_log: bool = False):
        super().__init__(show_log=show_log)

        # Get model configuration
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        if not isinstance(self.model_config, VLMModelConfig):
            raise TypeError(f"Expected VLMModelConfig for {self.MODEL_ID}")

        # Initialize model
        self._load_model(device)

    def _load_model(self, device: Optional[str] = None):
        """Load Qwen-VL model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_model_id,
            trust_remote_code=True,
            device_map=device or "auto"
        ).eval()

    def process(self, image: Union[str, Path, Image.Image], prompt: str, **kwargs):
        """Process a single image with prompt."""
        # Implementation here
        query = self.tokenizer.from_list_format([
            {'image': str(image)},
            {'text': prompt},
        ])

        response, history = self.model.chat(self.tokenizer, query=query, history=None)

        return {
            "response": response,
            "model": self.model_config.name,
            "prompt": prompt,
        }
```

## Task Types

Available task types:

- `LAYOUT_ANALYSIS`: Document layout detection
- `TABLE_EXTRACTION`: Table detection and structure recognition
- `OCR`: Optical character recognition
- `TEXT_EXTRACTION`: Text detection and ordering
- `MATH_EXPRESSION`: LaTeX/math formula extraction
- `VLM_PROCESSING`: Vision-language model processing
- `DOCUMENT_UNDERSTANDING`: Multi-modal document understanding

## Model Types

Available model types:

- `HUGGINGFACE_TRANSFORMERS`: Standard HuggingFace models
- `YOLO`: YOLO-based detection models
- `RTDETR`: RT-DETR models
- `LIBRARY_MANAGED`: External library models
- `VLM`: Vision Language Models
- `CUSTOM`: Custom implementations

## Best Practices

1. **Always use the registry**: Don't hardcode model paths or configurations
2. **Use type hints**: Leverage the type system for better IDE support
3. **Validate configurations**: Use Pydantic's validation features
4. **Document extra_config**: Document any custom configuration in `extra_config`
5. **Test model loading**: Ensure models can be downloaded and loaded correctly
6. **Version control**: Use the `version` field to track model versions

## Migration Guide

### Old Pattern (Before Standardization)

```python
class OldExtractor:
    MODEL_REPO = "some/model"
    MODEL_FILENAME = "model.pt"

    def __init__(self):
        # Hardcoded paths and configs
        self.model_path = "./models/some_model"
        self.conf_threshold = 0.5
```

### New Pattern (After Standardization)

```python
class NewExtractor:
    MODEL_ID = "some-model"

    def __init__(self):
        # Get from registry
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        self.model_path = ModelLoader.get_model_path(self.model_config)
        self.conf_threshold = self.model_config.confidence_threshold
```

## Examples

See the following files for complete examples:

- **Library-managed**: `omnidocs/tasks/layout_analysis/extractors/surya.py`
- **YOLO model**: `omnidocs/tasks/layout_analysis/extractors/doc_layout_yolo.py`
- **HuggingFace model**: `omnidocs/tasks/math_expression_extraction/extractors/nougat.py`
- **VLM (future)**: `omnidocs/tasks/vlm_processing/extractors/qwen_vl.py`

## Troubleshooting

### Model not found in registry

```python
# Check if model is registered
from omnidocs.core import ModelRegistry

all_models = ModelRegistry.list_all()
for model in all_models:
    print(model.id)
```

### Reload the registry

```python
# Force re-import to reload models
import importlib
from omnidocs.core import models_catalog

importlib.reload(models_catalog)
```

### Clear the registry (for testing)

```python
from omnidocs.core import ModelRegistry

ModelRegistry.clear()
```

## Contributing

When adding new models:

1. Add configuration to `omnidocs/core/models_catalog.py`
2. Register in `register_all_models()`
3. Update this documentation
4. Add tests for the new model configuration
5. Update the extractor to use the registry

## Future Enhancements

- [ ] Model versioning and updates
- [ ] Model performance benchmarks in config
- [ ] Automatic model selection based on requirements
- [ ] Model quantization configurations
- [ ] Remote model registry support
- [ ] Model caching strategies
