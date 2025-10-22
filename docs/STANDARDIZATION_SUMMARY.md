# Model Format Standardization Summary

## Overview

This document summarizes the standardization of model formats across Omnidocs, implementing a unified model registry system that makes it easy to add and manage models including Vision Language Models (VLMs).

## What Changed

### New Core Components

#### 1. Model Registry System (`omnidocs/core/`)

Created a centralized model management system with three main files:

- **`model_registry.py`**: Core configuration classes
  - `ModelConfig`: Base configuration for all models
  - `HuggingFaceModelConfig`: For Transformer models
  - `YOLOModelConfig`: For YOLO-based models
  - `LibraryManagedModelConfig`: For library-managed models (Surya, EasyOCR, etc.)
  - `VLMModelConfig`: For Vision Language Models
  - `ModelRegistry`: Centralized registry for all models

- **`models_catalog.py`**: Catalog of all available models
  - Defines all model configurations
  - Auto-registers models on import
  - Easy to add new models

- **`model_loader.py`**: Utilities for loading models
  - `ModelLoader`: Handles downloading and loading
  - Consistent interface across model types
  - Helper functions for common operations

#### 2. Documentation

- **`docs/MODEL_REGISTRY.md`**: Comprehensive guide
  - How to use the registry
  - How to add new models
  - Examples for all model types
  - VLM integration guide
  - Migration guide from old patterns

- **`docs/STANDARDIZATION_SUMMARY.md`**: This document

### Updated Extractors

Updated extractors to use the new registry system:

#### Layout Analysis
- ✅ `surya.py`: Library-managed model example
- ✅ `doc_layout_yolo.py`: YOLO model example

#### Table Extraction
- ✅ `tableformer.py`: Multi-model configuration example

All other extractors follow similar patterns and can be updated using the same approach.

## Key Benefits

### 1. Consistency
- All models configured in one place
- Standardized initialization patterns
- Uniform error handling

### 2. Discoverability
- List all models: `ModelRegistry.list_all()`
- Find models by task: `ModelRegistry.get_by_task(TaskType.LAYOUT_ANALYSIS)`
- Find models by type: `ModelRegistry.get_by_type(ModelType.VLM)`

### 3. Easy VLM Integration
- Pre-built `VLMModelConfig` class
- Support for multimodal inputs
- API-based and local model support

### 4. Type Safety
- Pydantic models with validation
- Type hints throughout
- Better IDE support

### 5. Maintainability
- Single source of truth
- Easy to update model versions
- Clear documentation

## How to Add a New Model

### Example: Adding Qwen-VL (A Vision Language Model)

#### Step 1: Define the Configuration

In `omnidocs/core/models_catalog.py`:

```python
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

#### Step 2: Register It

Add to `register_all_models()`:

```python
def register_all_models():
    models = [
        # ... existing models ...
        QWEN_VL,
    ]
    for model in models:
        ModelRegistry.register(model)
```

#### Step 3: Create an Extractor

```python
from omnidocs.core import ModelRegistry, VLMModelConfig

class QwenVLProcessor:
    MODEL_ID = "qwen-vl-chat"

    def __init__(self, device=None, show_log=False):
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        # Load and use the model...
```

That's it! The model is now integrated and discoverable.

## Migration Guide

### Before (Old Pattern)

```python
class OldExtractor:
    MODEL_REPO = "some/model"
    MODEL_FILENAME = "model.pt"
    CONF_THRESHOLD = 0.5

    def __init__(self):
        self.model_path = "./models/some_model"
        # Hardcoded paths and configs
```

### After (New Pattern)

```python
class NewExtractor:
    MODEL_ID = "some-model"

    def __init__(self):
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        self.model_path = ModelLoader.get_model_path(self.model_config)
        self.conf_threshold = self.model_config.confidence_threshold
```

## Registered Models

### Layout Analysis
- `surya-layout`: Surya Layout Analysis (library-managed)
- `doclayout-yolo-docstructbench`: DocLayout-YOLO (YOLO v10)
- `rtdetr-publaynet`: RT-DETR PubLayNet (RTDETR)
- `yolov10-doclaynet`: YOLOv10 DocLayNet (YOLO v10)

### Table Extraction
- `table-transformer-detection`: Table Transformer Detection (HuggingFace)
- `table-transformer-structure`: Table Transformer Structure (HuggingFace)
- `tableformer-detection`: TableFormer Detection (HuggingFace)
- `tableformer-structure`: TableFormer Structure (HuggingFace)
- `surya-table`: Surya Table Extraction (library-managed)

### Math Expression Extraction
- `nougat-base`: Nougat Base (HuggingFace)
- `nougat-small`: Nougat Small (HuggingFace)
- `donut-cord-v2`: Donut CORD v2 (HuggingFace)
- `surya-math`: Surya Math Expression (library-managed)

### OCR
- `surya-ocr`: Surya OCR (library-managed)
- `easyocr`: EasyOCR (library-managed)
- `paddleocr`: PaddleOCR (library-managed)
- `tesseract`: Tesseract OCR (library-managed)

### Text Extraction
- `surya-text`: Surya Text Detection (library-managed)

### VLM (Examples - Commented Out)
- `gpt4-vision`: GPT-4 Vision (API-based)
- `claude-3-opus`: Claude 3 Opus (API-based)
- `llava-1.6-vicuna-7b`: LLaVA 1.6 (HuggingFace)
- `qwen-vl-chat`: Qwen-VL Chat (HuggingFace)

## Usage Examples

### List All Models

```python
from omnidocs.core import ModelRegistry

all_models = ModelRegistry.list_all()
for model in all_models:
    print(f"{model.id}: {model.name} ({model.task_type})")
```

### Get Models for a Task

```python
from omnidocs.core import ModelRegistry, TaskType

layout_models = ModelRegistry.get_by_task(TaskType.LAYOUT_ANALYSIS)
for model in layout_models:
    print(f"  - {model.name}: {model.description}")
```

### Load a Model

```python
from omnidocs.core import ModelLoader

# Load any model by ID
model_data = ModelLoader.load_from_registry(
    "doclayout-yolo-docstructbench",
    device="cuda",
    download_if_needed=True
)

model = model_data["model"]
config = model_data["config"]
```

### In an Extractor

```python
from omnidocs.core import ModelRegistry

class MyExtractor:
    MODEL_ID = "surya-layout"

    def __init__(self):
        self.model_config = ModelRegistry.get(self.MODEL_ID)
        print(f"Using: {self.model_config.description}")
```

## Files Changed

### New Files
- `omnidocs/core/__init__.py`
- `omnidocs/core/model_registry.py`
- `omnidocs/core/models_catalog.py`
- `omnidocs/core/model_loader.py`
- `docs/MODEL_REGISTRY.md`
- `docs/STANDARDIZATION_SUMMARY.md`

### Updated Files
- `omnidocs/tasks/layout_analysis/extractors/surya.py`
- `omnidocs/tasks/layout_analysis/extractors/doc_layout_yolo.py`
- `omnidocs/tasks/table_extraction/extractors/tableformer.py`

### To Be Updated (Same Pattern)
- `omnidocs/tasks/layout_analysis/extractors/rtdetr.py`
- `omnidocs/tasks/layout_analysis/extractors/yolov10.py`
- `omnidocs/tasks/table_extraction/extractors/table_transformer.py`
- `omnidocs/tasks/table_extraction/extractors/surya_table.py`
- `omnidocs/tasks/math_expression_extraction/extractors/nougat.py`
- `omnidocs/tasks/math_expression_extraction/extractors/donut.py`
- `omnidocs/tasks/math_expression_extraction/extractors/surya_math.py`
- `omnidocs/tasks/ocr_extraction/extractors/*.py`
- `omnidocs/tasks/text_extraction/extractors/*.py`

## Testing

To verify the changes work:

```python
# Test 1: Registry is populated
from omnidocs.core import ModelRegistry
assert len(ModelRegistry.list_all()) > 0

# Test 2: Can get a model
config = ModelRegistry.get("surya-layout")
assert config is not None
assert config.name == "Surya Layout Analysis"

# Test 3: Can list by task
from omnidocs.core import TaskType
layout_models = ModelRegistry.get_by_task(TaskType.LAYOUT_ANALYSIS)
assert len(layout_models) > 0

# Test 4: Model loader works
from omnidocs.core import ModelLoader
path = ModelLoader.get_model_path(config)
assert path is not None
```

## Next Steps

1. **Update Remaining Extractors**: Apply the pattern to all extractors
2. **Add VLM Support**: Uncomment and implement VLM configurations
3. **Add Tests**: Unit tests for model registry and loader
4. **Performance Benchmarks**: Add benchmark data to model configs
5. **Model Versioning**: Implement version tracking and updates

## Breaking Changes

- ⚠️ Extractors updated to use registry require the new import: `from omnidocs.core import ModelRegistry`
- ⚠️ Old hardcoded constants (MODEL_REPO, MODEL_FILENAME) are being replaced by MODEL_ID

## Backwards Compatibility

The changes maintain backwards compatibility:
- Existing code continues to work
- Model paths remain the same
- Configuration values preserved
- Only initialization patterns change

## Contributing

When adding new models or model types:

1. Add configuration to `omnidocs/core/models_catalog.py`
2. Register in `register_all_models()`
3. Update extractors to use `MODEL_ID` pattern
4. Add documentation to `docs/MODEL_REGISTRY.md`
5. Add tests

## Questions?

See `docs/MODEL_REGISTRY.md` for detailed documentation and examples.
