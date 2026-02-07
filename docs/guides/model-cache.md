# Model Cache

OmniDocs automatically shares loaded models across extractors using a unified cache. This avoids loading the same multi-GB model twice when you use it for different tasks.

---

## Why Cache?

Vision-language models like Qwen3-VL and MinerU VL support both text extraction and layout detection. Without caching, each extractor loads its own copy of the model, doubling GPU memory usage and load time.

| Scenario | Without cache | With cache |
|----------|--------------|------------|
| Qwen text + layout | 2 models loaded (~16GB) | 1 model shared (~8GB) |
| Load time | ~60s total | ~30s (second is instant) |

---

## Basic Usage

The cache works automatically. Just create extractors normally:

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

# Loads model (~30s)
text_extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Reuses cached model (instant)
layout_detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

# Both work independently
text_result = text_extractor.extract(image, output_format="markdown")
layout_result = layout_detector.extract(image)
```

---

## How Sharing Works

The cache normalizes config class names to detect when two extractors use the same model:

```
QwenTextPyTorchConfig   → Qwen:PyTorchConfig  (model_family:backend)
QwenLayoutPyTorchConfig → Qwen:PyTorchConfig  (same key = shared)
```

Task markers (`Text`, `Layout`, `OCR`, `Table`, `ReadingOrder`, `Formula`) are stripped from config names. If the remaining model family, backend type, and loading parameters match, the model is shared.

### Runtime Parameters Are Excluded

Parameters that only affect inference are excluded from the cache key:

- `max_tokens` / `max_new_tokens`
- `temperature`
- `do_sample`
- `timeout` / `max_retries`

This means a text extractor with `max_tokens=8192` and a layout detector with `max_tokens=4096` still share the same model, since `max_tokens` is a generation parameter, not a model loading parameter.

### Parameters That Affect the Cache Key

Parameters that change how the model is loaded produce different cache keys:

- `model` (model name/path)
- `device` / `device_map`
- `torch_dtype`
- `gpu_memory_utilization`
- `max_model_len`
- `tensor_parallel_size`
- `enforce_eager`

---

## Cache Management

### Inspecting the Cache

```python
from omnidocs import get_cache_info, list_cached_keys

# List all cached model keys
keys = list_cached_keys()
print(f"Cached models: {len(keys)}")
for key in keys:
    print(f"  {key}")

# Detailed info with reference counts
info = get_cache_info()
for key, entry in info["entries"].items():
    print(f"  {key}: refs={entry['ref_count']}, accesses={entry['access_count']}")
```

### Clearing the Cache

```python
from omnidocs import clear_cache, remove_cached

# Clear everything
clear_cache()

# Remove a specific entry
remove_cached("Qwen:PyTorchConfig:device=cuda:model=Qwen/Qwen3-VL-8B-Instruct:...")
```

### Setting Cache Size

```python
from omnidocs import set_cache_config

# Allow up to 5 models (default: 10)
set_cache_config(max_entries=5)

# Unlimited cache (watch memory usage)
set_cache_config(max_entries=0)
```

---

## LRU Eviction

When the cache is full, the least recently used model is evicted. Models with active references (extractors still using them) are evicted last.

```python
set_cache_config(max_entries=3)

# Load 3 models - cache full
ext1 = QwenTextExtractor(backend=QwenTextPyTorchConfig())      # slot 1
ext2 = MinerUVLTextExtractor(backend=MinerUVLTextPyTorchConfig())  # slot 2
ext3 = NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig())  # slot 3

# Loading a 4th evicts the least recently used
del ext1  # Qwen now has ref_count=0 and is eviction candidate
ext4 = GraniteDoclingTextExtractor(backend=...)  # Qwen gets evicted
```

---

## Reference Counting

Each extractor registers as a reference to its cached model. When an extractor is deleted (garbage collected), its reference is removed. Models with zero references are eligible for LRU eviction but are not immediately removed.

```python
# ref_count = 1
text_ext = QwenTextExtractor(backend=config)

# ref_count = 2 (same model, shared)
layout_det = QwenLayoutDetector(backend=config)

del text_ext   # ref_count = 1
del layout_det # ref_count = 0 (eligible for eviction)
```

---

## API Backends

API backends (e.g., `QwenTextAPIConfig`) are **not cached** because they don't load a local model. They just create a lightweight HTTP client.

---

## Supported Models

| Model | Type | What's cached |
|-------|------|---------------|
| **Qwen3-VL** | Multi-backend | `(model, processor)` |
| **MinerU VL** | Multi-backend | `(client, layout_size)` |
| **Nanonets OCR2** | Multi-backend | `(model, processor)` |
| **Granite Docling** | Multi-backend | `(model, processor)` |
| **DotsOCR** | Multi-backend | PyTorch: `(model, processor)`, VLLM: `(backend,)` |
| **RT-DETR** | Single-backend | `(model, processor)` |
| **DocLayout-YOLO** | Single-backend | `(model,)` |
| **PaddleOCR** | Single-backend | `(ocr_engine,)` |
| **EasyOCR** | Single-backend | `(reader,)` |
| **TableFormer** | Single-backend | `(predictor, config)` |
| **TesseractOCR** | Single-backend | Not cached (module import only) |
