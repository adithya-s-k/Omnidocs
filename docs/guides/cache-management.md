---
title: "Cache Management"
description: "Configure model cache directories in OmniDocs"
category: "guides"
difficulty: "beginner"
time_estimate: "5 minutes"
keywords: ["cache", "storage", "models", "environment variables", "disk space"]
---

# Cache Management

OmniDocs provides unified cache directory management for all model weights across different backends (PyTorch, VLLM, MLX).

## Quick Start

Set the `OMNIDOCS_MODEL_CACHE` environment variable to control where all models are stored:

```bash
export OMNIDOCS_MODEL_CACHE=/data/models
```

All backends will now use `/data/models` for model storage.

## Environment Variables

### OMNIDOCS_MODEL_CACHE

Primary cache directory for all OmniDocs models.

**Priority order:**
1. `OMNIDOCS_MODEL_CACHE` (if set)
2. `HF_HOME` (if set)
3. Default: `~/.cache/huggingface`

**Example:**

```bash
# Store models on external drive
export OMNIDOCS_MODEL_CACHE=/Volumes/FastSSD/models

# Use in your code
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
)
```

### Backend-Specific Variables

OmniDocs automatically configures backend-specific cache variables:

| Backend | Environment Variable | Set By OmniDocs |
|---------|---------------------|-----------------|
| PyTorch/Transformers | `HF_HOME` | ✅ Yes |
| VLLM | `HF_HOME` + `download_dir` | ✅ Yes |
| MLX | `HF_HOME` | ✅ Yes |
| API | N/A (no local cache) | - |

## Per-Backend Configuration

You can override the global cache directory for specific backends using the `cache_dir` parameter:

### PyTorch Example

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        cache_dir="/mnt/fast-ssd/qwen-models"  # Override global cache
    )
)
```

### VLLM Example

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

extractor = QwenTextExtractor(
    backend=QwenTextVLLMConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        download_dir="/data/vllm-cache"  # VLLM uses download_dir
    )
)
```

### MLX Example

MLX uses `HF_HOME` environment variable (set automatically by OmniDocs):

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

# MLX respects OMNIDOCS_MODEL_CACHE via HF_HOME
extractor = QwenTextExtractor(
    backend=QwenTextMLXConfig(
        model="mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
)
```

## Programmatic Cache Control

### Get Cache Directory

```python
from omnidocs.utils.cache import get_model_cache_dir

# Get current cache directory
cache_dir = get_model_cache_dir()
print(f"Models stored in: {cache_dir}")

# Use custom directory
custom_cache = get_model_cache_dir("/data/custom-cache")
```

### Configure Backend Cache

```python
from omnidocs.utils.cache import configure_backend_cache

# Configure all backends with default cache
configure_backend_cache()

# Configure with custom directory
configure_backend_cache("/data/models")
```

### Get Cache Info

```python
from omnidocs.utils.cache import get_cache_info

info = get_cache_info()
print(info)
# {
#     'omnidocs_cache': '/data/models',
#     'omnidocs_model_cache_env': '/data/models',
#     'hf_home': '/data/models',
#     'transformers_cache': '/data/models'
# }
```

## Deployment Examples

### Local Development

```bash
# Store models on fast SSD
export OMNIDOCS_MODEL_CACHE=/Volumes/FastSSD/omnidocs-models
python my_script.py
```

### Modal Deployment

```python
import modal

app = modal.App("omnidocs-app")

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("omnidocs[pytorch]")
    .env({
        "OMNIDOCS_MODEL_CACHE": "/data/.cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume}
)
def process_document(pdf_path: str):
    from omnidocs import Document
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    # Models cached to /data/.cache (persisted via Modal volume)
    doc = Document.from_pdf(pdf_path)
    extractor = QwenTextExtractor(
        backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    )
    return extractor.extract(doc.get_page(0))
```

### Docker

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV OMNIDOCS_MODEL_CACHE=/app/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install omnidocs[pytorch]

VOLUME /app/models
```

```bash
docker run -v /mnt/models:/app/models my-omnidocs-image
```

## Cache Management Best Practices

### Disk Space Management

Monitor cache directory size:

```python
from omnidocs.utils.cache import get_model_cache_dir
from pathlib import Path

cache_dir = get_model_cache_dir()

def get_cache_size(directory: Path) -> float:
    """Get total size of cache directory in GB."""
    total = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    return total / (1024 ** 3)  # Convert to GB

print(f"Cache size: {get_cache_size(cache_dir):.2f} GB")
```

### Shared Cache Across Projects

Use a single cache for multiple projects:

```bash
# In ~/.bashrc or ~/.zshrc
export OMNIDOCS_MODEL_CACHE=/shared/ml-models/omnidocs

# All projects use same cache
cd project1 && python script1.py
cd project2 && python script2.py
```

### Fast Storage for Performance

Point cache to SSD for faster loading:

```bash
# Slow (HDD)
export OMNIDOCS_MODEL_CACHE=/mnt/hdd/models

# Fast (SSD)
export OMNIDOCS_MODEL_CACHE=/mnt/ssd/models

# Even faster (NVMe)
export OMNIDOCS_MODEL_CACHE=/mnt/nvme/models
```

### Cloud Storage Considerations

**Modal/Lambda/SageMaker:**
- Use persistent volumes for cache
- Set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads
- Pre-download models in Docker image to avoid cold starts

**Multi-node deployments:**
- Share cache via network filesystem (NFS, EFS)
- Or pre-bake models into container images

## Troubleshooting

### Models downloading to wrong location

Check environment variables:

```python
from omnidocs.utils.cache import get_cache_info
import pprint

pprint.pprint(get_cache_info())
```

### Disk full due to duplicate models

Consolidate to one cache:

```bash
# Find all HuggingFace caches
find ~ -name "huggingface" -type d

# Set unified cache
export OMNIDOCS_MODEL_CACHE=/data/unified-cache

# Optionally copy existing models
cp -r ~/.cache/huggingface/* /data/unified-cache/
```

### Permission errors

Ensure cache directory is writable:

```bash
chmod -R u+rwX,go+rX /data/models
chown -R $USER:$USER /data/models
```

## Migration Guide

### From HF_HOME to OMNIDOCS_MODEL_CACHE

Before (old way):
```bash
export HF_HOME=/data/models
```

After (new way):
```bash
export OMNIDOCS_MODEL_CACHE=/data/models
# HF_HOME is set automatically by OmniDocs
```

Both work, but `OMNIDOCS_MODEL_CACHE` is recommended for clarity.

### From Per-Backend Configs

Before (old way):
```python
# Different caches for each backend
pytorch_config = QwenTextPyTorchConfig(cache_dir="/data/pytorch-models")
vllm_config = QwenTextVLLMConfig(download_dir="/data/vllm-models")
```

After (new way):
```bash
# Unified cache via environment variable
export OMNIDOCS_MODEL_CACHE=/data/models
```

```python
# All backends use same cache
pytorch_config = QwenTextPyTorchConfig()
vllm_config = QwenTextVLLMConfig()
```

## See Also

- [Installation Guide](../installation.md)
- [Deployment Guide](../tasks/deployment.md)
- [Modal Deployment](../guides/deployment-modal.md)
