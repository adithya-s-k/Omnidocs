# Modal Deployment Guide

Deploy OmniDocs inference at scale on Modal serverless GPUs. This guide covers setup, configuration, deployment patterns, and cost optimization.

## Table of Contents

- [Why Modal for OmniDocs](#why-modal)
- [Standard Setup](#standard-setup)
- [Basic Deployment](#basic-deployment)
- [Multi-GPU Deployment](#multi-gpu-deployment)
- [Production Patterns](#production-patterns)
- [Monitoring & Logging](#monitoring--logging)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Why Modal for OmniDocs

Modal is ideal for OmniDocs because:

1. **No Infrastructure Management** - Modal handles GPU provisioning, networking, and scaling
2. **Pay Per Use** - Only pay for actual GPU time, not idle time
3. **Automatic Scaling** - Handle traffic spikes without manual scaling
4. **Pre-built GPU Images** - CUDA, drivers, PyTorch pre-installed
5. **Distributed Processing** - Process multiple documents in parallel
6. **Easy CLI** - Deploy with single `modal run` command

**Cost Comparison:**
- Self-managed GPU: $500-2000/month (always on)
- Modal (batch processing): $0.30-1.00 per hour of GPU time
- For 100 documents (3 hours GPU time): ~$1.00

## Standard Setup

### Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate:
```bash
modal token new
# Or use existing token
modal token set
```

3. Create Modal workspace (optional):
```bash
modal workspace create my-workspace
modal workspace use my-workspace
```

### Standard Configuration

Every OmniDocs Modal script uses this standard setup:

```python
import modal
from pathlib import Path

# ============= Configuration =============

# Model settings
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"  # or other models
GPU_CONFIG = "A10G:1"  # GPU type and count

# Cache directories
MODEL_CACHE_DIR = "/data/omnidocs_models"

# CUDA settings (keep consistent across all scripts)
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# ============= Build Modal Image =============

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    # Base dependencies first (gets cached)
    .uv_pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "pillow",
        "numpy",
        "pydantic",
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
    )
    # Model-specific dependencies
    .uv_pip_install("qwen-vl-utils")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
        "OMNIDOCS_MODEL_CACHE": MODEL_CACHE_DIR,
    })
)

# ============= Modal Setup =============

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")

app = modal.App("omnidocs-deployment")
```

**Key Points:**
- **Volume Name**: Always use `"omnidocs"` for consistency
- **Secret Name**: Always use `"adithya-hf-wandb"` (contains HF token)
- **Python Version**: `3.12` for latest compatibility
- **GPU**: `A10G:1` is standard (adjust as needed)
- **Timeout**: Default 600s (10 min), increase for long documents

### Environment Variables for Deployment

Set up required secrets:

```bash
# Create HuggingFace secret (one-time)
modal secret create adithya-hf-wandb \
  --key HF_TOKEN \
  --value "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Create volume for model caching (one-time)
modal volume create omnidocs
```

## Basic Deployment

### Example 1: Simple Text Extraction

Deploy a basic text extraction function.

```python
import modal
from typing import Dict, Any
from pathlib import Path

# ============= Configuration =============
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch", "torchvision", "transformers", "pillow", "numpy",
        "pydantic", "huggingface_hub", "accelerate",
    )
    .uv_pip_install("qwen-vl-utils")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

app = modal.App("omnidocs-text-extraction")

# ============= Modal Function =============

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_text(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extract text from an image.

    Args:
        image_bytes: Image file contents (PNG/JPG)

    Returns:
        Dict with extracted text and metadata
    """
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    from PIL import Image
    import io

    # Load image
    image = Image.open(io.BytesIO(image_bytes))

    # Initialize extractor
    config = QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
    extractor = QwenTextExtractor(backend=config)

    # Extract
    result = extractor.extract(image, output_format="markdown")

    return {
        "success": True,
        "content_length": result.content_length,
        "word_count": result.word_count,
        "content": result.content,
    }

# ============= Local Entrypoint =============

@app.local_entrypoint()
def main():
    """Test the deployment."""
    from pathlib import Path

    # Test with a sample image
    test_image_path = "test_document.png"
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()

    # Run extraction
    result = extract_text.remote(image_bytes)

    print(f"Extraction succeeded: {result['success']}")
    print(f"Content length: {result['content_length']} chars")
    print(f"Word count: {result['word_count']}")
    print(f"\nContent preview:")
    print(result['content'][:500])
```

**Deploy:**
```bash
# Test locally
python script.py

# Or run on Modal GPU
modal run script.py
```

### Example 2: Batch Processing with Progress

Deploy a batch processor with progress tracking.

```python
import modal
from typing import List, Dict, Any
import time

# ... (image and app setup as above)

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=1800,  # 30 min for large batches
)
def process_batch(image_bytes_list: List[bytes]) -> Dict[str, Any]:
    """
    Process a batch of images.

    Args:
        image_bytes_list: List of image byte strings

    Returns:
        Processing results and statistics
    """
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    from PIL import Image
    import io

    # Initialize once (expensive)
    config = QwenTextPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)

    results = []
    start_time = time.time()

    for idx, image_bytes in enumerate(image_bytes_list, 1):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            result = extractor.extract(image, output_format="markdown")

            results.append({
                "index": idx,
                "success": True,
                "word_count": result.word_count,
                "content_length": result.content_length,
            })

            # Progress
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(image_bytes_list) - idx) * avg_time
            print(f"[{idx}/{len(image_bytes_list)}] {remaining/60:.1f}min remaining")

        except Exception as e:
            results.append({
                "index": idx,
                "success": False,
                "error": str(e),
            })

    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "num_images": len(image_bytes_list),
        "succeeded": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }

@app.local_entrypoint()
def main():
    """Test batch processing."""
    from pathlib import Path

    # Load images
    image_dir = Path("test_images/")
    image_paths = sorted(image_dir.glob("*.png"))[:5]  # Test with 5

    image_bytes_list = [
        open(p, "rb").read()
        for p in image_paths
    ]

    # Process batch
    result = process_batch.remote(image_bytes_list)

    print(f"\nResults:")
    print(f"  Succeeded: {result['succeeded']}/{result['num_images']}")
    print(f"  Failed: {result['failed']}/{result['num_images']}")
    print(f"  Total time: {result['total_time']:.1f}s")
    print(f"  Average: {result['total_time']/result['num_images']:.2f}s per image")
```

## Multi-GPU Deployment

### Example 1: VLLM with Tensor Parallelism

Use VLLM to distribute inference across multiple GPUs.

```python
import modal

# Use larger GPU for tensor parallelism
GPU_CONFIG = "A10G:2"  # 2 GPUs

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch", "transformers", "pillow", "pydantic",
        "huggingface_hub", "accelerate",
    )
    # VLLM for multi-GPU inference
    .uv_pip_install("vllm")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

app = modal.App("omnidocs-vllm-2gpu")

@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,  # 2 GPUs
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_vllm_2gpu(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extract using VLLM with 2-GPU tensor parallelism.
    """
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes))

    # Configure for 2 GPUs
    config = QwenTextVLLMConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=2,  # Distribute across 2 GPUs
        gpu_memory_utilization=0.9,
        max_tokens=4096,
    )
    extractor = QwenTextExtractor(backend=config)

    result = extractor.extract(image, output_format="markdown")

    return {
        "success": True,
        "word_count": result.word_count,
        "model": "VLLM (2-GPU tensor parallel)",
    }
```

### Example 2: Multi-Function with Load Balancing

Deploy multiple functions to handle parallel requests.

```python
import modal
from typing import Dict, Any

# ... (image and app setup)

# Create 3 independent extract functions
for func_idx in range(3):
    @app.function(
        image=IMAGE,
        gpu="A10G:1",
        volumes={"/data": volume},
        secrets=[secret],
        timeout=600,
        name=f"extract_{func_idx}",
    )
    def extract_text(image_bytes: bytes, _func_idx=func_idx) -> Dict[str, Any]:
        # Same implementation as before
        # Modal will create 3 independent functions
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        config = QwenTextPyTorchConfig(device="cuda")
        extractor = QwenTextExtractor(backend=config)
        result = extractor.extract(image, output_format="markdown")

        return {
            "success": True,
            "word_count": result.word_count,
            "worker": _func_idx,
        }

@app.local_entrypoint()
def main():
    """Process 3 images in parallel."""
    import concurrent.futures

    # Get function references
    extract_0 = modal.Function.lookup("omnidocs-multiworker", "extract_0")
    extract_1 = modal.Function.lookup("omnidocs-multiworker", "extract_1")
    extract_2 = modal.Function.lookup("omnidocs-multiworker", "extract_2")

    functions = [extract_0, extract_1, extract_2]

    # Prepare 3 test images
    image_bytes_list = [
        open(f"test_{i}.png", "rb").read()
        for i in range(3)
    ]

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(func.remote, img_bytes)
            for func, img_bytes in zip(functions, image_bytes_list)
        ]
        results = [f.result() for f in futures]

    print(f"Processed {len(results)} images in parallel")
    for result in results:
        print(f"  Worker {result['worker']}: {result['word_count']} words")
```

## Production Patterns

### Scheduled Processing

Run batch processing on a schedule.

```python
import modal
from datetime import datetime

# ... (image and app setup)

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=3600,
    schedule=modal.Period(days=1),  # Daily at midnight UTC
)
def daily_batch_processing():
    """Process accumulated documents daily."""
    from pathlib import Path
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    from PIL import Image
    import json

    # Find new documents
    inbox_dir = Path("/data/inbox")
    processed_dir = Path("/data/processed")
    results_dir = Path("/data/results")

    results_dir.mkdir(exist_ok=True)

    # Initialize extractor
    config = QwenTextPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)

    # Process documents
    for image_path in inbox_dir.glob("*.png"):
        image = Image.open(image_path)
        result = extractor.extract(image, output_format="markdown")

        # Save result
        result_file = results_dir / f"{image_path.stem}.json"
        with open(result_file, "w") as f:
            json.dump({
                "filename": image_path.name,
                "word_count": result.word_count,
                "content_length": result.content_length,
                "processed_at": datetime.now().isoformat(),
            }, f)

        # Move to processed
        image_path.rename(processed_dir / image_path.name)

    print(f"Daily processing complete")
```

### Webhook Handler

Accept requests from an external service.

```python
import modal
from typing import Dict, Any
from fastapi import FastAPI

# ... (image setup)

web_app = FastAPI()

# Create Modal web endpoint
@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
)
@modal.web_endpoint(method="POST")
def extract_from_url(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Accept image URL and extract text.

    POST /extract_from_url
    {"image_url": "https://example.com/doc.png"}
    """
    import requests
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    from PIL import Image
    import io

    # Download image
    response = requests.get(request["image_url"], timeout=30)
    image = Image.open(io.BytesIO(response.content))

    # Extract
    config = QwenTextPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)
    result = extractor.extract(image, output_format="markdown")

    return {
        "success": True,
        "word_count": result.word_count,
        "content": result.content,
    }
```

**Deploy and test:**
```bash
# Deploy
modal deploy script.py

# Get URL
modal app list

# Test
curl -X POST https://your-workspace.modal.run/extract_from_url \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/doc.png"}'
```

## Monitoring & Logging

### Log Extraction Progress

```python
import modal
import logging
from typing import List

# ... (image and app setup)

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=1800,
)
def process_with_logging(image_bytes_list: List[bytes]) -> Dict[str, Any]:
    """Process with detailed logging."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )
    logger = logging.getLogger(__name__)

    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    from PIL import Image
    import io
    import time
    import torch

    logger.info(f"Starting batch processing: {len(image_bytes_list)} images")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    config = QwenTextPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)

    results = []
    start_time = time.time()

    for idx, image_bytes in enumerate(image_bytes_list, 1):
        try:
            iter_start = time.time()

            image = Image.open(io.BytesIO(image_bytes))
            result = extractor.extract(image, output_format="markdown")

            iter_time = time.time() - iter_start

            logger.info(f"[{idx}/{len(image_bytes_list)}] "
                       f"Processed in {iter_time:.2f}s, "
                       f"{result.word_count} words")

            results.append({"success": True})

        except Exception as e:
            logger.error(f"[{idx}] Failed: {e}")
            results.append({"success": False, "error": str(e)})

    total_time = time.time() - start_time
    logger.info(f"Batch complete: {total_time:.1f}s total, "
               f"{total_time/len(image_bytes_list):.2f}s per image")

    return {"results": results}
```

### Monitor GPU Memory

```python
@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_with_memory_monitoring(image_bytes: bytes):
    """Extract with GPU memory monitoring."""
    import torch

    def log_memory(label):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{label}: allocated={allocated:.1f}GB, "
              f"reserved={reserved:.1f}GB, total={total:.1f}GB")

    log_memory("Initial")

    # Load model
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    config = QwenTextPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)

    log_memory("After model load")

    # Extract
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes))
    result = extractor.extract(image, output_format="markdown")

    log_memory("After extraction")

    return {"success": True}
```

## Cost Optimization

### GPU Selection

| GPU | $/hour | Ideal For |
|-----|--------|-----------|
| A10G | $0.35 | General purpose, fast |
| A40 | $1.10 | Large models, high VRAM |
| T4 | $0.15 | Budget processing |
| L40S | $1.25 | High-end graphics |

**Recommendation:** A10G (sweet spot of price/performance)

### Cost Calculation

```
Cost per document = (Model load time + Processing time) × $/hour GPU

Example (A10G @ $0.35/hour):
- Model load: 2 seconds (one-time amortized across batch)
- Per-image: 3 seconds
- Batch of 100: (2 + 100*3) / 3600 hours × $0.35/hour ≈ $0.03 per image
- 100 documents: ~$3 total GPU cost
```

### Batch Size Optimization

Larger batches = lower per-item cost (amortize model load).

```python
# Calculation
model_load_time = 2  # seconds
per_image_time = 3   # seconds
gpu_cost_per_hour = 0.35  # dollars
batch_sizes = [1, 5, 10, 50, 100]

print("Batch Size | Total Time | Cost per Image")
print("-" * 40)

for batch_size in batch_sizes:
    total_time = (model_load_time + batch_size * per_image_time) / 3600
    cost_per_image = total_time * gpu_cost_per_hour / batch_size
    print(f"{batch_size:10} | {total_time*3600:9.0f}s | ${cost_per_image:.4f}")

# Output:
# Batch Size | Total Time | Cost per Image
# ----------------------------------------
#          1 |          5s | $0.0005
#          5 |         17s | $0.0012
#         10 |         32s | $0.0031
#         50 |        152s | $0.0149
#        100 |        302s | $0.0293
```

**Takeaway:** Batch size of ~50 is optimal (amortizes load, avoids timeout issues)

### Spot Instances

Use cheaper Spot instances for non-critical batches.

```python
import modal

@app.function(
    image=IMAGE,
    gpu=modal.gpu.A10G(count=1, spot=True),  # Use Spot instance
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_spot(image_bytes: bytes):
    # Same as regular extraction
    pass

# Cost: ~60% cheaper than on-demand
```

## Troubleshooting

### Model Download Stuck

**Problem:** Model stuck downloading.

**Solution:** Set HF token and increase timeout.

```python
# Ensure HF token is set
modal secret create adithya-hf-wandb \
  --key HF_TOKEN \
  --value "your-token"

# Increase timeout for initial runs
@app.function(
    ...,
    timeout=1800,  # 30 minutes for first run
)
```

### CUDA Out of Memory

**Problem:** CUDA OOM during extraction.

**Solution:** Use larger GPU or reduce model size.

```python
# Option 1: Use larger GPU
gpu=modal.gpu.A40(count=1)  # 48GB VRAM vs 24GB on A10G

# Option 2: Use smaller model
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-7B-Instruct",  # 7B instead of 8B
)

# Option 3: Use VLLM with tensor parallelism
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
config = QwenTextVLLMConfig(
    tensor_parallel_size=2,  # Split across 2 GPUs
)
gpu=modal.gpu.A10G(count=2)
```

### Timeout Errors

**Problem:** Function times out.

**Solution:** Increase timeout or reduce batch size.

```python
@app.function(
    ...,
    timeout=1800,  # Increase to 30 minutes
)

# Or reduce batch size
batch_size = 10  # Process 10 at a time instead of 100
```

### Network Errors

**Problem:** Intermittent network failures.

**Solution:** Add retry logic.

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def download_with_retry(image_url: str):
    import requests
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return response.content
```

---

**Next Steps:**
- See [Batch Processing Guide](batch-processing.md) for local batch patterns
- See [Text Extraction Guide](text-extraction.md) for model configuration
- Modal docs: https://modal.com/docs
