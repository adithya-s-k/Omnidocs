# Deployment

Deploy OmniDocs on Modal serverless GPUs.

---

## Why Modal

- **No infrastructure** - Modal handles GPU provisioning
- **Pay per use** - Only pay for GPU time used
- **Auto scaling** - Handles traffic spikes

**Cost:** ~$0.35/hour for A10G. 100 pages ≈ $1.

---

## Setup

```bash
# Install
pip install modal

# Authenticate
modal token new

# Create volume for model caching
modal volume create omnidocs

# Create HuggingFace secret
modal secret create adithya-hf-wandb --key HF_TOKEN --value "hf_xxx..."
```

---

## Basic Deployment

```python
import modal
from typing import Dict, Any

# Image configuration
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch", "torchvision", "transformers", "pillow",
        "pydantic", "huggingface_hub", "accelerate",
    )
    .uv_pip_install("qwen-vl-utils")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/.cache"})
)

volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

app = modal.App("omnidocs-extraction")


@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_text(image_bytes: bytes) -> Dict[str, Any]:
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes))

    config = QwenPyTorchConfig(device="cuda")
    extractor = QwenTextExtractor(backend=config)

    result = extractor.extract(image, output_format="markdown")

    return {
        "success": True,
        "word_count": result.word_count,
        "content": result.content,
    }


@app.local_entrypoint()
def main():
    with open("test.png", "rb") as f:
        image_bytes = f.read()

    result = extract_text.remote(image_bytes)
    print(f"Words: {result['word_count']}")
```

**Run:**
```bash
modal run script.py
```

---

## Batch Processing

```python
@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    secrets=[secret],
    timeout=1800,  # 30 min
)
def process_batch(image_bytes_list: list) -> Dict[str, Any]:
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig
    from PIL import Image
    import io

    extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))

    results = []
    for image_bytes in image_bytes_list:
        image = Image.open(io.BytesIO(image_bytes))
        result = extractor.extract(image, output_format="markdown")
        results.append({"word_count": result.word_count})

    return {"results": results}
```

---

## Multi-GPU with VLLM

```python
@app.function(
    image=IMAGE,
    gpu="A10G:2",  # 2 GPUs
    volumes={"/data": volume},
    secrets=[secret],
    timeout=600,
)
def extract_vllm(image_bytes: bytes):
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes))

    config = QwenVLLMConfig(
        tensor_parallel_size=2,  # Both GPUs
        gpu_memory_utilization=0.9,
    )
    extractor = QwenTextExtractor(backend=config)

    result = extractor.extract(image, output_format="markdown")
    return {"content": result.content}
```

---

## GPU Options

| GPU | $/hour | VRAM | Best For |
|-----|--------|------|----------|
| T4 | $0.15 | 16GB | Budget |
| A10G | $0.35 | 24GB | General (recommended) |
| A40 | $1.10 | 48GB | Large models |

---

## Troubleshooting

**Model download stuck**
- Check HF token in secret
- Increase timeout for first run

**CUDA out of memory**
- Use larger GPU (A40)
- Use smaller model (2B)
- Use VLLM with tensor parallelism

**Timeout errors**
- Increase `timeout` parameter
- Reduce batch size
