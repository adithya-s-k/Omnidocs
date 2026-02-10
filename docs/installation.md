# Installation

Complete guide to installing OmniDocs and its dependencies.

---

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **OS**: Linux, macOS, Windows
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU**: Optional but recommended for VLM models

---

## Quick Install

```bash
# PyPI (recommended)
pip install omnidocs

# Or with uv (faster)
uv pip install omnidocs
```

**Verify installation:**
```python
from omnidocs import Document
print("OmniDocs installed successfully!")
```

---

## Install with Extras

OmniDocs uses optional dependencies to keep the base install lightweight.

```bash
# Core + PyTorch (most users)
pip install omnidocs[pytorch]

# Core + VLLM (high throughput)
pip install omnidocs[vllm]

# Core + MLX (Apple Silicon)
pip install omnidocs[mlx]

# Everything
pip install omnidocs[all]
```

### What Each Extra Includes

| Extra | Includes | Best For |
|-------|----------|----------|
| `pytorch` | PyTorch, Transformers, Accelerate | Local GPU inference |
| `vllm` | VLLM, PyTorch | Production, high throughput |
| `mlx` | MLX, mlx-lm | Apple Silicon (M1/M2/M3) |
| `ocr` | Tesseract bindings, EasyOCR, PaddleOCR | OCR tasks |
| `all` | All of the above | Full functionality |

!!! note "API support included by default"
    LiteLLM is now a core dependency. Cloud API access (Gemini, OpenRouter, Azure, OpenAI) works with a base `pip install omnidocs` -- no extra needed.

---

## Install from Source

For development or latest features:

```bash
# Clone repository
git clone https://github.com/adithya-s-k/Omnidocs.git
cd Omnidocs

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Configure Model Cache

OmniDocs provides unified cache directory management for all model weights.

**Set cache directory:**
```bash
export OMNIDOCS_MODELS_DIR=/data/models
```

**Default cache locations:**
1. `$OMNIDOCS_MODELS_DIR` (if set)
2. `$HF_HOME` (if set)
3. `~/.cache/huggingface` (default)

**Examples:**

```bash
# Store models on external SSD
export OMNIDOCS_MODELS_DIR=/Volumes/FastSSD/models

# Store in custom directory
export OMNIDOCS_MODELS_DIR=/mnt/data/omnidocs-cache

# Check cache location
python -c "from omnidocs.utils.cache import get_storage_info; import pprint; pprint.pprint(get_storage_info())"
```

**See also:** [Cache Management Guide](guides/cache-management.md) for advanced configuration.

---

## Backend-Specific Setup

### PyTorch Backend

Most common setup for local GPU inference.

```bash
pip install omnidocs[pytorch]
```

**Requirements:**
- NVIDIA GPU with CUDA support (optional but recommended)
- CUDA 11.8+ for GPU acceleration

**Verify:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

### VLLM Backend

High-throughput inference for production.

```bash
pip install omnidocs[vllm]
```

**Requirements:**
- Linux (VLLM has limited Windows/macOS support)
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.0+

**Verify:**
```python
from vllm import LLM
print("VLLM installed successfully!")
```

---

### MLX Backend

Optimized for Apple Silicon Macs.

```bash
pip install omnidocs[mlx]
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- macOS 13.0+

**Verify:**
```python
import mlx.core as mx
print(f"MLX installed, default device: {mx.default_device()}")
```

---

### API Backend

No GPU required -- uses cloud VLMs via LiteLLM (included by default).

```bash
pip install omnidocs
```

**Supported providers:**

- Google Gemini
- OpenRouter
- Azure OpenAI
- OpenAI
- Any OpenAI-compatible API (including self-hosted VLLM)

**Setup:**
```bash
# Set the env var for your provider
export GOOGLE_API_KEY=...        # Gemini
export OPENROUTER_API_KEY=...    # OpenRouter
export AZURE_API_KEY=...         # Azure
export OPENAI_API_KEY=...        # OpenAI
```

**Verify:**
```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
extractor = VLMTextExtractor(config=config)
result = extractor.extract("document.png")
print(result.content[:200])
```

---

## OCR Dependencies

### Tesseract

Tesseract requires a system installation:

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

**Install additional languages:**
```bash
# Ubuntu
sudo apt install tesseract-ocr-chi-sim  # Chinese
sudo apt install tesseract-ocr-jpn      # Japanese
sudo apt install tesseract-ocr-ara      # Arabic

# macOS
brew install tesseract-lang
```

**Verify:**
```bash
tesseract --version
tesseract --list-langs
```

### EasyOCR

```bash
pip install easyocr
```

Downloads models automatically on first use (~100MB per language).

### PaddleOCR

```bash
pip install paddlepaddle paddleocr
```

For GPU support:
```bash
pip install paddlepaddle-gpu paddleocr
```

---

## Flash Attention (Optional)

Flash Attention 2 accelerates VLM inference. **Optional but recommended** for production.

### Requirements

- CUDA 11.8+ (12.3+ for FA3)
- PyTorch 2.0+
- NVIDIA GPU with compute capability 7.0+ (V100, A100, RTX 3090, etc.)
- Linux

### Option 1: Pre-built Wheels (Recommended)

Download matching wheel from [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases):

```bash
# Example: Python 3.12, CUDA 12, PyTorch 2.5
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

**Wheel naming:**
- `cp312` = Python 3.12 (cp311 for 3.11, cp310 for 3.10)
- `cu12` = CUDA 12.x (cu118 for CUDA 11.8)
- `torch2.5` = PyTorch 2.5.x

### Option 2: Compile from PyPI

```bash
pip install flash-attn --no-build-isolation
```

Speed up compilation:
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Option 3: Build from Source

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install ninja
python setup.py install
```

### Verify Flash Attention

```python
import torch
from flash_attn import flash_attn_func

print(f"Flash Attention installed: {flash_attn_func is not None}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Skip Flash Attention

If installation fails, use VLLM backend instead (includes optimized attention):

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig

# VLLM includes optimized attention
extractor = QwenTextExtractor(
    backend=QwenVLLMConfig(model="Qwen/Qwen3-VL-8B-Instruct")
)
```

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Use smaller model
pip install omnidocs[pytorch]
# Then use Qwen3-VL-2B instead of 8B
```

### "No module named 'omnidocs'"

```bash
# Reinstall
pip uninstall omnidocs
pip install omnidocs
```

### "tesseract not found"

```bash
# Install system package
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Linux
```

### PyTorch/CUDA version mismatch

```bash
# Check versions
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Reinstall matching versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Flash Attention compilation fails

```bash
# Install build tools
sudo apt install build-essential ninja-build

# Or use pre-built wheel instead
```

---

## Environment Setup

### Recommended: Virtual Environment

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install
pip install omnidocs[pytorch]
```

### With uv (Faster)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install omnidocs[pytorch]
```

### With conda

```bash
conda create -n omnidocs python=3.11
conda activate omnidocs
pip install omnidocs[pytorch]
```

---

## Verify Full Installation

```python
from omnidocs import Document
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

# Load a test image
from PIL import Image
import requests
from io import BytesIO

url = "https://raw.githubusercontent.com/adithya-s-k/OmniDocs/main/assets/sample_page.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Test layout extraction
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
result = layout.extract(image)

print(f"Detected {len(result.bboxes)} elements")
print("OmniDocs is ready!")
```

---

## Next Steps

- [Getting Started](getting-started.md) - First extraction in 5 minutes
- [Usage](usage/index.md) - Tasks, models, and workflows
