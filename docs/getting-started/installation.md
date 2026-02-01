# Installation Guide

Welcome to OmniDocs! This guide walks you through installing OmniDocs and choosing the right backend for your use case.

## System Requirements

Before installing, ensure you have:

- **Python**: 3.10, 3.11, or 3.12
- **pip** or **uv** (recommended for faster installation)
- **4 GB RAM** minimum (8 GB+ recommended for large documents)
- **GPU** (optional, but highly recommended for fast inference)

## Quick Install

The fastest way to get started:

```bash
# Basic installation (PyTorch backend, most common)
pip install omnidocs[pytorch]

# Or with uv (faster)
uv pip install omnidocs[pytorch]
```

This installs OmniDocs with PyTorch support for local GPU inference on NVIDIA/AMD GPUs and Apple Silicon.

## Choosing Your Backend

OmniDocs supports **4 inference backends**. Choose one based on your needs:

| Backend | Best For | GPU Required | Setup Time | Cost |
|---------|----------|--------------|-----------|------|
| **PyTorch** | Development, local testing, single GPU | Optional (faster with GPU) | ~5 min | Free |
| **VLLM** | Production, high throughput, batch processing | Yes (NVIDIA) | ~10 min | Free |
| **MLX** | Apple Silicon (M1/M2/M3+), no GPU needed | No (optimized for Mac) | ~5 min | Free |
| **API** | Cloud-based, no GPU setup, pay-per-use | No | ~2 min | $0.01-0.10/request |

### Backend Decision Tree

```
Do you have a Mac with Apple Silicon (M1/M2/M3+)?
├─ YES → Use MLX backend
└─ NO
   ├─ Do you need to process 100+ documents quickly?
   │  ├─ YES → Use VLLM backend
   │  └─ NO
   │     ├─ Do you have a GPU (NVIDIA, AMD, or other)?
   │     │  ├─ YES → Use PyTorch backend (recommended)
   │     │  └─ NO
   │     │     └─ Use API backend
   └─ Or just prototyping/learning?
      └─ Use PyTorch backend
```

## Installation Instructions

### Option 1: PyTorch (Recommended for Most Users)

Best for local development and GPU inference on NVIDIA/AMD GPUs.

```bash
# Install with pip
pip install omnidocs[pytorch]

# Or with uv
uv pip install omnidocs[pytorch]

# Verify installation
python -c "from omnidocs import Document; print('PyTorch backend ready!')"
```

**What's Installed:**
- `torch` - Deep learning framework
- `transformers` - HuggingFace model support
- `accelerate` - Multi-GPU support
- All core OmniDocs dependencies

**Requirements:**
- NVIDIA GPU: CUDA 12.1+ (install from [nvidia.com](https://developer.nvidia.com/cuda-downloads))
- AMD GPU: ROCm support (install from [rocmdocs.amd.com](https://rocmdocs.amd.com))
- Apple Silicon: Works out of the box (will use CPU, slower)

**GPU Check:**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Option 2: VLLM (High-Throughput Production)

Best for processing many documents or running as a service.

```bash
# Install with pip
pip install omnidocs[vllm]

# Or with uv
uv pip install omnidocs[vllm]

# Verify installation
python -c "from omnidocs.tasks.text_extraction import QwenTextExtractor; print('VLLM backend ready!')"
```

**What's Installed:**
- `vllm` - High-throughput inference engine
- All PyTorch dependencies (automatically included)

**Requirements:**
- NVIDIA GPU with 24+ GB VRAM (A40, A100, H100, RTX 4090, etc.)
- CUDA 12.1+ installed

**Why VLLM?**
- 10x faster throughput than PyTorch for batch processing
- Optimized tensor parallelism across multiple GPUs
- Built-in request batching and memory optimization

### Option 3: MLX (Apple Silicon)

Best for M1/M2/M3 Macs without external GPU dependencies.

```bash
# Install with pip
pip install omnidocs[mlx]

# Or with uv
uv pip install omnidocs[mlx]

# Verify installation
python -c "from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig; print('MLX backend ready!')"
```

**What's Installed:**
- `mlx` - Apple Silicon machine learning framework
- `mlx-vlm` - Vision-language model support for MLX

**Requirements:**
- Mac with Apple Silicon (M1, M1 Pro, M1 Max, M2, M3, etc.)
- macOS 12+
- 8 GB+ unified memory (RAM)

**Why MLX?**
- Native Apple Silicon optimization (2-3x faster than generic Python)
- No need for NVIDIA GPU drivers or CUDA
- Automatic unified memory management

**Check Your Mac:**
```bash
# See your chip type
sysctl -a | grep machdep.cpu.brand_string

# Output: Apple M3 Max → You can use MLX!
```

### Option 4: API (Cloud-Based, No Setup)

Best for quick testing without GPU, or outsourcing inference costs.

```bash
# Install with pip
pip install omnidocs[api]

# Set up API key
export OPENAI_API_KEY="sk-..."  # Your API key

# Or use in code
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Verify installation
python -c "from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig; print('API backend ready!')"
```

**What's Installed:**
- `openai` - API client library

**Requirements:**
- API key (from [OpenAI](https://platform.openai.com) or compatible provider)
- Internet connection
- ~$0.01-0.10 per document

**Cost Estimation:**
- Small documents (< 1 MB): $0.01-0.05
- Medium documents (1-10 MB): $0.05-0.15
- Large documents (> 10 MB): $0.15-0.50

## Advanced Installations

### Install Everything (All Backends)

```bash
pip install omnidocs[all]
```

This installs PyTorch, VLLM, MLX, and API backends. Useful for organizations with mixed infrastructure.

### Development Installation

For contributing to OmniDocs:

```bash
# Clone repository
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs

# Install with development tools
uv sync --group dev
```

### Using UV (Recommended for Performance)

[UV](https://github.com/astral-sh/uv) is 10-100x faster than pip for dependency resolution:

```bash
# Install uv (one time)
pip install uv

# Use uv instead of pip
uv pip install omnidocs[pytorch]

# Or create a virtual environment
uv venv
source .venv/bin/activate
uv pip install omnidocs[pytorch]
```

## Verification Steps

### Verify Core Installation

```python
# Test basic import
from omnidocs import Document
print("OmniDocs installed successfully!")

# Check version
import omnidocs
print(f"Version: {omnidocs.__version__}")
```

### Verify Backend-Specific Installation

**PyTorch:**
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
print("PyTorch backend available!")
```

**VLLM:**
```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
print("VLLM backend available!")
```

**MLX:**
```python
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig
print("MLX backend available!")
```

**API:**
```python
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig
print("API backend available!")
```

### Full Test

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from PIL import Image

# Create a test image
img = Image.new('RGB', (400, 300), color='white')

# Initialize extractor
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",  # Small model for testing
        device="cpu",  # Test on CPU first
    )
)

# Extract text
result = extractor.extract(img, output_format="markdown")
print(f"Success! Extracted {len(result.content)} characters")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Problem:** PyTorch backend not installed.

**Solution:**
```bash
pip install omnidocs[pytorch]
```

### "CUDA out of memory"

**Problem:** GPU doesn't have enough memory for the model.

**Solutions:**
1. Use a smaller model:
   ```python
   config = QwenTextPyTorchConfig(
       model="Qwen/Qwen3-VL-2B-Instruct"  # 2B instead of 8B
   )
   ```

2. Reduce batch size (if processing multiple images)

3. Clear GPU memory:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### "No GPU detected"

**Problem:** PyTorch installed but GPU not recognized.

**Solution:**
```bash
# Check CUDA installation
nvidia-smi  # For NVIDIA GPUs

# Reinstall torch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "ImportError: urllib3 not available"

**Problem:** Dependency conflict with requests library.

**Solution:**
```bash
pip install --upgrade requests urllib3
```

### "VLLM requires CUDA"

**Problem:** VLLM installed on non-NVIDIA system.

**Solution:** Use PyTorch backend instead:
```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
config = QwenTextPyTorchConfig(...)
```

## Next Steps

After installation, proceed to:

1. **[Quickstart](quickstart.md)** - Get running in 5 minutes
2. **[First Document](first-document.md)** - Load and process your first PDF
3. **[Choosing Backends](choosing-backends.md)** - Understand backend tradeoffs

## Getting Help

- **Documentation**: Visit [omnidocs.readthedocs.io](https://omnidocs.readthedocs.io)
- **GitHub Issues**: Report bugs at [github.com/adithya-s-k/OmniDocs/issues](https://github.com/adithya-s-k/OmniDocs/issues)
- **Discussions**: Ask questions at [github.com/adithya-s-k/OmniDocs/discussions](https://github.com/adithya-s-k/OmniDocs/discussions)

## Summary

| Use Case | Install Command | Backend | Approx Time |
|----------|-----------------|---------|------------|
| Quick prototyping | `pip install omnidocs[pytorch]` | PyTorch | 5 min |
| Production scale | `pip install omnidocs[vllm]` | VLLM | 10 min |
| Mac development | `pip install omnidocs[mlx]` | MLX | 5 min |
| No GPU available | `pip install omnidocs[api]` | API | 2 min |
| All options | `pip install omnidocs[all]` | All | 15 min |

Happy documenting!
