# Choosing the Right Backend

OmniDocs supports 4 inference backends. This guide helps you choose the right one for your use case.

## Quick Decision Tree

```
START HERE

Do you have a Mac with Apple Silicon (M1/M2/M3+)?
├─ YES → MLX backend (Best choice)
└─ NO
   ├─ Do you need to process 100+ documents per day?
   │  ├─ YES → VLLM backend (Fastest, GPU required)
   │  └─ NO
   │     ├─ Do you have a GPU (NVIDIA/AMD)?
   │     │  ├─ YES → PyTorch backend (Recommended, free)
   │     │  └─ NO
   │     │     └─ API backend (Easy, no setup)
```

## Backend Comparison

| Feature | PyTorch | VLLM | MLX | API |
|---------|---------|------|-----|-----|
| **Setup Time** | 5 min | 10 min | 5 min | 2 min |
| **Speed** | 1x (baseline) | 10x | 2x | 0.5x* |
| **Cost** | Free | Free | Free | $0.01-0.10/doc |
| **GPU Required** | Optional | Yes (NVIDIA) | No | No |
| **Batch Processing** | Good | Excellent | Good | Fair |
| **Model Size** | 2B-8B | 2B-32B | 2B-7B | 2B-8B |
| **Memory** | 4-16 GB | 24-80 GB | 8-16 GB | Minimal |
| **Best For** | Development | Production scale | Mac users | Quick testing |

*API speed depends on network latency

## Detailed Backend Profiles

### PyTorch (Recommended for Most Users)

**Best for:** Local development, prototyping, single GPU inference.

**Installation:**
```bash
pip install omnidocs[pytorch]
```

**Quick Setup:**
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda",  # or "cpu" or "mps" (Mac)
        torch_dtype="bfloat16",
    )
)

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

**Strengths:**
- Standard PyTorch/HuggingFace ecosystem
- Easy to install and use
- Excellent documentation and community support
- Works on NVIDIA, AMD, and Apple Silicon
- Good for experimentation and debugging

**Limitations:**
- Slower than VLLM for batch processing (1 image at a time)
- Requires GPU for fast inference
- Higher memory usage per inference

**Performance Expectations:**
- NVIDIA A100 (40 GB): ~0.5-1 sec per page
- NVIDIA RTX 4090 (24 GB): ~1-2 sec per page
- Mac M3 Max: ~3-5 sec per page (CPU optimization)
- CPU-only: ~20-30 sec per page (very slow)

**Configuration Options:**

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(
    # Model selection
    model="Qwen/Qwen3-VL-2B-Instruct",  # Small, fast (default)
    # model="Qwen/Qwen3-VL-8B-Instruct",  # Large, more accurate
    # model="Qwen/Qwen3-VL-32B-Instruct",  # Huge, slowest

    # Hardware
    device="cuda",              # NVIDIA/AMD GPU (recommended)
    # device="mps",             # Apple Silicon (native)
    # device="cpu",             # CPU-only (very slow)

    # Data precision
    torch_dtype="bfloat16",     # Fast, good quality (recommended)
    # torch_dtype="float16",    # Faster, slightly lower quality
    # torch_dtype="float32",    # Slowest, highest quality
    # torch_dtype="auto",       # Let model decide

    # Advanced
    device_map="auto",          # Automatic memory optimization
    trust_remote_code=True,     # Allow custom model code
    use_flash_attention=False,  # Flash attention (experimental)
)
```

**When to Use PyTorch:**
- Learning OmniDocs (simplest setup)
- Processing documents one at a time
- Experimenting with different models
- Don't need maximum throughput
- Prefer free, no external API dependencies

**When NOT to Use:**
- Need to process 100+ documents per hour
- Working with very large documents (>100 pages)
- Must minimize GPU memory usage

---

### VLLM (Production Throughput)

**Best for:** Production systems, batch processing, 100+ documents per day.

**Installation:**
```bash
pip install omnidocs[vllm]
```

**Quick Setup:**
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

extractor = QwenTextExtractor(
    backend=QwenTextVLLMConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=1,      # Devices to split model across
        gpu_memory_utilization=0.9,  # Optimize for throughput
    )
)

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

**Strengths:**
- 10x faster throughput than PyTorch
- Optimized tensor parallelism (split across multiple GPUs)
- Intelligent batching and memory management
- Built for production inference
- Lower latency per request

**Limitations:**
- NVIDIA GPU required (no CPU or Mac support)
- Requires 24+ GB VRAM (A40, A100, RTX 4090, H100, etc.)
- Steeper learning curve
- Requires separate VLLM server setup

**Performance Expectations:**
- NVIDIA H100 (80 GB): ~100-150 images/hour
- NVIDIA A100 (40 GB): ~50-80 images/hour
- NVIDIA RTX 4090 (24 GB): ~30-50 images/hour
- Tensor parallel (2x A100): ~150+ images/hour

**Configuration Options:**

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

config = QwenTextVLLMConfig(
    # Model selection
    model="Qwen/Qwen3-VL-8B-Instruct",

    # Parallelization
    tensor_parallel_size=1,         # Single GPU (default)
    # tensor_parallel_size=2,       # Split across 2 GPUs
    # tensor_parallel_size=4,       # Split across 4 GPUs

    # Memory optimization
    gpu_memory_utilization=0.9,     # 90% utilization (aggressive)
    # gpu_memory_utilization=0.6,   # 60% utilization (conservative)

    # Performance tuning
    max_model_len=8192,             # Max tokens (truncate if needed)
    dtype="auto",                   # Auto-select dtype
)
```

**Multi-GPU Tensor Parallelism:**

```python
# Distribute model across 2 GPUs for massive throughput
config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-32B-Instruct",
    tensor_parallel_size=2,  # Split across GPUs 0, 1
    gpu_memory_utilization=0.95,
)

# Model is automatically split:
# GPU 0: 50% of model weights
# GPU 1: 50% of model weights
# Inference uses both GPUs in parallel
```

**When to Use VLLM:**
- Processing 100+ documents per day
- Need consistent inference throughput
- Have multiple GPUs available
- Building production inference service
- Cost per inference matters

**When NOT to Use:**
- Learning OmniDocs (use PyTorch first)
- Don't have NVIDIA GPU
- Need quick one-off processing
- Working on Mac

---

### MLX (Apple Silicon)

**Best for:** M1/M2/M3 Mac users, no external dependencies.

**Installation:**
```bash
pip install omnidocs[mlx]
```

**Quick Setup:**
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

extractor = QwenTextExtractor(
    backend=QwenTextMLXConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        # Configuration options shown below
    )
)

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

**Strengths:**
- Native optimization for Apple Silicon
- No NVIDIA GPU required
- Simple setup, minimal dependencies
- Good performance on Mac hardware
- Efficient memory management (unified memory)

**Limitations:**
- Mac only (M1/M2/M3 required)
- Smaller model selection than PyTorch
- Slightly slower than VLLM
- Less community support than PyTorch

**Performance Expectations:**
- M3 Max (128 GB unified): ~2-3 sec per page
- M2 Pro (16 GB unified): ~3-5 sec per page
- M1 (8 GB unified): ~5-10 sec per page

**Configuration Options:**

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

config = QwenTextMLXConfig(
    # Model selection (MLX-optimized models)
    model="Qwen/Qwen3-VL-2B-Instruct",

    # Precision (auto-optimal for Apple Silicon)
    dtype="auto",  # MLX picks the best dtype

    # Optimization
    quantization="4bit",  # 4-bit quantization for speed
    # quantization=None,   # Full precision (slower but more accurate)

    # Max tokens
    max_model_len=8192,
)
```

**When to Use MLX:**
- Developing on Mac with Apple Silicon
- Want native performance optimization
- Prefer not to deal with NVIDIA drivers
- Building Mac-native applications
- Prefer simple, no-hassle setup

**When NOT to Use:**
- Need VLLM production performance
- Don't have Apple Silicon Mac
- Need many different model choices
- Working in cloud environment

---

### API Backend (Cloud-Based)

**Best for:** Quick testing, no GPU setup, cloud-first applications.

**Installation:**
```bash
pip install omnidocs[api]
export OPENAI_API_KEY="sk-..."
```

**Quick Setup:**
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

extractor = QwenTextExtractor(
    backend=QwenTextAPIConfig(
        model="gpt-4-vision",
        api_key="sk-...",  # Or use OPENAI_API_KEY env var
    )
)

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

**Strengths:**
- Zero GPU setup required
- Works anywhere with internet
- No local hardware needed
- Easy to switch models
- Excellent for prototyping

**Limitations:**
- Costs money ($0.01-0.10 per document)
- Network latency
- Depends on API availability
- Limited model selection
- Privacy concerns (data sent to external service)

**Pricing Estimates:**
- Small documents (< 1 MB): $0.01-0.05
- Medium documents (1-10 MB): $0.05-0.15
- Large documents (> 10 MB): $0.15-0.50

**For 1000 documents:**
- PyTorch: $0 (after initial setup)
- VLLM: $0 (after server setup)
- MLX: $0 (Mac only)
- API: $50-100 (at $0.05-0.10 per doc)

**Configuration Options:**

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

config = QwenTextAPIConfig(
    # Model selection
    model="gpt-4-vision",           # Most capable
    # model="gpt-4-turbo-vision",    # Faster
    # model="gpt-3.5-vision",        # Cheapest

    # Authentication
    api_key="sk-...",               # Or use env var
    base_url="https://api.openai.com/v1",  # Custom endpoint

    # Rate limiting
    rate_limit=10,                  # Requests per minute
)
```

**When to Use API:**
- Prototyping quickly
- Don't have GPU
- Occasional document processing (< 50/month)
- Data privacy not a concern
- Building SaaS with usage-based pricing

**When NOT to Use:**
- High volume (100+ documents/day) - too expensive
- Sensitive data (privacy concerns)
- No internet access
- Need offline capability
- Want total cost predictability

---

## Performance Comparison

### Throughput (Documents per Hour)

```
PyTorch (2B model, RTX 4090): 30-50 docs/hour
PyTorch (8B model, A100):     40-60 docs/hour
VLLM (8B model, RTX 4090):    300-500 docs/hour
VLLM (8B model, A100):        600-1000 docs/hour
MLX (2B model, M3 Max):       60-80 docs/hour
API (via OpenAI):             10-20 docs/hour (network limited)
```

### Quality (Accuracy)

Approximately equal across all backends if using same model:

```
2B models:     85-90% accuracy (fast, good for most documents)
8B models:     92-96% accuracy (slower, better for complex docs)
32B models:    96-98% accuracy (slowest, for critical documents)
```

Quality depends more on **model size** than **backend choice**.

### Cost Analysis

**Scenario 1: Process 100 documents once**
- PyTorch: $0 (one-time setup ~$100 GPU)
- VLLM: $0 (one-time setup ~$500-5000 GPU)
- MLX: $0 (Mac-only, built-in)
- API: $5-10

**Scenario 2: Process 10,000 documents**
- PyTorch: $0 (amortized: $0.01 per doc for GPU)
- VLLM: $0 (amortized: $0.05-0.20 per doc for GPU)
- MLX: $0 (Mac-only)
- API: $500-1000

**Scenario 3: Process 100,000+ documents**
- PyTorch: $0 amortized, but need to manage GPU scaling
- VLLM: $0 amortized, designed for this scale
- MLX: Not suitable (Mac only)
- API: $5,000-10,000 (prohibitive)

## Migration Between Backends

OmniDocs is designed for easy backend switching - code changes minimally:

### Switch from PyTorch to VLLM

```python
# PyTorch version
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
backend = QwenTextPyTorchConfig(device="cuda")

# VLLM version (same interface, different config)
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
backend = QwenTextVLLMConfig(tensor_parallel_size=1)

# Rest of code stays the same!
```

### Switch from VLLM to PyTorch

```python
# VLLM version
backend = QwenTextVLLMConfig(tensor_parallel_size=2)

# PyTorch version
backend = QwenTextPyTorchConfig(device="cuda", torch_dtype="bfloat16")

# Same .extract() interface works for both
```

## Troubleshooting Backend Selection

### "I get OutOfMemory errors"

**Option 1:** Use smaller model
```python
# Instead of 8B
backend = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-2B-Instruct"
)
```

**Option 2:** Use MLX (better memory management)
```bash
pip install omnidocs[mlx]
```

**Option 3:** Use API (no local GPU memory)
```bash
pip install omnidocs[api]
```

### "My GPU is slow"

**Option 1:** Use VLLM for better throughput
```bash
pip install omnidocs[vllm]
```

**Option 2:** Check CUDA version
```bash
nvcc --version  # Should be 12.1+
```

### "I only have a Mac"

**Use MLX (best for Mac):**
```bash
pip install omnidocs[mlx]
```

**Or use API (if MLX not suitable):**
```bash
pip install omnidocs[api]
```

### "I don't have a GPU"

**Option 1:** Use API backend
```bash
pip install omnidocs[api]
```

**Option 2:** Use PyTorch on CPU (very slow)
```python
backend = QwenTextPyTorchConfig(device="cpu")
```

## Decision Matrix by Use Case

| Use Case | Recommended | Alternative | Avoid |
|----------|-------------|-------------|-------|
| Learning OmniDocs | PyTorch | API | - |
| Development/Prototyping | PyTorch | MLX (Mac) | VLLM |
| Mac development | MLX | PyTorch | - |
| 100+ docs/day | VLLM | PyTorch | API (expensive) |
| Production service | VLLM | - | PyTorch |
| No GPU setup | API | - | PyTorch, VLLM |
| Cost sensitive | PyTorch | VLLM | API |
| Quick one-off | API | PyTorch | VLLM |
| High accuracy critical | 8B+ model | - | 2B model |
| Batch processing | VLLM | PyTorch | API |

## Summary and Recommendations

**For Most Users:** Start with **PyTorch** backend
- Simple setup
- Free (after initial GPU cost)
- Flexible
- Excellent documentation

**For Scale:** Upgrade to **VLLM** backend
- When processing 100+ documents per day
- When cost of GPU amortization matters
- When throughput is critical

**For Mac Users:** Use **MLX** backend
- Native optimization
- Simple setup
- Good performance

**For Quick Testing:** Use **API** backend
- Minimal setup
- No GPU needed
- Good for prototyping
- Acceptable for low volume

Next: [Quickstart](quickstart.md) to start extracting text!
