# Qwen3-VL Text Extraction

## Model Overview

Qwen3-VL is an advanced Vision-Language Model optimized for document understanding and text extraction. It excels at producing high-quality markdown and HTML output while maintaining document layout and semantic structure.

**Model Family**: Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B, Qwen3-VL-32B
**Repository**: [Qwen/Qwen3-VL](https://huggingface.co/Qwen)
**Recommended Variant**: Qwen3-VL-8B-Instruct (best balance of quality and speed)

### Key Capabilities

- **Multi-format Output**: Markdown, HTML, or custom formats
- **Layout-Aware**: Preserves document structure and semantic relationships
- **Multilingual**: Supports 25+ languages with native quality
- **Document Types**: PDFs, academic papers, technical docs, web pages, presentations
- **Scale Support**: Handles documents from single-page images to 16k+ token outputs
- **Custom Prompts**: Flexible prompt engineering for specialized extraction tasks

### Limitations

- Requires GPU for inference (2B variant: 4GB VRAM, 8B: 16GB, 32B: 40GB+)
- Slower than single-task models (100-300 tokens/sec depending on backend)
- Can struggle with highly stylized or unusual layouts
- No inherent language detection (specify language in config if needed)

---

## Supported Backends

Qwen3-VL supports **4 inference backends**, allowing you to choose the right deployment method:

| Backend | Use Case | Performance | Setup |
|---------|----------|-------------|-------|
| **PyTorch** | Local GPU inference | 50-150 tokens/sec | Easy, single GPU |
| **VLLM** | High-throughput batching | 200-400 tokens/sec | Requires GPU cluster |
| **MLX** | Apple Silicon (native) | 20-50 tokens/sec | macOS M1/M2/M3+ only |
| **API** | Hosted inference | Variable | Cloud provider |

---

## Installation & Configuration

### Basic Installation

```bash
# Install with PyTorch backend (most common)
pip install omnidocs[pytorch]

# Or install with VLLM for high throughput
pip install omnidocs[vllm]

# Or install with all backends
pip install omnidocs[all]
```

### PyTorch Backend Configuration

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    use_flash_attention=False,  # Set to True if flash-attn installed
    max_new_tokens=8192,
    temperature=0.1,
)

extractor = QwenTextExtractor(backend=config)
```

**PyTorch Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "Qwen/Qwen3-VL-8B-Instruct" | HuggingFace model ID |
| `device` | str | "cuda" | Device: "cuda", "mps", "cpu" |
| `torch_dtype` | str | "auto" | Data type: "float16", "bfloat16", "float32", "auto" |
| `device_map` | str | "auto" | Model parallelism: "auto", "balanced", "sequential", None |
| `trust_remote_code` | bool | True | Allow custom model code from HuggingFace |
| `use_flash_attention` | bool | False | Use Flash Attention 2 (faster, requires flash-attn) |
| `max_new_tokens` | int | 8192 | Max tokens to generate (256-32768) |
| `temperature` | float | 0.1 | Sampling temperature (0.0-2.0, lower = deterministic) |

### VLLM Backend Configuration

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,  # Use 2+ for large models
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

extractor = QwenTextExtractor(backend=config)
```

**VLLM Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | HuggingFace model ID |
| `tensor_parallel_size` | int | 1 | Number of GPUs for parallelism |
| `gpu_memory_utilization` | float | 0.9 | GPU memory usage (0.1-1.0) |
| `max_model_len` | int | None | Max context length in tokens |

### MLX Backend Configuration (Apple Silicon)

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

config = QwenTextMLXConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    quantization="4bit",  # or "8bit", "none"
    max_tokens=8192,
)

extractor = QwenTextExtractor(backend=config)
```

### API Backend Configuration

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

config = QwenTextAPIConfig(
    model="qwen3-vl-8b",
    api_key="your-api-key",
    base_url="https://api.provider.com/v1",
    rate_limit=10,  # Requests per second
)

extractor = QwenTextExtractor(backend=config)
```

---

## Usage Examples

### Basic Text Extraction (Markdown)

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from omnidocs import Document
from PIL import Image

# Initialize extractor
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
)
extractor = QwenTextExtractor(backend=config)

# Load document
image = Image.open("document.png")

# Extract text in markdown
result = extractor.extract(
    image,
    output_format="markdown",
)

print(result.content)  # Clean markdown
print(result.word_count)  # Approximate word count
```

### Multi-Format Extraction

```python
# HTML output (preserves more layout semantics)
result_html = extractor.extract(
    image,
    output_format="html",
)

# Custom prompt for specialized extraction
custom_prompt = """Extract all text as JSON with structure:
{
    "title": "...",
    "sections": [{"heading": "...", "content": "..."}],
    "tables": [...]
}
"""

result_custom = extractor.extract(
    image,
    output_format="markdown",
    custom_prompt=custom_prompt,
)
```

### Batch Processing with VLLM

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
from PIL import Image
import time

# Initialize with VLLM for high throughput
config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.8,
)
extractor = QwenTextExtractor(backend=config)

# Load multiple documents
images = [
    Image.open(f"doc_{i}.png") for i in range(10)
]

# Process with streaming
results = []
start = time.time()

for i, image in enumerate(images):
    result = extractor.extract(image, output_format="markdown")
    results.append(result)
    elapsed = time.time() - start
    throughput = (i + 1) / elapsed * 1000  # chars/sec
    print(f"[{i+1}/10] {result.content_length} chars - {throughput:.0f} chars/sec")

print(f"\nTotal time: {time.time() - start:.1f}s")
print(f"Avg length: {sum(r.content_length for r in results) / len(results):.0f} chars")
```

### Layout-Aware Extraction

```python
# Include layout information
result = extractor.extract(
    image,
    output_format="markdown",
    include_layout=True,
)

# Access raw output with bounding boxes
print(result.raw_output)  # Contains bbox annotations
```

### API-Based Extraction (Cloud)

```python
import os
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig
from PIL import Image

# Configure API backend
config = QwenTextAPIConfig(
    model="qwen3-vl-8b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://api.together.xyz/v1",
    rate_limit=5,
)

extractor = QwenTextExtractor(backend=config)

# Extract from image
image = Image.open("document.png")
result = extractor.extract(
    image,
    output_format="markdown",
)

print(result.content)
```

---

## Performance Characteristics

### Memory Requirements by Variant

| Model | Min VRAM | Optimal VRAM | Batch Size (VLLM) |
|-------|----------|--------------|-------------------|
| Qwen3-VL-2B | 4 GB | 8 GB | 8-16 |
| Qwen3-VL-4B | 8 GB | 12 GB | 4-8 |
| Qwen3-VL-8B | 16 GB | 24 GB | 2-4 |
| Qwen3-VL-32B | 40 GB | 80 GB | 1 |

### Inference Speed (Single Document)

| Backend | Model | Speed | Device |
|---------|-------|-------|--------|
| PyTorch | 8B | 50-100 tok/s | Single A10 GPU |
| VLLM | 8B | 200-300 tok/s | 2x A10 GPU (tensor parallel) |
| MLX | 8B-quantized | 20-40 tok/s | M3 Max (48GB) |
| API | 8B | Variable | Cloud (depends on provider) |

### Typical Output Sizes

| Document Type | Tokens | Characters |
|---------------|--------|-----------|
| Single-page document | 500-2000 | 3-12 KB |
| Academic paper page | 1000-4000 | 6-24 KB |
| Multi-page scanned doc | 2000-8000 | 12-48 KB |

---

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# 1. Reduce max_new_tokens
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    max_new_tokens=4096,  # Reduced from 8192
)

# 2. Use smaller model variant
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-4B-Instruct",  # Smaller variant
)

# 3. Enable quantization
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,  # Requires bitsandbytes
)

# 4. Use CPU
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cpu",  # Slower but works with limited VRAM
)
```

### Slow Inference

**Symptom**: Processing takes 30+ seconds per document

**Solutions**:

```python
# 1. Enable Flash Attention (requires flash-attn package)
config = QwenTextPyTorchConfig(
    use_flash_attention=True,
)

# 2. Use VLLM for batching
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,
)

# 3. Use smaller model
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-4B-Instruct",  # 2x faster
)

# 4. Reduce image size
from PIL import Image
image = Image.open("document.png")
image.thumbnail((1024, 1024))  # Resize to 1024x1024 max
```

### Poor Quality Output

**Symptom**: Garbled or incomplete text extraction

**Solutions**:

```python
# 1. Lower temperature for more deterministic output
config = QwenTextPyTorchConfig(
    temperature=0.01,  # Very low for consistency
)

# 2. Use larger model variant
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-32B-Instruct",  # Better quality
)

# 3. Pre-process image (enhance contrast, de-skew)
from PIL import ImageEnhance
image = Image.open("document.png")
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)  # Increase contrast

# 4. Custom prompt for better guidance
custom_prompt = """Extract all text exactly as it appears.
Preserve formatting, structure, and special characters."""
result = extractor.extract(image, custom_prompt=custom_prompt)
```

### API Rate Limiting

**Symptom**: `429 Too Many Requests` errors

**Solutions**:

```python
# Reduce rate limit
config = QwenTextAPIConfig(
    model="qwen3-vl-8b",
    api_key="...",
    rate_limit=2,  # Reduced from 10
)

# Implement retry logic
import time
max_retries = 3
for attempt in range(max_retries):
    try:
        result = extractor.extract(image)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise
```

### Model Download Issues

**Symptom**: `ConnectionError` or timeout during model loading

**Solutions**:

```python
# Set HuggingFace cache directory
import os
os.environ["HF_HOME"] = "/path/to/cache"

# Pre-download model
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-8B-Instruct")

# Use local model path
config = QwenTextPyTorchConfig(
    model="/local/path/to/model",
)
```

---

## Model Selection Guide

### When to Use Qwen3-VL

**Best for**:
- High-quality document extraction (academic papers, technical docs)
- Multilingual documents
- Complex layouts with mixed content types
- Production systems needing consistent quality

**Not ideal for**:
- Real-time processing (see: Nanonuts OCR for speed)
- Handwritten documents (see: Surya OCR)
- Fixed-label layout detection (see: DocLayout-YOLO)

### Qwen vs DotsOCR Comparison

| Feature | Qwen3-VL | DotsOCR |
|---------|----------|---------|
| Output Quality | Excellent | Very Good |
| Layout Info | Basic | Detailed (11 categories) |
| Speed | Medium | Fast |
| Memory | High | Medium |
| Multilingual | Yes (25+ langs) | Limited |
| Model Size Options | 2B-32B | Single |

**Choose Qwen3-VL if**: You need high-quality text and multilingual support
**Choose DotsOCR if**: You need detailed layout information with good performance

---

## API Reference

### QwenTextExtractor.extract()

```python
def extract(
    image: Union[Image.Image, np.ndarray, str, Path],
    output_format: str = "markdown",
    include_layout: bool = False,
    custom_prompt: Optional[str] = None,
) -> TextOutput:
    """
    Extract text from image using Qwen3-VL.

    Args:
        image: Input image (PIL Image, numpy array, or path)
        output_format: "markdown" or "html"
        include_layout: Include layout information in raw output
        custom_prompt: Override default extraction prompt

    Returns:
        TextOutput with extracted content
    """
```

### TextOutput Properties

```python
result = extractor.extract(image)

# Access extracted content
print(result.content)        # Formatted text (markdown/html)
print(result.format)         # Output format
print(result.plain_text)     # Plain text without formatting
print(result.content_length) # Character count
print(result.word_count)     # Approximate word count
print(result.image_width)    # Source image width
print(result.image_height)   # Source image height
print(result.model_name)     # Model used
print(result.raw_output)     # Raw model output (with artifacts)
```

---

## Advanced Configuration

### Device Map Strategies

```python
# Auto device mapping (recommended)
device_map = "auto"

# Balanced distribution across GPUs
device_map = "balanced"

# Sequential loading (one GPU at a time)
device_map = "sequential"

# Manual: First layer on GPU0, rest on CPU
device_map = {
    "model.layers.0": 0,
    "model.layers.1-31": "cpu",
}
```

### Data Type Selection

```python
# float32: Full precision (slower, more VRAM)
torch_dtype = "float32"

# float16: Half precision (faster, less VRAM, less accurate)
torch_dtype = "float16"

# bfloat16: Brain float (recommended for stability)
torch_dtype = "bfloat16"

# auto: Let model choose based on hardware
torch_dtype = "auto"
```

---

## See Also

- [Qwen HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [DotsOCR Documentation](./dotsocr.md) - For layout-aware extraction
- [Qwen Layout Detection](../layout-analysis/qwen-layout.md) - For layout analysis
- [Comparison Guide](./comparison.md) - Model selection matrix
