# Batch Processing Guide

Process multiple documents efficiently at scale. This guide covers batch loading, processing patterns, memory optimization, progress tracking, and GPU deployment.

## Table of Contents

- [Batch Loading](#batch-loading)
- [Processing Patterns](#processing-patterns)
- [Memory Optimization](#memory-optimization)
- [Progress Tracking](#progress-tracking)
- [Error Handling](#error-handling)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

## Batch Loading

### Load from Directory

Load all images or PDFs from a directory.

```python
from pathlib import Path
from omnidocs import Document
from PIL import Image

# Find all image files
image_dir = Path("documents/images")
image_paths = sorted(
    list(image_dir.glob("*.png")) +
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg"))
)

print(f"Found {len(image_paths)} images")

# Load as PIL Images
images = [Image.open(p) for p in image_paths]

# Load PDFs
pdf_dir = Path("documents/pdfs")
pdf_paths = sorted(pdf_dir.glob("*.pdf"))

documents = [Document.from_pdf(p) for p in pdf_paths]
print(f"Found {len(documents)} PDFs with {sum(d.page_count for d in documents)} total pages")
```

### Lazy Loading for Large Batches

Don't load all images upfront - load as needed to save memory.

```python
from pathlib import Path
from PIL import Image

image_dir = Path("documents/")
image_paths = sorted(image_dir.glob("*.png"))

# Generator: loads images on-demand
def image_generator(paths):
    """Generator that yields images one at a time."""
    for path in paths:
        yield Image.open(path)

# Usage: iterate without loading all at once
for idx, image in enumerate(image_generator(image_paths)):
    print(f"Processing image {idx+1}/{len(image_paths)}")
    # Process one image, then load next
    # image is garbage collected automatically
```

### Load with Metadata

Track source information for each batch item.

```python
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BatchItem:
    """Container for batch item with metadata."""
    path: Path
    image: Image.Image
    metadata: Dict[str, Any]

# Load with metadata
items = []
for image_path in image_paths:
    image = Image.open(image_path)
    item = BatchItem(
        path=image_path,
        image=image,
        metadata={
            "filename": image_path.name,
            "size_bytes": image_path.stat().st_size,
            "dimensions": image.size,
            "format": image.format,
        }
    )
    items.append(item)

print(f"Loaded {len(items)} items with metadata")
```

## Processing Patterns

### Pattern 1: Simple Loop

Process items sequentially (smallest memory footprint).

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from pathlib import Path
from PIL import Image
import time

# Initialize extractor once
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

# Load image paths
images = sorted(Path("images/").glob("*.png"))

# Process sequentially
results = []
start = time.time()

for idx, image_path in enumerate(images):
    image = Image.open(image_path)
    result = extractor.extract(image, output_format="markdown")
    results.append({
        "path": str(image_path),
        "content_length": result.content_length,
        "word_count": result.word_count,
    })

elapsed = time.time() - start
print(f"Processed {len(results)} images in {elapsed:.1f}s")
print(f"Average: {elapsed/len(images):.2f}s per image")
```

### Pattern 2: Batched Processing

Group images into batches (more efficient for VLLM).

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
from pathlib import Path
from PIL import Image

# Use VLLM for better batch efficiency
config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    max_tokens=4096,
)
extractor = QwenTextExtractor(backend=config)

# Load images
images = [Image.open(p) for p in sorted(Path("images/").glob("*.png"))]

# Process in batches
batch_size = 4
results = []

for batch_idx in range(0, len(images), batch_size):
    batch = images[batch_idx:batch_idx + batch_size]
    print(f"Processing batch {batch_idx//batch_size + 1}")

    for image in batch:
        result = extractor.extract(image, output_format="markdown")
        results.append(result)

print(f"Processed {len(results)} images")
```

### Pattern 3: PDF with Multiple Pages

Process all pages of multiple PDFs.

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from pathlib import Path

# Initialize
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

# Load PDFs
pdf_files = sorted(Path("pdfs/").glob("*.pdf"))

# Process all pages
all_results = []

for pdf_path in pdf_files:
    print(f"Processing {pdf_path.name}")
    doc = Document.from_pdf(pdf_path)

    for page_idx in range(doc.page_count):
        page_image = doc.get_page(page_idx)
        result = extractor.extract(page_image, output_format="markdown")

        all_results.append({
            "pdf": pdf_path.name,
            "page": page_idx + 1,
            "word_count": result.word_count,
            "content": result.content,
        })

print(f"Processed {sum(d['page_count'] for d in documents)} pages total")
```

### Pattern 4: Parallel Processing (Per-Document)

Use multiprocessing for CPU-bound preprocessing.

```python
from multiprocessing import Pool
from PIL import Image
from pathlib import Path

def preprocess_image(image_path):
    """Preprocess a single image."""
    image = Image.open(image_path)

    # Resize if needed
    if image.width < 1024:
        image = image.resize((image.width * 2, image.height * 2))

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image_path, image

# Parallel preprocessing
image_paths = sorted(Path("images/").glob("*.png"))

with Pool(4) as pool:  # 4 processes
    results = pool.map(preprocess_image, image_paths)

print(f"Preprocessed {len(results)} images")

# Then process with GPU (sequential, since we only have 1 GPU)
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

for path, image in results:
    result = extractor.extract(image, output_format="markdown")
    # Process...
```

## Memory Optimization

### Monitor GPU Memory

```python
import torch

print("GPU Memory:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.1f}GB")
print(f"  Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Clear cache between batches
torch.cuda.empty_cache()
print("Cache cleared")
```

### Optimize Model Configuration

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Memory-optimized configuration
config = QwenTextPyTorchConfig(
    device="cuda",
    torch_dtype="float16",  # Half precision (less memory)
    max_new_tokens=2048,  # Smaller context (less memory)
)
```

### Process in Streaming Fashion

Never keep all results in memory - stream to disk.

```python
import json
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from pathlib import Path
from PIL import Image

config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

images = sorted(Path("images/").glob("*.png"))

# Stream results to JSON Lines file
output_file = "results.jsonl"

with open(output_file, "w") as f:
    for image_path in images:
        image = Image.open(image_path)
        result = extractor.extract(image, output_format="markdown")

        # Write immediately (don't accumulate in memory)
        record = {
            "path": str(image_path),
            "content_length": result.content_length,
            "word_count": result.word_count,
        }
        f.write(json.dumps(record) + "\n")

# Results are on disk, not in memory
print(f"Streamed results to {output_file}")

# Read results later
results = []
with open(output_file) as f:
    for line in f:
        results.append(json.loads(line))
```

### Garbage Collection

Explicitly free memory between batches.

```python
import gc
import torch

for batch_idx, images in enumerate(batches):
    # Process batch
    for image in images:
        result = extractor.extract(image)

    # Free memory
    del images
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Batch {batch_idx + 1} complete, memory freed")
```

## Progress Tracking

### Simple Counter

```python
images = sorted(Path("images/").glob("*.png"))
total = len(images)

for idx, image_path in enumerate(images, 1):
    image = Image.open(image_path)
    result = extractor.extract(image)

    # Print progress
    print(f"[{idx}/{total}] {image_path.name}", end=" ")
    print(f"✓ {result.word_count} words")
```

**Output:**
```
[1/100] document_1.png ✓ 245 words
[2/100] document_2.png ✓ 312 words
[3/100] document_3.png ✓ 189 words
```

### Progress Bar with tqdm

```python
from tqdm import tqdm
from pathlib import Path
from PIL import Image

images = sorted(Path("images/").glob("*.png"))

for image_path in tqdm(images, desc="Processing"):
    image = Image.open(image_path)
    result = extractor.extract(image)
    # Process...
```

**Output:**
```
Processing: 45%|████▌     | 45/100 [5:23<6:32, 8.22s/it]
```

### Detailed Progress with ETA

```python
import time
from pathlib import Path
from PIL import Image

images = sorted(Path("images/").glob("*.png"))
total = len(images)

start_time = time.time()

for idx, image_path in enumerate(images, 1):
    image = Image.open(image_path)
    result = extractor.extract(image)

    # Calculate metrics
    elapsed = time.time() - start_time
    avg_time = elapsed / idx
    remaining = (total - idx) * avg_time
    remaining_mins = remaining / 60

    # Print progress
    percent = 100 * idx / total
    print(f"[{idx:3d}/{total}] {percent:5.1f}% "
          f"{image_path.name:20} "
          f"ETA: {remaining_mins:5.1f}min")
```

**Output:**
```
[  1/100]   1.0% document_1.png         ETA:  8.2min
[ 10/100]  10.0% document_10.png        ETA:  7.4min
[ 50/100]  50.0% document_50.png        ETA:  3.7min
[100/100] 100.0% document_100.png       ETA:  0.0min
```

### Save Progress Periodically

```python
import json
from pathlib import Path
from PIL import Image

images = sorted(Path("images/").glob("*.png"))
checkpoint_file = "progress.json"

# Load existing progress
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        completed = set(json.load(f).get("completed", []))
else:
    completed = set()

results = []

for image_path in images:
    if str(image_path) in completed:
        print(f"Skipping {image_path.name} (already processed)")
        continue

    image = Image.open(image_path)
    result = extractor.extract(image)
    results.append({
        "path": str(image_path),
        "word_count": result.word_count,
    })

    # Save progress periodically
    completed.add(str(image_path))
    if len(results) % 10 == 0:
        with open(checkpoint_file, "w") as f:
            json.dump({"completed": list(completed)}, f)
        print(f"Saved progress: {len(completed)}/{len(images)} completed")

# Final save
with open(checkpoint_file, "w") as f:
    json.dump({"completed": list(completed)}, f)
```

## Error Handling

### Graceful Degradation

```python
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

images = sorted(Path("images/").glob("*.png"))
results = []
errors = []

for image_path in images:
    try:
        image = Image.open(image_path)
        result = extractor.extract(image)
        results.append({"path": str(image_path), "success": True})

    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM on {image_path.name}")
        errors.append({"path": str(image_path), "error": "OOM"})
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error on {image_path.name}: {e}")
        errors.append({"path": str(image_path), "error": str(e)})

print(f"\nResults: {len(results)} succeeded, {len(errors)} failed")

if errors:
    print("\nFailed items:")
    for error in errors:
        print(f"  {error['path']}: {error['error']}")
```

### Retry on Error

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def extract_with_retry(extractor, image):
    """Extract with automatic retry on failure."""
    try:
        return extractor.extract(image)
    except Exception as e:
        logger.warning(f"Extraction failed: {e}, retrying...")
        raise

# Use in batch processing
for image_path in images:
    try:
        image = Image.open(image_path)
        result = extract_with_retry(extractor, image)
        results.append(result)
    except Exception as e:
        logger.error(f"Failed after retries: {image_path}: {e}")
```

## Performance Benchmarks

### Typical Performance

Processing a standard page (300 DPI, ~2000x3000px):

**PyTorch (Single GPU):**
- Model load: ~2-3 seconds (one-time)
- Per-page latency: ~2-3 seconds
- Throughput: ~1 page/second
- GPU Memory: ~16GB

**VLLM (Single GPU):**
- Model load: ~5-8 seconds (one-time)
- Per-page latency: ~2-3 seconds
- Throughput: ~1-2 pages/second (batched)
- GPU Memory: ~20GB

**Multi-GPU VLLM:**
- Model load: ~8-12 seconds
- Per-page latency: ~1-2 seconds
- Throughput: ~2-4 pages/second (batched)
- GPU Memory: ~10GB per GPU

### 100-Document Benchmark

Processing 100 pages (typical):

```python
import time

images = [...] # 100 images

# PyTorch
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

start = time.time()
for image in images:
    result = extractor.extract(image)
elapsed = time.time() - start

print(f"PyTorch: {elapsed:.1f}s ({elapsed/100:.2f}s per page)")
# Expected: ~3-4 minutes total

# VLLM
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

config = QwenTextVLLMConfig(
    tensor_parallel_size=1,
    max_tokens=4096,
)
extractor = QwenTextExtractor(backend=config)

start = time.time()
for image in images:
    result = extractor.extract(image)
elapsed = time.time() - start

print(f"VLLM: {elapsed:.1f}s ({elapsed/100:.2f}s per page)")
# Expected: ~2-3 minutes total
```

## Troubleshooting

### Out of Memory During Batch Processing

**Problem:** CUDA OOM after processing several documents.

**Solutions:**
1. Reduce batch size
2. Process one item at a time
3. Use smaller model
4. Clear cache between items

```python
# Solution 1: Reduce batch
for batch in batches:
    for image in batch[:2]:  # Process 2 at a time instead of 4
        result = extractor.extract(image)

# Solution 2: Clear cache
torch.cuda.empty_cache()
gc.collect()

# Solution 3: Use smaller model
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-7B-Instruct",  # Smaller
    device="cuda",
)

# Solution 4: Lower token limit
config = QwenTextPyTorchConfig(
    device="cuda",
    max_new_tokens=2048,  # Reduced
)
```

### Very Slow Processing

**Problem:** Processing taking much longer than expected.

**Solutions:**
1. Check GPU utilization
2. Use VLLM instead of PyTorch
3. Reduce image resolution
4. Verify model is on GPU

```python
import torch
import subprocess

# Check GPU usage
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
    capture_output=True, text=True
)
gpu_util = result.stdout.strip()
print(f"GPU Utilization: {gpu_util}%")

if gpu_util < "50%":
    # GPU not being fully used - try VLLM
    from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
    config = QwenTextVLLMConfig()
    extractor = QwenTextExtractor(backend=config)

# Verify model on GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
```

### Variable Processing Times

**Problem:** Some documents take much longer to process.

**Solutions:**
1. Check image sizes
2. Set token limit
3. Log processing times

```python
import time

for image_path in images:
    start = time.time()
    image = Image.open(image_path)

    # Check size
    if image.size[0] > 4000:
        print(f"Warning: Large image {image.size}, may be slow")

    result = extractor.extract(image)
    elapsed = time.time() - start

    # Flag slow items
    if elapsed > 5:
        print(f"Slow: {image_path.name} took {elapsed:.1f}s")

    # Limit tokens for very large documents
    if result.word_count > 5000:
        print(f"Very long output: {result.word_count} words")
```

### Failed Documents

**Problem:** Some documents fail to process.

**Solutions:**
1. Check file integrity
2. Try with different model
3. Check image format

```python
from PIL import Image
import traceback

for image_path in images:
    try:
        # Verify image
        image = Image.open(image_path)
        image.verify()

        # Reload (verify closes the file)
        image = Image.open(image_path)

        # Try extraction
        result = extractor.extract(image)

    except Exception as e:
        print(f"Failed {image_path.name}:")
        traceback.print_exc()

        # Try alternative
        try:
            # Fallback to Tesseract (simple OCR)
            from omnidocs.tasks.ocr_extraction import Tesseract
            ocr = Tesseract()
            result = ocr.extract(image)
            print("  Fallback succeeded with Tesseract")
        except:
            print("  Fallback also failed")
```

---

**Next Steps:**
- See [Text Extraction Guide](text-extraction.md) for extraction configuration
- See [Deployment Guide](deployment-modal.md) for scaling batches on GPU
- See [OCR Guide](ocr-extraction.md) for text with locations
