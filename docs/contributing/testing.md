# Testing Guide

This guide covers how to run tests for OmniDocs across different platforms and backends.

## Test Architecture

OmniDocs uses a multi-tier testing approach:

| Tier | Platform | Tests | Command |
|------|----------|-------|---------|
| **Local CPU** | Any machine | CPU-based extractors | `uv run python -m tests.runners.local_runner --cpu-only` |
| **Local MLX** | Apple Silicon | MLX extractors | `uv run python -m tests.runners.local_runner --mlx` |
| **Modal GPU** | Cloud (L4/A10G) | VLLM, PyTorch GPU | `modal run scripts/.../modal_runner.py` |
| **pytest** | Any | Unit/integration | `uv run pytest` |

## Directory Structure

```
Omnidocs/tests/
├── fixtures/
│   └── images/              # Test images
│       └── test_simple.png
├── standalone/              # Standalone test scripts
│   ├── text_extraction/
│   │   ├── qwen_vllm.py
│   │   ├── qwen_pytorch.py
│   │   ├── qwen_mlx.py
│   │   ├── nanonets_vllm.py
│   │   └── ...
│   ├── layout_extraction/
│   │   ├── doclayout_yolo_cpu.py
│   │   ├── doclayout_yolo_gpu.py
│   │   ├── rtdetr_cpu.py
│   │   └── ...
│   ├── ocr_extraction/
│   ├── table_extraction/
│   └── reading_order/
├── runners/
│   ├── local_runner.py      # Local test runner
│   ├── registry.py          # Test registry
│   └── report.py            # Result reporting
├── integration/             # pytest integration tests
└── utils/                   # Test utilities
```

---

## Running Local Tests

### Prerequisites

```bash
cd Omnidocs

# Install with test dependencies
uv sync --group dev

# For MLX tests (Apple Silicon only)
uv sync --group mlx

# For OCR tests
uv sync --group ocr
```

### Using the Local Runner

The local runner executes standalone test scripts on your machine.

```bash
# Basic usage - run all CPU tests
uv run python -m tests.runners.local_runner \
    --image tests/fixtures/images/test_simple.png \
    --cpu-only

# Run all MLX tests (Apple Silicon)
uv run python -m tests.runners.local_runner \
    --image tests/fixtures/images/test_simple.png \
    --mlx

# Filter by task
uv run python -m tests.runners.local_runner \
    --image tests/fixtures/images/test_simple.png \
    --task layout_extraction \
    --cpu-only

# Run specific test
uv run python -m tests.runners.local_runner \
    --image tests/fixtures/images/test_simple.png \
    --test doclayout_yolo_cpu
```

### Local Runner Options

| Option | Description | Example |
|--------|-------------|---------|
| `--image` | Path to test image (required) | `tests/fixtures/images/test_simple.png` |
| `--cpu-only` | Run only CPU tests | |
| `--mlx` | Run only MLX tests | |
| `--task` | Filter by task type | `text_extraction`, `layout_extraction` |
| `--test` | Run specific test | `doclayout_yolo_cpu` |
| `--output` | Output JSON file | `results.json` |

### Example Output

```
Running 4 tests
Image: tests/fixtures/images/test_simple.png
---------------------------------------------------------------------------
  Running qwen_layout_mlx... [PASS] (3.78s)
  Running qwen_layout_api... [FAIL] (0.00s)
    Error: api_key required
  Running doclayout_yolo_cpu... [PASS] (0.46s)
  Running rtdetr_cpu... [PASS] (0.90s)

===========================================================================
SUMMARY: 3 passed, 1 failed (13.2s)
===========================================================================
```

---

## Running GPU Tests on Modal

GPU tests run on Modal cloud infrastructure with NVIDIA GPUs.

### Prerequisites

1. Install Modal CLI:
   ```bash
   pip install modal
   ```

2. Authenticate:
   ```bash
   modal setup
   ```

3. Create HuggingFace secret (for model downloads):
   ```bash
   modal secret create adithya-hf-wandb HF_TOKEN=your_hf_token
   ```

### Using the Modal Runner

```bash
cd /path/to/omnidocs_Master

# List all available tests
modal run scripts/text_extract_omnidocs/modal_runner.py --list-tests

# Run a specific test
modal run scripts/text_extract_omnidocs/modal_runner.py --test qwen_vllm

# Run all tests
modal run scripts/text_extract_omnidocs/modal_runner.py --run-all
```

### Available Modal Tests

#### Text Extraction
| Test | Backend | GPU | Model |
|------|---------|-----|-------|
| `qwen_vllm` | VLLM | L4 | Qwen3-VL-4B |
| `qwen_pytorch` | PyTorch | L4 | Qwen3-VL-4B |
| `nanonets_vllm` | VLLM | L4 | Nanonets-OCR-s |
| `nanonets_pytorch` | PyTorch | L4 | Nanonets-OCR-s |
| `dotsocr_vllm` | VLLM | L4 | dots.ocr |
| `dotsocr_pytorch` | PyTorch | L4 | dots.ocr |

#### Layout Extraction
| Test | Backend | GPU | Model |
|------|---------|-----|-------|
| `qwen_layout_vllm` | VLLM | L4 | Qwen3-VL-4B |
| `qwen_layout_pytorch` | PyTorch | L4 | Qwen3-VL-4B |
| `doclayout_yolo_gpu` | PyTorch | L4 | DocLayoutYOLO |
| `rtdetr_gpu` | PyTorch | L4 | RTDETR |

### Example Output

```
Running test: doclayout_yolo_gpu
============================================================
Testing DocLayoutYOLO with GPU
============================================================
Model load time: 8.34s
Inference time: 2.27s

--- Detected Layout Elements ---
Number of boxes: 8
  1. LayoutLabel.TITLE: conf=0.32
  2. LayoutLabel.TEXT: conf=0.72
  ...

============================================================
TEST RESULTS
============================================================
  status: success
  test: doclayout_yolo_gpu
  backend: pytorch_gpu
  model: DocLayoutYOLO
  num_boxes: 8
  load_time: 8.34
  inference_time: 2.27
```

---

## Running pytest

For unit tests and integration tests:

```bash
cd Omnidocs

# Run all tests
uv run pytest

# Run with specific markers
uv run pytest -m "cpu"              # CPU-only tests
uv run pytest -m "not slow"         # Skip slow tests
uv run pytest -m "layout_extraction" # Layout tests only

# Run specific test file
uv run pytest tests/integration/test_layout_extractors.py -v

# Run with coverage
uv run pytest --cov=omnidocs
```

### pytest Markers

Defined in `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `slow` | Long-running tests (network, large files) |
| `gpu` | Requires GPU |
| `cpu` | CPU-only tests |
| `vllm` | VLLM backend tests |
| `pytorch` | PyTorch backend tests |
| `mlx` | MLX backend tests (Apple Silicon) |
| `api` | API backend tests |
| `text_extraction` | Text extraction task |
| `layout_extraction` | Layout extraction task |
| `ocr_extraction` | OCR extraction task |
| `table_extraction` | Table extraction task |
| `reading_order` | Reading order task |
| `integration` | Integration tests requiring model inference |

---

## Writing Tests

### Standalone Test Template

Create a new test in `tests/standalone/<task>/<model>_<backend>.py`:

```python
"""
Model Name - Backend

Usage:
    python -m tests.standalone.<task>.<model>_<backend> path/to/image.png
"""
import sys
import time
from pathlib import Path
from PIL import Image


def run_extraction(img: Image.Image) -> dict:
    """Run extraction and return results."""
    from omnidocs.tasks.<task> import MyExtractor, MyConfig

    start = time.time()
    extractor = MyExtractor(config=MyConfig(device="cpu"))
    load_time = time.time() - start

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start

    return {
        "result": result,
        "load_time": load_time,
        "inference_time": inference_time,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.standalone.<task>.<model>_<backend> <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    result = run_extraction(img)
    print(f"Load time: {result['load_time']:.2f}s")
    print(f"Inference time: {result['inference_time']:.2f}s")
```

### Register the Test

Add to `tests/runners/registry.py`:

```python
from .registry import TestSpec, Backend, Task

# Add to TEST_REGISTRY list
TestSpec(
    name="mymodel_cpu",
    module="<task>.mymodel_cpu",
    backend=Backend.PYTORCH_CPU,
    task=Task.<TASK>,
    gpu_type=None,  # None for CPU tests
),
```

---

## Troubleshooting

### VLLM Multiprocessing Error

If you see `Cannot re-initialize CUDA in forked subprocess`:

```python
import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

### Flash Attention Version Mismatch

Use `attn_implementation="sdpa"` instead of `flash_attention_2`:

```python
config = MyConfig(attn_implementation="sdpa")
```

### MLX Tests Fail on Non-Apple Hardware

MLX only works on Apple Silicon. Skip with:

```bash
uv run python -m tests.runners.local_runner --cpu-only  # Exclude MLX
```

### API Tests Need Credentials

API tests require environment variables:

```bash
export OPENROUTER_API_KEY=your_key
uv run python -m tests.runners.local_runner --test qwen_layout_api
```

---

## Test Results Reference

### Text Extraction Performance (Modal L4 GPU)

| Model | Backend | Load Time | Inference Time |
|-------|---------|-----------|----------------|
| Qwen3-VL-4B | VLLM | 84s | 7.0s |
| Qwen3-VL-4B | PyTorch | 54s | 6.2s |
| Nanonets-OCR-s | VLLM | 194s | 8.4s |
| Nanonets-OCR-s | PyTorch | 44s | 6.3s |
| DotsOCR | VLLM | 94s | 10.0s |
| DotsOCR | PyTorch | 42s | 11.4s |

### Layout Extraction Performance

| Model | Backend | Load Time | Inference Time |
|-------|---------|-----------|----------------|
| Qwen Layout | VLLM (L4) | 237s | 27.2s |
| Qwen Layout | PyTorch (L4) | 54s | 13.3s |
| Qwen Layout | MLX (local) | 8.8s | 13.7s |
| DocLayoutYOLO | GPU (L4) | 8.3s | 2.3s |
| DocLayoutYOLO | CPU (local) | 0.5s | 0.3s |
| RTDETR | GPU (L4) | 12.4s | 1.9s |
| RTDETR | CPU (local) | 0.9s | 0.5s |
