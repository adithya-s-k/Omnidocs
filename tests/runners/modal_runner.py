"""
Modal Test Runner for OmniDocs.

Runs OmniDocs tests on Modal with GPU concurrency control.
Supports filtering by task, backend, and compute type.

Usage:
    cd Omnidocs
    modal run tests/runners/modal_runner.py --task text_extraction --concurrency 5
    modal run tests/runners/modal_runner.py --backend vllm --concurrency 3
    modal run tests/runners/modal_runner.py --gpu-only --concurrency 5
    modal run tests/runners/modal_runner.py --list
    modal run tests/runners/modal_runner.py --test qwen_text_vllm
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import modal

from tests.runners.registry import Backend, Task, TestSpec, get_tests

# Modal configuration
MODEL_CACHE_DIR = "/data/.cache"
VOLUME_NAME = "omnidocs"

app = modal.App("omnidocs-tests")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# CUDA configuration for VLLM
VLLM_CUDA_VERSION = "12.8.1"
VLLM_FLAVOR = "devel"
VLLM_OS = "ubuntu24.04"
VLLM_TAG = f"{VLLM_CUDA_VERSION}-{VLLM_FLAVOR}-{VLLM_OS}"

# CUDA configuration for PyTorch
PYTORCH_CUDA_VERSION = "12.8.0"
PYTORCH_FLAVOR = "devel"
PYTORCH_OS = "ubuntu24.04"
PYTORCH_TAG = f"{PYTORCH_CUDA_VERSION}-{PYTORCH_FLAVOR}-{PYTORCH_OS}"

flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

# VLLM GPU Image
VLLM_GPU_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{VLLM_TAG}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .uv_pip_install(
        "transformers==4.57.6",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "numpy",
        "pydantic",
    )
    .uv_pip_install(
        "qwen-vl-utils",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
            "OMNIDOCS_MODEL_CACHE": MODEL_CACHE_DIR,
            "VLLM_USE_V1": "0",
        }
    )
)

# PyTorch GPU Image
PYTORCH_GPU_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{PYTORCH_TAG}", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .uv_pip_install(
        "torch==2.6",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        "transformers==4.57.6",
        "pillow",
        "numpy",
        "pydantic",
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
    )
    .uv_pip_install(flash_attn_wheel)
    .uv_pip_install(
        "ultralytics",
        "easyocr",
        "paddlepaddle",
        "paddleocr",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
            "OMNIDOCS_MODEL_CACHE": MODEL_CACHE_DIR,
        }
    )
)

# CPU Image (for CPU-only tests)
CPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr", "libglib2.0-0", "libgl1", "libglx-mesa0")
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers==4.57.6",
        "pillow",
        "numpy",
        "pydantic",
        "ultralytics",
        "easyocr",
        "paddlepaddle",
        "paddleocr",
        "pytesseract",
        "rtree",
    )
)


def get_image_for_backend(backend: Backend) -> modal.Image:
    """Get the appropriate Modal image for a backend type."""
    if backend == Backend.VLLM:
        return VLLM_GPU_IMAGE
    elif backend == Backend.PYTORCH_GPU:
        return PYTORCH_GPU_IMAGE
    elif backend in (Backend.PYTORCH_CPU, Backend.MLX, Backend.API):
        return CPU_IMAGE
    else:
        return CPU_IMAGE


def parse_gpu_type(gpu_type: Optional[str]) -> Optional[str]:
    """Parse GPU type string to Modal GPU spec."""
    if gpu_type is None:
        return None
    # GPU type format: "L40S:1", "A10G:1", "T4:1"
    return gpu_type


@app.function(
    image=VLLM_GPU_IMAGE,
    gpu="L40S:1",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def run_vllm_test(test_spec: Dict, image_bytes: bytes) -> Dict:
    """Run a VLLM backend test."""
    return _run_test_impl(test_spec, image_bytes)


@app.function(
    image=PYTORCH_GPU_IMAGE,
    gpu="A10G:1",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def run_pytorch_gpu_test(test_spec: Dict, image_bytes: bytes) -> Dict:
    """Run a PyTorch GPU backend test."""
    return _run_test_impl(test_spec, image_bytes)


@app.function(
    image=PYTORCH_GPU_IMAGE,
    gpu="T4:1",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def run_pytorch_gpu_light_test(test_spec: Dict, image_bytes: bytes) -> Dict:
    """Run a lighter PyTorch GPU test (T4 GPU for YOLO, RT-DETR, etc.)."""
    return _run_test_impl(test_spec, image_bytes)


@app.function(
    image=CPU_IMAGE,
    volumes={"/data": vol},
    timeout=600,
)
def run_cpu_test(test_spec: Dict, image_bytes: bytes) -> Dict:
    """Run a CPU-only test."""
    return _run_test_impl(test_spec, image_bytes)


def _run_test_impl(test_spec: Dict, image_bytes: bytes) -> Dict:
    """Internal implementation of test execution."""
    import importlib
    import io
    import sys
    import time
    from pathlib import Path

    from PIL import Image

    # Add the omnidocs package to path
    sys.path.insert(0, "/root")

    # Load the image
    img = Image.open(io.BytesIO(image_bytes))

    # Import the test module
    module_path = f"tests.standalone.{test_spec['module']}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        return {
            "success": False,
            "test_name": test_spec["name"],
            "error": f"Failed to import module {module_path}: {e}",
            "load_time": 0,
            "inference_time": 0,
        }

    # Find the test class (first class that ends with 'Test')
    test_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and name.endswith("Test")
            and name != "BaseOmnidocsTest"
        ):
            test_class = obj
            break

    if test_class is None:
        return {
            "success": False,
            "test_name": test_spec["name"],
            "error": f"No test class found in {module_path}",
            "load_time": 0,
            "inference_time": 0,
        }

    # Run the test
    test = test_class()
    result = test.run(img)

    return result.to_dict()


def select_runner_for_test(spec: TestSpec):
    """Select the appropriate Modal function for a test spec."""
    if spec.backend == Backend.VLLM:
        return run_vllm_test
    elif spec.backend == Backend.PYTORCH_GPU:
        # Use light GPU for simpler models
        if any(tag in spec.tags for tag in ["yolo", "rtdetr", "easyocr", "paddleocr", "tesseract", "tableformer"]):
            return run_pytorch_gpu_light_test
        return run_pytorch_gpu_test
    else:
        return run_cpu_test


@app.local_entrypoint()
def main(
    task: str = None,
    backend: str = None,
    concurrency: int = 5,
    gpu_only: bool = False,
    cpu_only: bool = False,
    list_tests: bool = False,
    test: str = None,
    image_path: str = None,
):
    """
    Run OmniDocs tests on Modal.

    Args:
        task: Filter by task (text_extraction, layout_extraction, ocr_extraction,
              table_extraction, reading_order)
        backend: Filter by backend (vllm, pytorch_gpu, pytorch_cpu, mlx, api)
        concurrency: Maximum number of concurrent GPU jobs
        gpu_only: Only run tests that require GPU
        cpu_only: Only run tests that run on CPU
        list_tests: List available tests without running
        test: Run a specific test by name
        image_path: Path to test image (required unless --list)
    """
    # Parse filters
    task_enum = Task(task) if task else None
    backend_enum = Backend(backend) if backend else None
    names = [test] if test else None

    # Get filtered tests
    tests = get_tests(
        task=task_enum,
        backend=backend_enum,
        gpu_only=gpu_only,
        cpu_only=cpu_only,
        names=names,
    )

    # Skip MLX and API tests on Modal (they run locally)
    tests = [t for t in tests if t.backend not in (Backend.MLX, Backend.API)]

    if list_tests:
        print(f"\n{'Name':<30} {'Task':<20} {'Backend':<15} {'GPU':<10}")
        print("-" * 75)
        for t in tests:
            gpu = t.gpu_type or "CPU"
            print(f"{t.name:<30} {t.task.value:<20} {t.backend.value:<15} {gpu:<10}")
        print(f"\nTotal: {len(tests)} tests")
        return

    if not image_path:
        print("Error: --image-path is required to run tests")
        print("Usage: modal run tests/runners/modal_runner.py --image-path test.png")
        return

    # Load test image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"\nRunning {len(tests)} tests with concurrency={concurrency}")
    print(f"Image: {image_path}")
    print("-" * 75)

    # Group tests by runner function for better concurrency control
    results = []
    start_time = datetime.now()

    # Run tests with their appropriate runners
    for spec in tests:
        runner = select_runner_for_test(spec)
        spec_dict = {
            "name": spec.name,
            "module": spec.module,
            "backend": spec.backend.value,
            "task": spec.task.value,
            "tags": spec.tags,
        }

        print(f"  Running {spec.name}...")
        try:
            result = runner.remote(spec_dict, image_bytes)
            results.append(result)
            status = "PASS" if result["success"] else "FAIL"
            print(f"    [{status}] {spec.name} ({result.get('inference_time', 0):.2f}s)")
        except Exception as e:
            results.append({
                "success": False,
                "test_name": spec.name,
                "error": str(e),
            })
            print(f"    [FAIL] {spec.name}: {e}")

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print("\n" + "=" * 75)
    print(f"SUMMARY: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("=" * 75)

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['test_name']}: {r.get('error', 'Unknown error')}")

    # Write results to file
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_file}")
