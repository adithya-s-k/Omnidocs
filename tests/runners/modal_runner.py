"""
Modal Test Runner for OmniDocs.

Explicit test functions pattern - each test is a dedicated @app.function.
Installs omnidocs[vllm] or omnidocs[pytorch] from local source.

Usage:
    cd Omnidocs
    modal run tests/runners/modal_runner.py --test qwen_vllm
    modal run tests/runners/modal_runner.py --test granite_docling_pytorch
    modal run tests/runners/modal_runner.py --list
    modal run tests/runners/modal_runner.py --run-all
"""

from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent  # Omnidocs/
MODEL_CACHE_DIR = "/data/.cache"

# ============= Modal Images =============

cuda_vllm = "12.8.1"
VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_CACHE_DIR,
            "VLLM_USE_V1": "0",
        }
    )
)

cuda_pytorch = "12.8.0"
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .uv_pip_install(flash_attn_wheel)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_CACHE_DIR,
        }
    )
)

# ============= Modal App =============

app = modal.App("omnidocs-tests")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")


# ============= Test Image Generation =============


def create_test_image():
    """Create a simple test document image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    draw.text((50, 30), "Sample Document", fill="black", font=font)
    draw.text(
        (50, 80),
        "This is a test document for OCR extraction.",
        fill="black",
        font=small_font,
    )
    draw.text((50, 110), "It contains multiple lines of text.", fill="black", font=small_font)

    # Table
    draw.rectangle([50, 160, 350, 280], outline="black")
    draw.line([50, 200, 350, 200], fill="black")
    draw.line([150, 160, 150, 280], fill="black")
    draw.line([250, 160, 250, 280], fill="black")
    draw.text((70, 170), "Name", fill="black", font=small_font)
    draw.text((170, 170), "Value", fill="black", font=small_font)
    draw.text((270, 170), "Unit", fill="black", font=small_font)
    draw.text((70, 220), "Alpha", fill="black", font=small_font)
    draw.text((170, 220), "100", fill="black", font=small_font)
    draw.text((270, 220), "kg", fill="black", font=small_font)

    # List
    draw.text((50, 320), "Key Points:", fill="black", font=small_font)
    draw.text((50, 350), "- First item in the list", fill="black", font=small_font)
    draw.text((50, 380), "- Second item in the list", fill="black", font=small_font)
    draw.text((50, 410), "- Third item in the list", fill="black", font=small_font)

    return img


# ============= TEXT EXTRACTION TESTS =============


@app.function(
    image=VLLM_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_qwen_vllm(img_bytes: bytes) -> dict:
    """Test Qwen text extraction with VLLM backend."""
    import io
    import os
    import time

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from PIL import Image

    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing QwenTextExtractor with VLLM backend")
    print("=" * 60)

    start = time.time()
    extractor = QwenTextExtractor(
        backend=QwenTextVLLMConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            enforce_eager=True,
            download_dir=MODEL_CACHE_DIR,
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "qwen_vllm",
        "backend": "vllm",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_qwen_pytorch(img_bytes: bytes) -> dict:
    """Test Qwen text extraction with PyTorch backend."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing QwenTextExtractor with PyTorch backend")
    print("=" * 60)

    start = time.time()
    extractor = QwenTextExtractor(
        backend=QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="cuda",
            torch_dtype="bfloat16",
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "qwen_pytorch",
        "backend": "pytorch",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=VLLM_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_nanonets_vllm(img_bytes: bytes) -> dict:
    """Test Nanonets text extraction with VLLM backend."""
    import io
    import os
    import time

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from PIL import Image

    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextVLLMConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing NanonetsTextExtractor with VLLM backend")
    print("=" * 60)

    start = time.time()
    extractor = NanonetsTextExtractor(
        backend=NanonetsTextVLLMConfig(
            model="nanonets/Nanonets-OCR-s",
            gpu_memory_utilization=0.85,
            download_dir=MODEL_CACHE_DIR,
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "nanonets_vllm",
        "backend": "vllm",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_nanonets_pytorch(img_bytes: bytes) -> dict:
    """Test Nanonets text extraction with PyTorch backend."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing NanonetsTextExtractor with PyTorch backend")
    print("=" * 60)

    start = time.time()
    extractor = NanonetsTextExtractor(
        backend=NanonetsTextPyTorchConfig(
            model="nanonets/Nanonets-OCR-s",
            device="cuda",
            torch_dtype="bfloat16",
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "nanonets_pytorch",
        "backend": "pytorch",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=VLLM_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_dotsocr_vllm(img_bytes: bytes) -> dict:
    """Test DotsOCR text extraction with VLLM backend."""
    import io
    import os
    import time

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from PIL import Image

    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing DotsOCRTextExtractor with VLLM backend")
    print("=" * 60)

    start = time.time()
    extractor = DotsOCRTextExtractor(
        backend=DotsOCRVLLMConfig(
            model="rednote-hilab/dots.ocr",
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            enforce_eager=True,
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "dotsocr_vllm",
        "backend": "vllm",
        "model": "dots.ocr",
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_dotsocr_pytorch(img_bytes: bytes) -> dict:
    """Test DotsOCR text extraction with PyTorch backend."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing DotsOCRTextExtractor with PyTorch backend")
    print("=" * 60)

    start = time.time()
    extractor = DotsOCRTextExtractor(
        backend=DotsOCRPyTorchConfig(
            model="rednote-hilab/dots.ocr",
            device="cuda",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "dotsocr_pytorch",
        "backend": "pytorch",
        "model": "dots.ocr",
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=VLLM_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_granite_docling_vllm(img_bytes: bytes) -> dict:
    """Test Granite Docling text extraction with VLLM backend."""
    import io
    import os
    import time

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from PIL import Image

    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextVLLMConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing GraniteDoclingTextExtractor with VLLM backend")
    print("=" * 60)

    start = time.time()
    extractor = GraniteDoclingTextExtractor(
        backend=GraniteDoclingTextVLLMConfig(
            gpu_memory_utilization=0.85,
            download_dir=MODEL_CACHE_DIR,
            fast_boot=True,
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "granite_docling_vllm",
        "backend": "vllm",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_granite_docling_pytorch(img_bytes: bytes) -> dict:
    """Test Granite Docling text extraction with PyTorch backend."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import (
        GraniteDoclingTextPyTorchConfig,
    )

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing GraniteDoclingTextExtractor with PyTorch backend")
    print("=" * 60)

    start = time.time()
    extractor = GraniteDoclingTextExtractor(
        backend=GraniteDoclingTextPyTorchConfig(
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,  # Use SDPA instead due to flash-attn version mismatch
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Extracted Content ---")
    print(result.content[:500] if len(result.content) > 500 else result.content)
    print("---")

    return {
        "status": "success",
        "test": "granite_docling_pytorch",
        "backend": "pytorch",
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": load_time,
        "inference_time": inference_time,
    }


# ============= LAYOUT EXTRACTION TESTS =============


@app.function(
    image=VLLM_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_qwen_layout_vllm(img_bytes: bytes) -> dict:
    """Test Qwen layout detection with VLLM backend."""
    import io
    import os
    import time

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from PIL import Image

    from omnidocs.tasks.layout_extraction import QwenLayoutDetector
    from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing QwenLayoutDetector with VLLM backend")
    print("=" * 60)

    start = time.time()
    detector = QwenLayoutDetector(
        backend=QwenLayoutVLLMConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            enforce_eager=True,
            download_dir=MODEL_CACHE_DIR,
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = detector.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Detected Layout Elements ---")
    print(f"Number of boxes: {len(result.bboxes)}")
    for i, box in enumerate(result.bboxes[:5]):
        print(f"  {i + 1}. {box.label}: conf={box.confidence:.2f}")
    if len(result.bboxes) > 5:
        print(f"  ... and {len(result.bboxes) - 5} more")
    print("---")

    return {
        "status": "success",
        "test": "qwen_layout_vllm",
        "backend": "vllm",
        "model": "Qwen3-VL-4B",
        "num_boxes": len(result.bboxes),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def test_qwen_layout_pytorch(img_bytes: bytes) -> dict:
    """Test Qwen layout detection with PyTorch backend."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.layout_extraction import QwenLayoutDetector
    from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing QwenLayoutDetector with PyTorch backend")
    print("=" * 60)

    start = time.time()
    detector = QwenLayoutDetector(
        backend=QwenLayoutPyTorchConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="cuda",
            torch_dtype="bfloat16",
        )
    )
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = detector.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Detected Layout Elements ---")
    print(f"Number of boxes: {len(result.bboxes)}")
    for i, box in enumerate(result.bboxes[:5]):
        print(f"  {i + 1}. {box.label}: conf={box.confidence:.2f}")
    if len(result.bboxes) > 5:
        print(f"  ... and {len(result.bboxes) - 5} more")
    print("---")

    return {
        "status": "success",
        "test": "qwen_layout_pytorch",
        "backend": "pytorch",
        "model": "Qwen3-VL-4B",
        "num_boxes": len(result.bboxes),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_doclayout_yolo_gpu(img_bytes: bytes) -> dict:
    """Test DocLayoutYOLO with GPU."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing DocLayoutYOLO with GPU")
    print("=" * 60)

    start = time.time()
    extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Detected Layout Elements ---")
    print(f"Number of boxes: {len(result.bboxes)}")
    for i, box in enumerate(result.bboxes[:5]):
        print(f"  {i + 1}. {box.label}: conf={box.confidence:.2f}")
    print("---")

    return {
        "status": "success",
        "test": "doclayout_yolo_gpu",
        "backend": "pytorch_gpu",
        "model": "DocLayoutYOLO",
        "num_boxes": len(result.bboxes),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_rtdetr_gpu(img_bytes: bytes) -> dict:
    """Test RTDETRLayoutExtractor with GPU."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.layout_extraction import RTDETRConfig, RTDETRLayoutExtractor

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing RTDETRLayoutExtractor with GPU")
    print("=" * 60)

    start = time.time()
    extractor = RTDETRLayoutExtractor(config=RTDETRConfig(device="cuda"))
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Detected Layout Elements ---")
    print(f"Number of boxes: {len(result.bboxes)}")
    for i, box in enumerate(result.bboxes[:5]):
        print(f"  {i + 1}. {box.label}: conf={box.confidence:.2f}")
    print("---")

    return {
        "status": "success",
        "test": "rtdetr_gpu",
        "backend": "pytorch_gpu",
        "model": "RTDETR",
        "num_boxes": len(result.bboxes),
        "load_time": load_time,
        "inference_time": inference_time,
    }


# ============= OCR EXTRACTION TESTS =============


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_easyocr_gpu(img_bytes: bytes) -> dict:
    """Test EasyOCR with GPU."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing EasyOCR with GPU")
    print("=" * 60)

    start = time.time()
    extractor = EasyOCR(config=EasyOCRConfig(device="cuda"))
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- OCR Results ---")
    print(f"Number of text blocks: {len(result.text_blocks)}")
    for i, block in enumerate(result.text_blocks[:5]):
        print(f"  {i + 1}. '{block.text}' (conf={block.confidence:.2f})")
    print("---")

    return {
        "status": "success",
        "test": "easyocr_gpu",
        "backend": "pytorch_gpu",
        "model": "EasyOCR",
        "num_blocks": len(result.text_blocks),
        "load_time": load_time,
        "inference_time": inference_time,
    }


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_paddleocr_gpu(img_bytes: bytes) -> dict:
    """Test PaddleOCR with GPU."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing PaddleOCR with GPU")
    print("=" * 60)

    start = time.time()
    extractor = PaddleOCR(config=PaddleOCRConfig(device="cuda"))
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- OCR Results ---")
    print(f"Number of text blocks: {len(result.text_blocks)}")
    for i, block in enumerate(result.text_blocks[:5]):
        print(f"  {i + 1}. '{block.text}' (conf={block.confidence:.2f})")
    print("---")

    return {
        "status": "success",
        "test": "paddleocr_gpu",
        "backend": "pytorch_gpu",
        "model": "PaddleOCR",
        "num_blocks": len(result.text_blocks),
        "load_time": load_time,
        "inference_time": inference_time,
    }


# ============= TABLE EXTRACTION TESTS =============


@app.function(
    image=PYTORCH_IMAGE,
    gpu="L4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_tableformer_gpu(img_bytes: bytes) -> dict:
    """Test TableFormerExtractor with GPU."""
    import io
    import time

    from PIL import Image

    from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

    img = Image.open(io.BytesIO(img_bytes))

    print("=" * 60)
    print("Testing TableFormerExtractor with GPU")
    print("=" * 60)

    start = time.time()
    extractor = TableFormerExtractor(config=TableFormerConfig(device="cuda"))
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    result = extractor.extract(img)
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    print("\n--- Table Extraction Results ---")
    print(f"Number of tables: {len(result.tables)}")
    for i, table in enumerate(result.tables[:3]):
        print(f"  Table {i + 1}: {table.num_rows}x{table.num_cols}")
    print("---")

    return {
        "status": "success",
        "test": "tableformer_gpu",
        "backend": "pytorch_gpu",
        "model": "TableFormer",
        "num_tables": len(result.tables),
        "load_time": load_time,
        "inference_time": inference_time,
    }


# ============= Test Registry =============

AVAILABLE_TESTS = {
    # Text extraction
    "qwen_vllm": test_qwen_vllm,
    "qwen_pytorch": test_qwen_pytorch,
    "nanonets_vllm": test_nanonets_vllm,
    "nanonets_pytorch": test_nanonets_pytorch,
    "dotsocr_vllm": test_dotsocr_vllm,
    "dotsocr_pytorch": test_dotsocr_pytorch,
    "granite_docling_vllm": test_granite_docling_vllm,
    "granite_docling_pytorch": test_granite_docling_pytorch,
    # Layout extraction
    "qwen_layout_vllm": test_qwen_layout_vllm,
    "qwen_layout_pytorch": test_qwen_layout_pytorch,
    "doclayout_yolo_gpu": test_doclayout_yolo_gpu,
    "rtdetr_gpu": test_rtdetr_gpu,
    # OCR extraction
    "easyocr_gpu": test_easyocr_gpu,
    "paddleocr_gpu": test_paddleocr_gpu,
    # Table extraction
    "tableformer_gpu": test_tableformer_gpu,
}


# ============= Local Entrypoint =============


@app.local_entrypoint()
def main(test: str = "qwen_vllm", list_tests: bool = False, run_all: bool = False):
    """
    Run OmniDocs tests on Modal.

    Args:
        test: Test name (e.g., "qwen_vllm", "granite_docling_pytorch")
        list_tests: List all available tests
        run_all: Run all available tests
    """
    import io

    if list_tests:
        print("\nAvailable tests:")
        print("-" * 40)
        for name in sorted(AVAILABLE_TESTS.keys()):
            print(f"  {name}")
        print(f"\nTotal: {len(AVAILABLE_TESTS)} tests")
        return

    # Create test image
    img = create_test_image()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    print(f"Created test image: {img.size[0]}x{img.size[1]}")

    if run_all:
        print("\n" + "=" * 60)
        print("RUNNING ALL TESTS")
        print("=" * 60)

        all_results = []
        for test_name, test_fn in AVAILABLE_TESTS.items():
            print(f"\n>>> Running: {test_name}")
            try:
                result = test_fn.remote(img_bytes)
                all_results.append(result)
                print(f"    OK: {result['status']}")
            except Exception as e:
                all_results.append({"status": "failed", "test": test_name, "error": str(e)})
                print(f"    FAIL: {e}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in all_results if r["status"] == "success")
        failed = len(all_results) - passed
        print(f"Passed: {passed}/{len(all_results)}")
        print(f"Failed: {failed}/{len(all_results)}")

        for result in all_results:
            status = "OK" if result["status"] == "success" else "FAIL"
            test_name = result.get("test", "unknown")
            if result["status"] == "success":
                print(
                    f"  [{status}] {test_name}: "
                    f"load={result['load_time']:.1f}s, "
                    f"inference={result['inference_time']:.1f}s"
                )
            else:
                print(f"  [{status}] {test_name}: {result.get('error', 'unknown')}")
        return

    if test not in AVAILABLE_TESTS:
        print(f"Unknown test: {test}")
        print("\nAvailable tests:")
        for name in sorted(AVAILABLE_TESTS.keys()):
            print(f"  {name}")
        return

    print(f"\nRunning test: {test}")
    print("=" * 60)

    test_fn = AVAILABLE_TESTS[test]
    result = test_fn.remote(img_bytes)

    print("\n" + "=" * 60)
    print("TEST RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")
