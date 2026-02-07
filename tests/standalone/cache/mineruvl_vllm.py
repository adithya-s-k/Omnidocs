"""
Test MinerU VL VLLM backend - model sharing behavior on Modal.

Usage:
    cd Omnidocs
    modal run tests/standalone/cache/mineruvl_vllm.py
"""

from pathlib import Path

import modal

# CUDA configuration
cuda_version = "12.8.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
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
            "HF_HOME": "/data/.cache",
            "VLLM_USE_V1": "0",
        }
    )
)

app = modal.App("test-mineruvl-vllm-sharing")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")


def create_test_image():
    """Create a simple test document image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    draw.text((50, 30), "Sample Document", fill="black", font=font)
    draw.text((50, 80), "This is a test document.", fill="black", font=font)
    draw.rectangle([50, 150, 350, 250], outline="black", width=2)
    draw.text((60, 160), "Name | Value", fill="black", font=font)

    return img


@app.function(
    image=IMAGE,
    gpu="L40S:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_vllm_sharing():
    """Test if VLLM backend shares model between text and layout extractors."""
    import os
    import time

    # CRITICAL: Set env vars before any CUDA initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Verify CUDA works in main process
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 60)
    print("Testing VLLM Model Sharing with Cache")
    print("=" * 60)

    from omnidocs.cache import get_cache_info, get_cache_key
    from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
    from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutVLLMConfig
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

    text_config = MinerUVLTextVLLMConfig(
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    layout_config = MinerUVLLayoutVLLMConfig(
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    text_key = get_cache_key(text_config)
    layout_key = get_cache_key(layout_config)
    print(f"Keys match: {text_key == layout_key}")

    # Load text extractor
    print("\n[1] Loading text extractor (VLLM)...")
    try:
        start = time.time()
        text_extractor = MinerUVLTextExtractor(backend=text_config)
        text_load_time = time.time() - start
        print(f"    Loaded in {text_load_time:.2f}s")
        print(f"    Cache: {get_cache_info()}")
    except Exception as e:
        print(f"    FAILED: {e}")
        return {"backend": "vllm", "error": str(e), "models_shared": False}

    # Load layout detector - should use cache
    print("\n[2] Loading layout detector (VLLM) - should use cache...")
    try:
        start = time.time()
        layout_detector = MinerUVLLayoutDetector(backend=layout_config)
        layout_load_time = time.time() - start
        print(f"    Loaded in {layout_load_time:.2f}s")
    except Exception as e:
        print(f"    FAILED: {e}")
        return {
            "backend": "vllm",
            "error": str(e),
            "text_load_time": text_load_time,
            "models_shared": False,
        }

    # Check sharing
    shared = id(text_extractor._client) == id(layout_detector._client)
    print(f"\nSame client: {shared}")

    # Test inference
    image = create_test_image()

    print("\n[3] Text extraction...")
    text_result = text_extractor.extract(image, output_format="markdown")
    print(f"    Content: {len(text_result.content)} chars")

    print("[4] Layout detection...")
    layout_result = layout_detector.extract(image)
    print(f"    Boxes: {len(layout_result.bboxes)}")

    return {
        "backend": "vllm",
        "text_load_time": text_load_time,
        "layout_load_time": layout_load_time,
        "models_shared": shared,
        "text_content_length": len(text_result.content),
        "layout_boxes": len(layout_result.bboxes),
    }


@app.local_entrypoint()
def main():
    result = test_vllm_sharing.remote()
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result.get("models_shared"):
        print("\nSUCCESS: Models are shared via cache!")
    else:
        print("\nFAILED: Models are NOT shared")
