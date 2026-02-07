"""Test unified cache for PyTorch on Modal.

Usage:
    cd Omnidocs
    modal run tests/standalone/cache/unified_pytorch.py
"""

from pathlib import Path

import modal

# CUDA configuration
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent

IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

app = modal.App("test-unified-cache-pytorch")
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
    gpu="A10G:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def test_pytorch_unified_cache():
    """Test unified cache with PyTorch backend."""
    import time

    from omnidocs import clear_cache, get_cache_info
    from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
    from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

    image = create_test_image()
    print(f"Test image size: {image.size}")

    # Clear cache
    clear_cache()
    print(f"\nCache cleared: {get_cache_info()}")

    # Config
    text_config = MinerUVLTextPyTorchConfig(
        device="cuda",
        torch_dtype="float16",
        use_flash_attention=False,
    )
    layout_config = MinerUVLLayoutPyTorchConfig(
        device="cuda",
        torch_dtype="float16",
        use_flash_attention=False,
    )

    print("\n" + "=" * 60)
    print("Testing Unified Cache for PyTorch")
    print("=" * 60)

    print("\n[1] Loading MinerUVLTextExtractor (PyTorch)...")
    start = time.time()
    text_extractor = MinerUVLTextExtractor(backend=text_config)
    text_load = time.time() - start
    print(f"    Load time: {text_load:.2f}s")

    print(f"\nCache after text extractor: {get_cache_info()}")

    print("\n[2] Loading MinerUVLLayoutDetector (PyTorch)...")
    start = time.time()
    layout_detector = MinerUVLLayoutDetector(backend=layout_config)
    layout_load = time.time() - start
    print(f"    Load time: {layout_load:.2f}s")

    # Check if models are shared
    text_model_id = id(text_extractor._client.model)
    layout_model_id = id(layout_detector._client.model)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Text model id:   {text_model_id}")
    print(f"Layout model id: {layout_model_id}")
    print(f"Same model:      {text_model_id == layout_model_id}")

    # Test inference
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    print("\n[3] Running text extraction...")
    start = time.time()
    text_result = text_extractor.extract(image, output_format="markdown")
    print(f"    Inference time: {time.time() - start:.2f}s")
    print(f"    Content length: {len(text_result.content)} chars")

    print("\n[4] Running layout detection...")
    start = time.time()
    layout_result = layout_detector.extract(image)
    print(f"    Inference time: {time.time() - start:.2f}s")
    print(f"    Boxes detected: {len(layout_result.bboxes)}")

    shared = text_model_id == layout_model_id
    if shared:
        print("\n✅ Unified cache WORKS for PyTorch!")
        print(f"   Layout detector load time: {layout_load:.4f}s (instant)")
    else:
        print("\n❌ Cache did NOT work")

    return {
        "shared": shared,
        "text_load": text_load,
        "layout_load": layout_load,
        "inference_works": len(text_result.content) > 0,
    }


@app.local_entrypoint()
def main():
    result = test_pytorch_unified_cache.remote()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Models shared: {result['shared']}")
    print(f"Text load: {result['text_load']:.2f}s")
    print(f"Layout load: {result['layout_load']:.4f}s")
    print(f"Inference works: {result['inference_works']}")

    if result["shared"] and result["inference_works"]:
        print("\n✅ SUCCESS - Unified cache works on PyTorch/Modal!")
    else:
        print("\n❌ FAILED")
