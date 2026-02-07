"""
Test MinerU VL MLX backend - model sharing behavior.

Tests current behavior (loads twice) vs desired behavior (load once, share).

Usage:
    uv run python tests/standalone/cache/mineruvl_mlx.py
"""

import time
from pathlib import Path

from PIL import Image


def get_test_image():
    """Get a test image."""
    test_images = [
        Path("tests/fixtures/images/research_paper_1.jpg"),
        Path("tests/fixtures/images/test_simple.png"),
    ]
    for img_path in test_images:
        if img_path.exists():
            return Image.open(img_path)

    # Create a simple test image if none found
    img = Image.new("RGB", (800, 600), "white")
    return img


def test_current_behavior():
    """Test current behavior - does it load model twice?"""
    print("=" * 60)
    print("Testing CURRENT behavior (separate loads)")
    print("=" * 60)

    from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
    from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutMLXConfig
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

    image = get_test_image()
    print(f"Test image size: {image.size}")

    # First load - Text Extractor
    print("\n[1] Loading MinerUVLTextExtractor (MLX)...")
    start = time.time()
    text_extractor = MinerUVLTextExtractor(backend=MinerUVLTextMLXConfig())
    text_load_time = time.time() - start
    print(f"    Text extractor load time: {text_load_time:.2f}s")

    # Second load - Layout Detector
    print("\n[2] Loading MinerUVLLayoutDetector (MLX)...")
    start = time.time()
    layout_detector = MinerUVLLayoutDetector(backend=MinerUVLLayoutMLXConfig())
    layout_load_time = time.time() - start
    print(f"    Layout detector load time: {layout_load_time:.2f}s")

    # Check if they share the same model (they currently don't)
    text_model_id = id(text_extractor._client.model)
    layout_model_id = id(layout_detector._client.model)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Text extractor model id:   {text_model_id}")
    print(f"Layout detector model id:  {layout_model_id}")
    print(f"Same model object?         {text_model_id == layout_model_id}")
    print(f"Total load time:           {text_load_time + layout_load_time:.2f}s")

    if text_model_id == layout_model_id:
        print("\n✅ Models are SHARED (good!)")
    else:
        print("\n❌ Models are NOT shared (loaded twice - wasteful)")

    # Test both work correctly
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    print("\n[3] Running text extraction...")
    start = time.time()
    text_result = text_extractor.extract(image, output_format="markdown")
    text_inference_time = time.time() - start
    print(f"    Inference time: {text_inference_time:.2f}s")
    print(f"    Content length: {len(text_result.content)} chars")

    print("\n[4] Running layout detection...")
    start = time.time()
    layout_result = layout_detector.extract(image)
    layout_inference_time = time.time() - start
    print(f"    Inference time: {layout_inference_time:.2f}s")
    print(f"    Boxes detected: {len(layout_result.bboxes)}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Text load:       {text_load_time:.2f}s")
    print(f"Layout load:     {layout_load_time:.2f}s")
    print(f"Text inference:  {text_inference_time:.2f}s")
    print(f"Layout inference:{layout_inference_time:.2f}s")
    print(f"Total time:      {text_load_time + layout_load_time + text_inference_time + layout_inference_time:.2f}s")

    return {
        "text_load_time": text_load_time,
        "layout_load_time": layout_load_time,
        "models_shared": text_model_id == layout_model_id,
        "text_content_length": len(text_result.content),
        "layout_boxes": len(layout_result.bboxes),
    }


if __name__ == "__main__":
    results = test_current_behavior()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    if not results["models_shared"]:
        print("Currently loading model TWICE.")
        print(f"Wasted time: ~{results['layout_load_time']:.2f}s")
        print("Implementing cache would eliminate this overhead.")
