"""
Test helpers for OmniDocs inference scripts.

Provides synthetic image generation, result verification, timing,
and environment setup utilities. No Modal imports.
"""

import json
import os
import sys
import time

# ============= Environment Setup =============


def setup_vllm_env():
    """Set VLLM environment variables before any imports.

    Must be called at the top of VLLM scripts, before importing
    torch, vllm, or omnidocs.
    """
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Force CUDA initialization in parent process before VLLM import.
    import torch

    torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.cuda.init()


def check_api_keys(*env_vars):
    """Check that required API keys are set. Exit(0) with SKIP if missing.

    Args:
        *env_vars: Environment variable names to check.
                   If none given, checks OPENROUTER_API_KEY and OPENAI_API_KEY.
    """
    if not env_vars:
        env_vars = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")

    for var in env_vars:
        if os.environ.get(var):
            return

    print(f"SKIP: No API keys found ({', '.join(env_vars)}). Skipping test.")
    sys.exit(0)


def check_gpu_available():
    """Check CUDA GPU is available. Exit(0) with SKIP if not."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("SKIP: No CUDA GPU available. Skipping test.")
            sys.exit(0)
    except ImportError:
        print("SKIP: PyTorch not installed. Skipping test.")
        sys.exit(0)


# ============= Test Image Generation =============


def create_test_image(width=800, height=1000):
    """Create a synthetic document image with title, paragraphs, table, and list.

    Returns:
        PIL.Image.Image: Synthetic document image.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    margin = 50
    y = margin

    # Title
    draw.text((margin, y), "Sample Document Title", fill="black", font=title_font)
    y += 40

    # Paragraphs
    paragraphs = [
        "This is the first paragraph of the document. It contains some sample text "
        "that can be used for testing text extraction capabilities.",
        "The second paragraph provides additional content. Testing various scenarios "
        "helps ensure robust extraction across different document layouts.",
        "A third paragraph concludes the main text section. This demonstrates multi-paragraph document handling.",
    ]

    text_width = width - 2 * margin
    for para in paragraphs:
        words = para.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            if bbox[2] > text_width:
                current_line.pop()
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

        for line in lines:
            draw.text((margin, y), line, fill="black", font=body_font)
            y += 20
        y += 15

    # Table
    y += 10
    table_data = [
        ["Header 1", "Header 2", "Header 3"],
        ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
        ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"],
    ]
    cell_w = (width - 2 * margin) // 3
    cell_h = 30

    for row_idx, row in enumerate(table_data):
        for col_idx, cell_text in enumerate(row):
            x1 = margin + col_idx * cell_w
            y1 = y + row_idx * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            draw.rectangle([x1, y1, x2, y2], outline="black", width=1)
            draw.text((x1 + 5, y1 + 5), cell_text, fill="black", font=body_font)

    y += len(table_data) * cell_h + 20

    # List
    draw.text((margin, y), "Key Points:", fill="black", font=body_font)
    y += 25
    for item in ["First item in the list", "Second item in the list", "Third item in the list"]:
        draw.text((margin, y), f"- {item}", fill="black", font=body_font)
        y += 22

    return img


def create_table_image(rows=4, cols=3):
    """Create a simple table image for table extraction tests.

    Returns:
        PIL.Image.Image: Table image.
    """
    from PIL import Image, ImageDraw, ImageFont

    cell_w, cell_h = 120, 40
    margin = 10
    width = cols * cell_w + 2 * margin
    height = rows * cell_h + 2 * margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * cell_w
            y1 = margin + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            draw.rectangle([x1, y1, x2, y2], outline="black", width=1)
            text = f"Col {c + 1}" if r == 0 else f"R{r}C{c}"
            draw.text((x1 + 5, y1 + 10), text, fill="black", font=font)

    return img


# ============= Timer =============


class Timer:
    """Context manager for timing blocks.

    Usage:
        with Timer("Model load") as t:
            model = load_model()
        print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.2f}s")


# ============= Result Verification =============


def verify_text_result(result, min_length=10):
    """Verify a text extraction result.

    Args:
        result: TextOutput from an extractor.
        min_length: Minimum content length to consider valid.
    """
    assert hasattr(result, "content"), "Result missing 'content' attribute"
    assert isinstance(result.content, str), f"content is {type(result.content)}, expected str"
    assert len(result.content) >= min_length, f"Content too short: {len(result.content)} chars (min {min_length})"
    print(f"[Verify] Text OK: {len(result.content)} chars")


def verify_layout_result(result, min_boxes=1):
    """Verify a layout extraction result.

    Args:
        result: LayoutOutput from a detector.
        min_boxes: Minimum number of bounding boxes expected.
    """
    assert hasattr(result, "bboxes"), "Result missing 'bboxes' attribute"
    assert len(result.bboxes) >= min_boxes, f"Too few boxes: {len(result.bboxes)} (min {min_boxes})"
    for box in result.bboxes:
        assert hasattr(box, "label"), "Box missing 'label' attribute"
        assert hasattr(box, "confidence"), "Box missing 'confidence' attribute"
    print(f"[Verify] Layout OK: {len(result.bboxes)} boxes")


def verify_ocr_result(result, min_blocks=1):
    """Verify an OCR extraction result.

    Args:
        result: OCROutput from an OCR extractor.
        min_blocks: Minimum number of text blocks expected.
    """
    assert hasattr(result, "text_blocks"), "Result missing 'text_blocks' attribute"
    assert len(result.text_blocks) >= min_blocks, f"Too few blocks: {len(result.text_blocks)} (min {min_blocks})"
    for block in result.text_blocks:
        assert hasattr(block, "text"), "Block missing 'text' attribute"
    print(f"[Verify] OCR OK: {len(result.text_blocks)} blocks")


def verify_table_result(result, min_tables=1):
    """Verify a table extraction result.

    Args:
        result: TableOutput from a table extractor.
        min_tables: Minimum number of tables expected.
    """
    assert hasattr(result, "cells"), "Result missing 'tables' attribute"
    assert len(result.cells) >= min_tables, f"Too few tables: {len(result.cells)} (min {min_tables})"
    print(f"[Verify] Table OK: {len(result.cells)} tables")


def verify_reading_order_result(result, min_elements=1):
    """Verify a reading order result.

    Args:
        result: ReadingOrderOutput from a predictor.
        min_elements: Minimum number of ordered elements expected.
    """
    assert hasattr(result, "ordered_elements"), "Result missing 'elements' attribute"
    assert len(result.ordered_elements) >= min_elements, (
        f"Too few elements: {len(result.ordered_elements)} (min {min_elements})"
    )
    print(f"[Verify] Reading order OK: {len(result.ordered_elements)} elements")


# ============= Result Output =============


def print_result(test_name, data_dict):
    """Print human-readable result and a JSON line for machine parsing.

    The JSON line starts with `__RESULT_JSON__:` so modal_runner can parse it.

    Args:
        test_name: Name of the test.
        data_dict: Dictionary of result data.
    """
    print("\n" + "=" * 60)
    print(f"TEST RESULT: {test_name}")
    print("=" * 60)
    for key, value in data_dict.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    result = {"status": "success", "test": test_name, **data_dict}
    print(f"__RESULT_JSON__:{json.dumps(result)}")
