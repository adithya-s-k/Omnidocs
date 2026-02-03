"""
Pytest configuration and fixtures for integration tests.
"""

import os
from pathlib import Path

import pytest
from PIL import Image

from tests.utils.synthetic_document import (
    create_simple_text_image,
    create_synthetic_document,
    create_table_image,
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: requires GPU for execution")
    config.addinivalue_line("markers", "cpu: CPU-only tests")
    config.addinivalue_line("markers", "vllm: VLLM backend tests")
    config.addinivalue_line("markers", "pytorch: PyTorch backend tests")
    config.addinivalue_line("markers", "mlx: MLX backend tests (Apple Silicon)")
    config.addinivalue_line("markers", "api: API backend tests")
    config.addinivalue_line("markers", "text_extraction: text extraction task tests")
    config.addinivalue_line("markers", "layout_extraction: layout extraction task tests")
    config.addinivalue_line("markers", "ocr_extraction: OCR extraction task tests")
    config.addinivalue_line("markers", "table_extraction: table extraction task tests")
    config.addinivalue_line("markers", "reading_order: reading order prediction task tests")
    config.addinivalue_line("markers", "integration: integration tests requiring model inference")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and available hardware."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_mlx = pytest.mark.skip(reason="MLX not available (requires Apple Silicon)")
    skip_api = pytest.mark.skip(reason="API credentials not configured")

    # Check for GPU availability
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    # Check for MLX availability
    import importlib.util

    has_mlx = importlib.util.find_spec("mlx") is not None

    # Check for API credentials
    has_api = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "mlx" in item.keywords and not has_mlx:
            item.add_marker(skip_mlx)
        if "api" in item.keywords and not has_api:
            item.add_marker(skip_api)


@pytest.fixture
def sample_document() -> Image.Image:
    """Create a sample synthetic document for testing."""
    doc = create_synthetic_document(
        width=800,
        height=1000,
        include_title=True,
        include_table=True,
    )
    return doc.image


@pytest.fixture
def sample_document_with_ground_truth():
    """Create a sample document with ground truth text."""
    doc = create_synthetic_document(
        width=800,
        height=1000,
        texts=[
            "This is a test paragraph for OCR evaluation.",
            "Another paragraph with different content.",
        ],
        include_title=True,
        include_table=False,
    )
    return doc


@pytest.fixture
def simple_text_image():
    """Create a simple text image for basic testing."""
    image, text = create_simple_text_image(
        text="Hello World",
        width=200,
        height=50,
    )
    return {"image": image, "text": text}


@pytest.fixture
def table_image():
    """Create a table image for table extraction testing."""
    image, data = create_table_image(
        rows=3,
        cols=3,
        include_headers=True,
    )
    return {"image": image, "data": data}


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def test_images_dir(fixtures_dir) -> Path:
    """Return the path to the test images directory."""
    return fixtures_dir / "images"
