"""
Fixtures for layout extraction tests.
"""

import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple test image."""
    # Create a simple white image with some colored regions
    img = Image.new("RGB", (800, 600), color="white")
    return img


@pytest.fixture
def sample_document_image() -> Image.Image:
    """Create a more realistic document-like test image."""
    from PIL import ImageDraw

    img = Image.new("RGB", (1024, 1400), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a "title" region (dark text area at top)
    draw.rectangle([100, 50, 924, 120], fill="lightgray", outline="black")

    # Draw "text" regions
    draw.rectangle([100, 150, 924, 400], fill="whitesmoke", outline="gray")
    draw.rectangle([100, 450, 924, 700], fill="whitesmoke", outline="gray")

    # Draw a "figure" region
    draw.rectangle([100, 750, 500, 1000], fill="lightblue", outline="blue")

    # Draw a "table" region
    draw.rectangle([520, 750, 924, 1000], fill="lightyellow", outline="orange")

    return img
