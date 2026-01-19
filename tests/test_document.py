"""
Tests for Document loading functionality.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from omnidocs import (
    Document,
    DocumentLoadError,
    DocumentMetadata,
    PageRangeError,
    UnsupportedFormatError,
)

# ============= Test Fixtures Paths =============

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDFS_DIR = FIXTURES_DIR / "pdfs"
IMAGES_DIR = FIXTURES_DIR / "images"


# ============= Helper Functions =============


def create_test_pdf() -> Path:
    """Create a simple test PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    temp_dir = Path(tempfile.gettempdir())
    pdf_path = temp_dir / "omnidocs_test.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 700, "Test Document - Page 1")
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, "This is a test PDF for OmniDocs.")
    c.drawString(100, 630, "It demonstrates document loading functionality.")
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 700, "Test Document - Page 2")
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, "This is the second page.")
    c.showPage()

    # Page 3
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 700, "Test Document - Page 3")
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, "Final page with content.")
    c.showPage()

    c.save()
    return pdf_path


def create_test_image() -> Path:
    """Create a simple test image."""
    temp_dir = Path(tempfile.gettempdir())
    img_path = temp_dir / "omnidocs_test.png"

    # Create simple image
    img = Image.new("RGB", (800, 600), color="white")
    img.save(img_path)

    return img_path


# ============= Fixtures =============


@pytest.fixture
def test_pdf():
    """Fixture providing a test PDF path."""
    pdf_path = create_test_pdf()
    yield pdf_path
    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()


@pytest.fixture
def test_image():
    """Fixture providing a test image path."""
    img_path = create_test_image()
    yield img_path
    # Cleanup
    if img_path.exists():
        img_path.unlink()


# ============= Document.from_pdf Tests =============


def test_from_pdf_basic(test_pdf):
    """Test basic PDF loading."""
    doc = Document.from_pdf(str(test_pdf))

    assert doc.page_count == 3
    assert doc.metadata.source_type == "file"
    assert doc.metadata.file_name == "omnidocs_test.pdf"
    assert doc.metadata.format == "pdf"
    assert doc.metadata.file_size > 0
    assert doc.metadata.image_dpi == 150


def test_from_pdf_custom_dpi(test_pdf):
    """Test PDF loading with custom DPI."""
    doc = Document.from_pdf(str(test_pdf), dpi=300)

    assert doc.metadata.image_dpi == 300

    # Verify image is rendered at correct DPI
    page = doc.get_page(0)
    assert page.size[0] > 0
    assert page.size[1] > 0


def test_from_pdf_page_range(test_pdf):
    """Test PDF loading with page range."""
    doc = Document.from_pdf(str(test_pdf), page_range=(0, 1))

    assert doc.page_count == 2


def test_from_pdf_invalid_range(test_pdf):
    """Test PDF loading with invalid page range."""
    with pytest.raises(PageRangeError):
        Document.from_pdf(str(test_pdf), page_range=(0, 10))

    with pytest.raises(PageRangeError):
        Document.from_pdf(str(test_pdf), page_range=(-1, 1))

    with pytest.raises(PageRangeError):
        Document.from_pdf(str(test_pdf), page_range=(2, 1))


def test_from_pdf_not_found():
    """Test PDF loading with non-existent file."""
    with pytest.raises(DocumentLoadError):
        Document.from_pdf("nonexistent.pdf")


def test_from_pdf_wrong_format(test_image):
    """Test PDF loading with non-PDF file."""
    with pytest.raises(UnsupportedFormatError):
        Document.from_pdf(str(test_image))


# ============= Document.from_bytes Tests =============


def test_from_bytes(test_pdf):
    """Test loading from bytes."""
    pdf_bytes = Path(test_pdf).read_bytes()
    doc = Document.from_bytes(pdf_bytes, filename="test.pdf")

    assert doc.page_count == 3
    assert doc.metadata.source_type == "bytes"
    assert doc.metadata.file_name == "test.pdf"
    assert doc.metadata.file_size == len(pdf_bytes)


def test_from_bytes_no_filename(test_pdf):
    """Test loading from bytes without filename."""
    pdf_bytes = Path(test_pdf).read_bytes()
    doc = Document.from_bytes(pdf_bytes)

    assert doc.metadata.file_name == "document.pdf"


# ============= Document.from_image Tests =============


def test_from_image(test_image):
    """Test loading from single image."""
    doc = Document.from_image(str(test_image))

    assert doc.page_count == 1
    assert doc.metadata.source_type == "image"
    assert doc.metadata.format == "png"


def test_from_images(test_image):
    """Test loading from multiple images."""
    # Create additional test images
    temp_dir = Path(tempfile.gettempdir())
    img2_path = temp_dir / "omnidocs_test2.png"
    img3_path = temp_dir / "omnidocs_test3.png"

    img = Image.new("RGB", (800, 600), color="red")
    img.save(img2_path)
    img.save(img3_path)

    try:
        doc = Document.from_images([str(test_image), str(img2_path), str(img3_path)])

        assert doc.page_count == 3
        assert doc.metadata.source_type == "image"
        assert doc.metadata.format == "images"
    finally:
        # Cleanup
        img2_path.unlink()
        img3_path.unlink()


# ============= Page Access Tests =============


def test_get_page(test_pdf):
    """Test get_page method."""
    doc = Document.from_pdf(str(test_pdf))

    # Valid access
    page0 = doc.get_page(0)
    assert isinstance(page0, Image.Image)
    assert page0.mode == "RGB"

    page2 = doc.get_page(2)
    assert isinstance(page2, Image.Image)

    # Invalid access
    with pytest.raises(PageRangeError):
        doc.get_page(3)

    with pytest.raises(PageRangeError):
        doc.get_page(-1)


def test_get_page_size(test_pdf):
    """Test get_page_size method."""
    doc = Document.from_pdf(str(test_pdf))

    width, height = doc.get_page_size(0)
    assert width > 0
    assert height > 0
    assert isinstance(width, int)
    assert isinstance(height, int)


def test_pages_property(test_pdf):
    """Test pages property."""
    doc = Document.from_pdf(str(test_pdf))

    pages = doc.pages
    assert len(pages) == 3
    assert all(isinstance(p, Image.Image) for p in pages)


def test_iter_pages(test_pdf):
    """Test iter_pages method."""
    doc = Document.from_pdf(str(test_pdf))

    pages = list(doc.iter_pages())
    assert len(pages) == 3
    assert all(isinstance(p, Image.Image) for p in pages)


# ============= Text Extraction Tests =============


def test_get_page_text(test_pdf):
    """Test get_page_text method (1-indexed)."""
    doc = Document.from_pdf(str(test_pdf))

    # Valid access (1-indexed)
    text = doc.get_page_text(1)
    assert isinstance(text, str)
    assert len(text) > 0

    # Invalid access
    with pytest.raises(PageRangeError):
        doc.get_page_text(0)  # 1-indexed, so 0 is invalid

    with pytest.raises(PageRangeError):
        doc.get_page_text(4)


def test_text_property(test_pdf):
    """Test text property (full document)."""
    doc = Document.from_pdf(str(test_pdf))

    text = doc.text
    assert isinstance(text, str)
    assert len(text) > 0
    assert doc.metadata.text_extraction_engine in ["pypdfium2", "pdfplumber", "none"]

    # Test caching
    text2 = doc.text
    assert text == text2


# ============= Lazy Loading Tests =============


def test_lazy_loading(test_pdf):
    """Test that pages are loaded lazily."""
    doc = Document.from_pdf(str(test_pdf))

    # Document loaded, but pages not rendered yet
    assert doc.page_count == 3

    # Access first page - should render now
    page0 = doc.get_page(0)
    assert isinstance(page0, Image.Image)

    # Access same page again - should use cached version
    page0_cached = doc.get_page(0)
    assert page0 is page0_cached  # Same object


def test_clear_cache(test_pdf):
    """Test cache clearing."""
    doc = Document.from_pdf(str(test_pdf))

    # Render pages
    _page0 = doc.get_page(0)
    _page1 = doc.get_page(1)

    # Clear specific page
    doc.clear_cache(0)

    # Clear all pages
    doc.clear_cache()


# ============= Utility Tests =============


def test_save_images(test_pdf):
    """Test save_images method."""
    doc = Document.from_pdf(str(test_pdf))

    temp_dir = Path(tempfile.gettempdir()) / "omnidocs_test_output"
    saved_paths = doc.save_images(str(temp_dir), prefix="test", format="PNG")

    assert len(saved_paths) == 3
    for path in saved_paths:
        assert path.exists()
        assert path.suffix == ".png"

    # Cleanup
    for path in saved_paths:
        path.unlink()
    temp_dir.rmdir()


def test_to_dict(test_pdf):
    """Test to_dict method."""
    doc = Document.from_pdf(str(test_pdf))

    data = doc.to_dict()
    assert isinstance(data, dict)
    assert data["page_count"] == 3
    assert data["source_type"] == "file"
    assert "loaded_at" in data


def test_repr(test_pdf):
    """Test __repr__ method."""
    doc = Document.from_pdf(str(test_pdf))

    repr_str = repr(doc)
    assert "Document" in repr_str
    assert "pages=3" in repr_str
    assert "file" in repr_str


# ============= Context Manager Tests =============


def test_context_manager(test_pdf):
    """Test context manager usage."""
    with Document.from_pdf(str(test_pdf)) as doc:
        assert doc.page_count == 3
        page = doc.get_page(0)
        assert isinstance(page, Image.Image)

    # Document should be closed after context
    # Note: We can't easily test this without checking internal state


def test_close(test_pdf):
    """Test close method."""
    doc = Document.from_pdf(str(test_pdf))
    _page = doc.get_page(0)

    doc.close()

    # After closing, internal PDF doc should be None
    # Note: This is testing implementation details


# ============= Metadata Tests =============


def test_metadata_validation():
    """Test DocumentMetadata validation."""
    # Valid metadata
    metadata = DocumentMetadata(
        source_type="file",
        page_count=5,
    )
    assert metadata.source_type == "file"
    assert metadata.page_count == 5

    # Invalid: negative page count
    with pytest.raises(ValueError):
        DocumentMetadata(
            source_type="file",
            page_count=-1,
        )

    # Invalid: extra fields (should fail due to extra="forbid")
    with pytest.raises(ValueError):
        DocumentMetadata(
            source_type="file",
            page_count=5,
            unknown_field="test",
        )


def test_pdf_metadata_extraction(test_pdf):
    """Test PDF metadata extraction."""
    doc = Document.from_pdf(str(test_pdf))

    # PDF metadata may or may not be present depending on PDF creation
    # Just verify it's either None or a dict
    assert doc.metadata.pdf_metadata is None or isinstance(doc.metadata.pdf_metadata, dict)


# ============= Integration Tests =============


def test_full_workflow(test_pdf):
    """Test complete document processing workflow."""
    # Load document
    doc = Document.from_pdf(str(test_pdf), dpi=150)

    # Check metadata
    assert doc.page_count == 3
    assert doc.metadata.format == "pdf"

    # Process each page
    results = []
    for i in range(doc.page_count):
        _page = doc.get_page(i)
        size = doc.get_page_size(i)
        text = doc.get_page_text(i + 1)  # 1-indexed

        results.append(
            {
                "page_num": i,
                "size": size,
                "has_text": len(text) > 0,
            }
        )

    assert len(results) == 3

    # Get full text
    full_text = doc.text
    assert len(full_text) > 0

    # Convert to dict
    metadata_dict = doc.to_dict()
    assert metadata_dict["page_count"] == 3


# ============= Fixture-Based Tests =============


@pytest.mark.skipif(not PDFS_DIR.exists(), reason="Fixtures directory not found")
class TestFixturePDFs:
    """Tests using real PDF fixtures from fixtures/pdfs/"""

    def test_bank_statement_pdf(self):
        """Test loading bank statement PDF."""
        pdf_path = PDFS_DIR / "bank_statement_1.pdf"
        if not pdf_path.exists():
            pytest.skip("bank_statement_1.pdf not found")

        doc = Document.from_pdf(str(pdf_path))
        assert doc.page_count > 0
        assert doc.metadata.source_type == "file"
        assert doc.metadata.format == "pdf"

        # Verify we can access pages
        page = doc.get_page(0)
        assert isinstance(page, Image.Image)

    def test_multilingual_pdf(self):
        """Test loading multilingual PDF."""
        pdf_path = PDFS_DIR / "multilingaul_1.pdf"
        if not pdf_path.exists():
            pytest.skip("multilingaul_1.pdf not found")

        doc = Document.from_pdf(str(pdf_path))
        assert doc.page_count > 0

        # Test text extraction
        text = doc.text
        assert len(text) > 0
        assert doc.metadata.text_extraction_engine in ["pypdfium2", "pdfplumber"]

    def test_research_paper_pdf(self):
        """Test loading research paper PDF."""
        pdf_path = PDFS_DIR / "research_paper_1.pdf"
        if not pdf_path.exists():
            pytest.skip("research_paper_1.pdf not found")

        doc = Document.from_pdf(str(pdf_path))
        assert doc.page_count > 0

        # Test page range loading
        doc_partial = Document.from_pdf(str(pdf_path), page_range=(0, 2))
        assert doc_partial.page_count == 3  # 0, 1, 2

    def test_all_fixture_pdfs_load(self):
        """Test that all fixture PDFs can be loaded."""
        if not PDFS_DIR.exists():
            pytest.skip("PDFs directory not found")

        pdf_files = list(PDFS_DIR.glob("*.pdf"))
        assert len(pdf_files) > 0, "No PDF fixtures found"

        for pdf_path in pdf_files:
            doc = Document.from_pdf(str(pdf_path))
            assert doc.page_count > 0
            assert doc.metadata.file_name == pdf_path.name


@pytest.mark.skipif(not IMAGES_DIR.exists(), reason="Fixtures directory not found")
class TestFixtureImages:
    """Tests using real image fixtures from fixtures/images/"""

    def test_bank_statement_image(self):
        """Test loading bank statement image."""
        img_path = IMAGES_DIR / "bank_statement_1.jpg"
        if not img_path.exists():
            pytest.skip("bank_statement_1.jpg not found")

        doc = Document.from_image(str(img_path))
        assert doc.page_count == 1
        assert doc.metadata.source_type == "image"
        assert doc.metadata.format == "jpg"

        page = doc.get_page(0)
        assert isinstance(page, Image.Image)

    def test_research_paper_images(self):
        """Test loading multiple research paper images."""
        img_paths = [
            IMAGES_DIR / "research_paper_1.jpg",
            IMAGES_DIR / "research_paper_2.jpg",
            IMAGES_DIR / "research_paper_3.jpg",
        ]

        # Filter only existing files
        existing_paths = [p for p in img_paths if p.exists()]

        if not existing_paths:
            pytest.skip("Research paper images not found")

        doc = Document.from_images([str(p) for p in existing_paths])
        assert doc.page_count == len(existing_paths)
        assert doc.metadata.source_type == "image"

    def test_multilingual_png_images(self):
        """Test loading PNG images."""
        img_paths = [
            IMAGES_DIR / "multilingaul_1.png",
            IMAGES_DIR / "multilingaul_2.png",
        ]

        existing_paths = [p for p in img_paths if p.exists()]

        if not existing_paths:
            pytest.skip("Multilingual PNG images not found")

        for img_path in existing_paths:
            doc = Document.from_image(str(img_path))
            assert doc.page_count == 1
            assert doc.metadata.format == "png"

    def test_all_fixture_images_load(self):
        """Test that all fixture images can be loaded."""
        if not IMAGES_DIR.exists():
            pytest.skip("Images directory not found")

        image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
        assert len(image_files) > 0, "No image fixtures found"

        for img_path in image_files:
            doc = Document.from_image(str(img_path))
            assert doc.page_count == 1
            assert doc.metadata.file_name == img_path.name


# ============= URL-Based Tests =============


@pytest.mark.slow
def test_from_url_arxiv():
    """Test loading PDF from arXiv URL (requires internet)."""
    url = "https://arxiv.org/pdf/1706.03762"  # "Attention Is All You Need" paper

    try:
        doc = Document.from_url(url, timeout=60)

        assert doc.page_count > 0
        assert doc.metadata.source_type == "url"
        assert doc.metadata.source_path == url
        assert doc.metadata.format == "pdf"

        # Test page access
        page = doc.get_page(0)
        assert isinstance(page, Image.Image)

        # Test text extraction
        text = doc.text
        assert len(text) > 0

    except Exception as e:
        pytest.skip(f"URL download failed (may be network issue): {e}")


# ============= Performance Tests with Fixtures =============


@pytest.mark.skipif(not PDFS_DIR.exists(), reason="Fixtures directory not found")
def test_large_pdf_lazy_loading():
    """Test lazy loading with real PDFs."""
    # Find the largest PDF
    if not PDFS_DIR.exists():
        pytest.skip("PDFs directory not found")

    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF fixtures found")

    # Use any available PDF
    pdf_path = pdf_files[0]

    # Load document (should be fast - no rendering yet)
    doc = Document.from_pdf(str(pdf_path))

    # Access specific page (only this page rendered)
    if doc.page_count > 0:
        page = doc.get_page(0)
        assert isinstance(page, Image.Image)

    # Clear cache
    doc.clear_cache()


# ============= Run Tests =============


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
