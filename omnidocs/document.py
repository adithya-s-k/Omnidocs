"""
OmniDocs Document Loader

Stateless document container for loading and accessing PDF/image data.
Uses pypdfium2 (Apache 2.0) for PDF rendering and pdfplumber (MIT) for text extraction.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

try:
    import pypdfium2 as pdfium
except ImportError:
    raise ImportError("pypdfium2 is required for document loading. Install with: pip install pypdfium2")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# ============= Exceptions =============


class DocumentLoadError(Exception):
    """Failed to load document."""

    pass


class URLDownloadError(Exception):
    """Failed to download from URL."""

    pass


class PageRangeError(Exception):
    """Invalid page range."""

    pass


class UnsupportedFormatError(Exception):
    """Unsupported file format."""

    pass


# ============= Pydantic Models =============


class DocumentMetadata(BaseModel):
    """Metadata container for documents."""

    # Source information
    source_type: str = Field(..., description="Source type: file, url, bytes, image")
    source_path: Optional[str] = Field(default=None, description="Path or URL to source")
    file_name: Optional[str] = Field(default=None, description="File name")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")

    # PDF metadata
    pdf_metadata: Optional[Dict[str, Any]] = Field(default=None, description="PDF document metadata")

    # Document properties
    page_count: int = Field(..., description="Number of pages", ge=0)
    format: str = Field(default="pdf", description="Document format")

    # Image properties
    image_dpi: int = Field(default=150, description="Image DPI for rendering", ge=50, le=600)
    image_format: str = Field(default="RGB", description="Image color format")

    # Text extraction info
    text_extraction_engine: Optional[str] = Field(default=None, description="Engine used for text extraction")

    # Timestamps
    loaded_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="ISO timestamp of when document was loaded"
    )

    model_config = ConfigDict(extra="forbid")


# ============= Lazy Page Wrapper =============


class LazyPage:
    """
    Lazy page wrapper - renders only when accessed.

    This avoids loading all pages into memory upfront for large PDFs.
    """

    def __init__(self, pdf_doc: pdfium.PdfDocument, page_index: int, dpi: int = 150):
        self._pdf_doc = pdf_doc
        self._page_index = page_index
        self._dpi = dpi
        self._cached_image: Optional[Image.Image] = None
        self._cached_text: Optional[str] = None

    @property
    def image(self) -> Image.Image:
        """Render page to PIL Image (cached after first access)."""
        if self._cached_image is None:
            scale = self._dpi / 72  # PDF points to pixels
            page = self._pdf_doc[self._page_index]
            bitmap = page.render(scale=scale)
            self._cached_image = bitmap.to_pil().convert("RGB")
        return self._cached_image

    @property
    def text(self) -> str:
        """Extract text from page using pypdfium2 (cached)."""
        if self._cached_text is None:
            page = self._pdf_doc[self._page_index]
            textpage = page.get_textpage()
            self._cached_text = textpage.get_text_range()
        return self._cached_text

    @property
    def size(self) -> tuple:
        """Get page dimensions without full render (fast)."""
        page = self._pdf_doc[self._page_index]
        width, height = page.get_size()
        scale = self._dpi / 72
        return (int(width * scale), int(height * scale))

    def clear_cache(self):
        """Clear cached image to free memory."""
        self._cached_image = None


# ============= Document Class =============


class Document:
    """
    Stateless document container for OmniDocs.

    Features:
    - Lazy page rendering (pages only rendered when accessed)
    - Page caching (rendered pages cached to avoid re-rendering)
    - Multiple source support (PDF file, URL, bytes, images)
    - Text extraction with pypdfium2 first, pdfplumber fallback
    - Memory efficient for large documents

    Design:
    - Document is SOURCE DATA only - does NOT store task results
    - Users manage their own analysis results and caching strategy

    Examples:
        >>> # Load from file
        >>> doc = Document.from_pdf("paper.pdf")
        >>>
        >>> # Access pages
        >>> page = doc.get_page(0)  # 0-indexed
        >>> text = doc.get_page_text(1)  # 1-indexed for compatibility
        >>>
        >>> # Iterate efficiently
        >>> for page in doc.iter_pages():
        ...     result = layout.extract(page)
    """

    def __init__(
        self,
        pdf_doc: Optional[pdfium.PdfDocument],
        pdf_bytes: Optional[bytes],
        metadata: DocumentMetadata,
        dpi: int = 150,
        page_range: Optional[tuple] = None,
        preloaded_images: Optional[List[Image.Image]] = None,
    ):
        self._pdf_doc = pdf_doc
        self._pdf_bytes = pdf_bytes
        self._metadata = metadata
        self._dpi = dpi
        self._page_range = page_range

        # For image-based documents (no PDF)
        self._preloaded_images = preloaded_images

        # Lazy page wrappers
        self._lazy_pages: Optional[List[LazyPage]] = None
        if pdf_doc is not None:
            start = page_range[0] if page_range else 0
            end = page_range[1] if page_range else len(pdf_doc) - 1
            self._lazy_pages = [LazyPage(pdf_doc, i, dpi) for i in range(start, end + 1)]

        # Full text cache
        self._full_text_cache: Optional[str] = None

    # ============= Constructors =============

    @classmethod
    def from_pdf(
        cls,
        path: str,
        page_range: Optional[tuple] = None,
        dpi: int = 150,
    ) -> "Document":
        """
        Load document from PDF file (lazy - pages not rendered yet).

        Args:
            path: Path to PDF file
            page_range: Optional (start, end) tuple for page range (0-indexed, inclusive)
            dpi: Resolution for page rendering (default: 150)

        Returns:
            Document instance

        Raises:
            DocumentLoadError: If file not found
            UnsupportedFormatError: If not a PDF file
            PageRangeError: If page range is invalid

        Examples:
            >>> doc = Document.from_pdf("paper.pdf")
            >>> doc = Document.from_pdf("paper.pdf", page_range=(0, 4))
            >>> doc = Document.from_pdf("paper.pdf", dpi=300)
        """
        path = Path(path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise UnsupportedFormatError(f"Expected PDF file, got: {path.suffix}")

        # Read file bytes
        pdf_bytes = path.read_bytes()
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(pdf_doc)

        # Validate page range
        if page_range:
            start, end = page_range
            if start < 0 or end >= total_pages or start > end:
                raise PageRangeError(f"Invalid page range ({start}, {end}) for {total_pages} pages")

        # Extract metadata (fast, no rendering)
        pdf_meta = {}
        try:
            meta = pdf_doc.get_metadata_dict()
            if meta:
                pdf_meta = {k: v for k, v in meta.items() if v}
        except Exception:
            pass

        actual_pages = (page_range[1] - page_range[0] + 1) if page_range else total_pages

        metadata = DocumentMetadata(
            source_type="file",
            source_path=str(path.absolute()),
            file_name=path.name,
            file_size=len(pdf_bytes),
            pdf_metadata=pdf_meta or None,
            page_count=actual_pages,
            format="pdf",
            image_dpi=dpi,
        )

        return cls(
            pdf_doc=pdf_doc,
            pdf_bytes=pdf_bytes,
            metadata=metadata,
            dpi=dpi,
            page_range=page_range,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        page_range: Optional[tuple] = None,
        dpi: int = 150,
        timeout: int = 30,
    ) -> "Document":
        """
        Download and load document from URL (lazy).

        Args:
            url: URL to PDF file
            page_range: Optional (start, end) tuple for page range
            dpi: Resolution for page rendering
            timeout: Download timeout in seconds

        Returns:
            Document instance

        Raises:
            URLDownloadError: If download fails
            PageRangeError: If page range is invalid

        Examples:
            >>> doc = Document.from_url("https://example.com/doc.pdf")
            >>> doc = Document.from_url("https://example.com/doc.pdf", timeout=60)
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for URL downloads. Install with: pip install requests")

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise URLDownloadError(f"Failed to download: {e}")

        pdf_bytes = response.content
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(pdf_doc)

        if page_range:
            start, end = page_range
            if start < 0 or end >= total_pages or start > end:
                raise PageRangeError("Invalid page range")

        pdf_meta = {}
        try:
            meta = pdf_doc.get_metadata_dict()
            if meta:
                pdf_meta = {k: v for k, v in meta.items() if v}
        except Exception:
            pass

        file_name = url.split("/")[-1].split("?")[0]
        if not file_name.endswith(".pdf"):
            file_name = "downloaded.pdf"

        actual_pages = (page_range[1] - page_range[0] + 1) if page_range else total_pages

        metadata = DocumentMetadata(
            source_type="url",
            source_path=url,
            file_name=file_name,
            file_size=len(pdf_bytes),
            pdf_metadata=pdf_meta or None,
            page_count=actual_pages,
            format="pdf",
            image_dpi=dpi,
        )

        return cls(
            pdf_doc=pdf_doc,
            pdf_bytes=pdf_bytes,
            metadata=metadata,
            dpi=dpi,
            page_range=page_range,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: Optional[str] = None,
        page_range: Optional[tuple] = None,
        dpi: int = 150,
    ) -> "Document":
        """
        Load document from PDF bytes (lazy).

        Args:
            data: PDF file bytes
            filename: Optional filename for metadata
            page_range: Optional (start, end) tuple for page range
            dpi: Resolution for page rendering

        Returns:
            Document instance

        Raises:
            PageRangeError: If page range is invalid

        Examples:
            >>> with open("doc.pdf", "rb") as f:
            ...     doc = Document.from_bytes(f.read())
        """
        pdf_doc = pdfium.PdfDocument(data)
        total_pages = len(pdf_doc)

        if page_range:
            start, end = page_range
            if start < 0 or end >= total_pages or start > end:
                raise PageRangeError("Invalid page range")

        pdf_meta = {}
        try:
            meta = pdf_doc.get_metadata_dict()
            if meta:
                pdf_meta = {k: v for k, v in meta.items() if v}
        except Exception:
            pass

        actual_pages = (page_range[1] - page_range[0] + 1) if page_range else total_pages

        metadata = DocumentMetadata(
            source_type="bytes",
            file_name=filename or "document.pdf",
            file_size=len(data),
            pdf_metadata=pdf_meta or None,
            page_count=actual_pages,
            format="pdf",
            image_dpi=dpi,
        )

        return cls(
            pdf_doc=pdf_doc,
            pdf_bytes=data,
            metadata=metadata,
            dpi=dpi,
            page_range=page_range,
        )

    @classmethod
    def from_image(cls, path: str) -> "Document":
        """
        Load document from single image file.

        Args:
            path: Path to image file

        Returns:
            Document instance

        Raises:
            DocumentLoadError: If file not found

        Examples:
            >>> doc = Document.from_image("page.png")
        """
        path = Path(path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        img = Image.open(path).convert("RGB")

        metadata = DocumentMetadata(
            source_type="image",
            source_path=str(path.absolute()),
            file_name=path.name,
            file_size=path.stat().st_size,
            page_count=1,
            format=path.suffix.lower().replace(".", ""),
        )

        return cls(
            pdf_doc=None,
            pdf_bytes=None,
            metadata=metadata,
            preloaded_images=[img],
        )

    @classmethod
    def from_images(cls, paths: List[str]) -> "Document":
        """
        Load document from multiple images (multi-page).

        Args:
            paths: List of paths to image files

        Returns:
            Document instance

        Raises:
            DocumentLoadError: If any file not found

        Examples:
            >>> doc = Document.from_images(["page1.png", "page2.png"])
        """
        images = []
        total_size = 0

        for p in paths:
            path = Path(p)
            if not path.exists():
                raise DocumentLoadError(f"File not found: {path}")
            images.append(Image.open(path).convert("RGB"))
            total_size += path.stat().st_size

        metadata = DocumentMetadata(
            source_type="image",
            file_name=f"{len(paths)}_images",
            file_size=total_size,
            page_count=len(images),
            format="images",
        )

        return cls(
            pdf_doc=None,
            pdf_bytes=None,
            metadata=metadata,
            preloaded_images=images,
        )

    # ============= Properties =============

    @property
    def page_count(self) -> int:
        """Number of pages in document."""
        return self._metadata.page_count

    @property
    def metadata(self) -> DocumentMetadata:
        """Document metadata."""
        return self._metadata

    @property
    def pages(self) -> List[Image.Image]:
        """
        List of all page images.

        Note: This renders ALL pages. For large documents, use get_page() or iter_pages() instead.

        Returns:
            List of PIL Images
        """
        if self._preloaded_images:
            return self._preloaded_images

        if self._lazy_pages:
            return [lp.image for lp in self._lazy_pages]

        return []

    @property
    def text(self) -> str:
        """
        Full document text (lazy, cached).

        Uses pypdfium2 first (fast), falls back to pdfplumber if needed.

        Returns:
            Full document text
        """
        if self._full_text_cache is not None:
            return self._full_text_cache

        if self._lazy_pages:
            # Try pypdfium2 first (faster)
            texts = []
            for lp in self._lazy_pages:
                page_text = lp.text
                texts.append(page_text)

            combined = "\n\n".join(texts)

            # If pypdfium2 got reasonable text, use it
            if len(combined.strip()) > 50:
                self._full_text_cache = combined
                self._metadata.text_extraction_engine = "pypdfium2"
                return self._full_text_cache

            # Fallback to pdfplumber for complex layouts
            if self._pdf_bytes and pdfplumber:
                try:
                    with pdfplumber.open(io.BytesIO(self._pdf_bytes)) as pdf:
                        texts = []
                        start = self._page_range[0] if self._page_range else 0
                        end = self._page_range[1] if self._page_range else len(pdf.pages) - 1

                        for i in range(start, min(end + 1, len(pdf.pages))):
                            page_text = pdf.pages[i].extract_text() or ""
                            texts.append(page_text)

                        self._full_text_cache = "\n\n".join(texts)
                        self._metadata.text_extraction_engine = "pdfplumber"
                        return self._full_text_cache
                except Exception:
                    pass

        self._full_text_cache = ""
        self._metadata.text_extraction_engine = "none"
        return self._full_text_cache

    # ============= Methods =============

    def get_page(self, page_num: int) -> Image.Image:
        """
        Get single page image (0-indexed).

        More memory efficient than accessing .pages for large documents.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            PIL Image

        Raises:
            PageRangeError: If page number out of range

        Examples:
            >>> page = doc.get_page(0)  # First page
            >>> page = doc.get_page(doc.page_count - 1)  # Last page
        """
        if self._preloaded_images:
            if page_num < 0 or page_num >= len(self._preloaded_images):
                raise PageRangeError(f"Page {page_num} out of range (0-{len(self._preloaded_images) - 1})")
            return self._preloaded_images[page_num]

        if self._lazy_pages:
            if page_num < 0 or page_num >= len(self._lazy_pages):
                raise PageRangeError(f"Page {page_num} out of range (0-{len(self._lazy_pages) - 1})")
            return self._lazy_pages[page_num].image

        raise PageRangeError("No pages available")

    def get_page_text(self, page_num: int) -> str:
        """
        Get text for specific page (1-indexed for compatibility with PDF page numbers).

        Args:
            page_num: Page number (1-indexed, like PDF viewers)

        Returns:
            Page text

        Raises:
            PageRangeError: If page number out of range

        Examples:
            >>> text = doc.get_page_text(1)  # First page
        """
        idx = page_num - 1  # Convert to 0-based

        if self._lazy_pages:
            if idx < 0 or idx >= len(self._lazy_pages):
                raise PageRangeError(f"Page {page_num} out of range (1-{len(self._lazy_pages)})")
            return self._lazy_pages[idx].text

        return ""

    def get_page_size(self, page_num: int) -> tuple:
        """
        Get page dimensions without rendering (fast).

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (width, height) in pixels

        Examples:
            >>> width, height = doc.get_page_size(0)
        """
        if self._lazy_pages:
            if page_num < 0 or page_num >= len(self._lazy_pages):
                raise PageRangeError(f"Page {page_num} out of range")
            return self._lazy_pages[page_num].size

        if self._preloaded_images:
            if page_num < 0 or page_num >= len(self._preloaded_images):
                raise PageRangeError(f"Page {page_num} out of range")
            return self._preloaded_images[page_num].size

        raise PageRangeError("No pages available")

    def iter_pages(self) -> Iterator[Image.Image]:
        """
        Iterate over pages one at a time (memory efficient).

        Use this for large documents instead of .pages property.

        Yields:
            PIL Images

        Examples:
            >>> for page in doc.iter_pages():
            ...     result = layout.extract(page)
        """
        for i in range(self.page_count):
            yield self.get_page(i)

    def clear_cache(self, page_num: Optional[int] = None):
        """
        Clear cached page images to free memory.

        Args:
            page_num: Specific page to clear, or None for all pages

        Examples:
            >>> doc.clear_cache()  # Clear all
            >>> doc.clear_cache(0)  # Clear just first page
        """
        if self._lazy_pages:
            if page_num is not None:
                if 0 <= page_num < len(self._lazy_pages):
                    self._lazy_pages[page_num].clear_cache()
            else:
                for lp in self._lazy_pages:
                    lp.clear_cache()

    def save_images(
        self,
        output_dir: str,
        prefix: str = "page",
        format: str = "PNG",
    ) -> List[Path]:
        """
        Save all pages as individual image files.

        Args:
            output_dir: Output directory path
            prefix: Filename prefix (default: "page")
            format: Image format (default: "PNG")

        Returns:
            List of saved file paths

        Examples:
            >>> paths = doc.save_images("output/", prefix="doc", format="PNG")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for i in range(self.page_count):
            img = self.get_page(i)
            file_path = output_path / f"{prefix}_{i + 1:03d}.{format.lower()}"
            img.save(file_path, format=format)
            saved.append(file_path)

        return saved

    def to_dict(self) -> dict:
        """
        Convert document metadata to dictionary.

        Returns:
            Dictionary of metadata

        Examples:
            >>> data = doc.to_dict()
            >>> print(data['page_count'])
        """
        return self._metadata.model_dump()

    def close(self):
        """
        Close PDF document and free resources.

        Examples:
            >>> doc.close()
        """
        if self._pdf_doc:
            self._pdf_doc.close()
            self._pdf_doc = None
        self._lazy_pages = None
        self._pdf_bytes = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Document(pages={self.page_count}, source={self._metadata.source_type}, file={self._metadata.file_name})"
        )
