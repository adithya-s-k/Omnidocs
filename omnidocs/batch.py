"""
OmniDocs Batch Processing Utilities.

Provides utilities for processing multiple documents efficiently:
- DocumentBatch: Load and iterate over multiple PDFs
- process_directory: Convenience function for batch processing
- process_document: Process all pages of a single document
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional

from .document import Document

if TYPE_CHECKING:
    from .utils.aggregation import BatchResult, DocumentResult


class DocumentBatch:
    """
    Batch document loader for processing multiple PDFs.

    Features:
    - Lazy loading (documents loaded on iteration)
    - Memory efficient (processes one document at a time)
    - Glob pattern support
    - Progress callbacks

    Examples:
        ```python
        # Load from directory
        batch = DocumentBatch.from_directory("pdfs/")

        # Load from list
        batch = DocumentBatch.from_paths(["doc1.pdf", "doc2.pdf"])

        # Iterate
        for doc in batch:
            for page in doc.iter_pages():
                result = extractor.extract(page)
        ```
    """

    def __init__(
        self,
        paths: List[Path],
        dpi: int = 150,
        page_range: Optional[tuple] = None,
    ):
        """
        Initialize DocumentBatch.

        Args:
            paths: List of PDF file paths
            dpi: Resolution for page rendering (default: 150)
            page_range: Optional (start, end) tuple for page range (applied to all docs)
        """
        self._paths = paths
        self._dpi = dpi
        self._page_range = page_range

    @classmethod
    def from_directory(
        cls,
        directory: str,
        pattern: str = "*.pdf",
        recursive: bool = False,
        dpi: int = 150,
        page_range: Optional[tuple] = None,
    ) -> "DocumentBatch":
        """
        Load all PDFs from directory.

        Args:
            directory: Path to directory
            pattern: Glob pattern (default: "*.pdf")
            recursive: Search subdirectories
            dpi: Resolution for rendering
            page_range: Optional page range for all documents

        Returns:
            DocumentBatch instance

        Raises:
            FileNotFoundError: If directory doesn't exist

        Examples:
            ```python
            batch = DocumentBatch.from_directory("pdfs/")
            batch = DocumentBatch.from_directory("docs/", pattern="*.pdf", recursive=True)
            ```
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            paths = list(dir_path.rglob(pattern))
        else:
            paths = list(dir_path.glob(pattern))

        paths = sorted(paths)  # Consistent ordering

        return cls(paths=paths, dpi=dpi, page_range=page_range)

    @classmethod
    def from_paths(
        cls,
        paths: List[str],
        dpi: int = 150,
        page_range: Optional[tuple] = None,
    ) -> "DocumentBatch":
        """
        Load documents from explicit list of paths.

        Args:
            paths: List of PDF paths
            dpi: Resolution for rendering
            page_range: Optional page range for all documents

        Returns:
            DocumentBatch instance

        Examples:
            ```python
            batch = DocumentBatch.from_paths(["doc1.pdf", "doc2.pdf"])
            ```
        """
        return cls(
            paths=[Path(p) for p in paths],
            dpi=dpi,
            page_range=page_range,
        )

    @property
    def count(self) -> int:
        """Number of documents in batch."""
        return len(self._paths)

    @property
    def paths(self) -> List[Path]:
        """List of document paths."""
        return self._paths

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Document]:
        """
        Iterate over documents (lazy loading).

        Each document is loaded when iterated, then available
        until the next iteration or explicit close.

        Yields:
            Document instances
        """
        for path in self._paths:
            doc = Document.from_pdf(
                str(path),
                page_range=self._page_range,
                dpi=self._dpi,
            )
            yield doc

    def iter_with_progress(
        self,
        callback: Callable[[int, int, str], None],
    ) -> Iterator[Document]:
        """
        Iterate with progress callback.

        Args:
            callback: Function(current, total, filename) called for each document

        Yields:
            Document instances

        Examples:
            ```python
            def progress(current, total, filename):
                print(f"[{current}/{total}] {filename}")

            for doc in batch.iter_with_progress(progress):
                # Process document...
            ```
        """
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            callback(i + 1, total, path.name)
            doc = Document.from_pdf(
                str(path),
                page_range=self._page_range,
                dpi=self._dpi,
            )
            yield doc

    def iter_all_pages(self) -> Iterator[tuple]:
        """
        Iterate over all pages from all documents.

        Memory efficient - loads one document at a time.

        Yields:
            Tuples of (doc_index, page_index, page_image, doc_path)

        Examples:
            ```python
            for doc_idx, page_idx, page_img, doc_path in batch.iter_all_pages():
                result = extractor.extract(page_img)
            ```
        """
        for doc_idx, path in enumerate(self._paths):
            doc = Document.from_pdf(
                str(path),
                page_range=self._page_range,
                dpi=self._dpi,
            )
            for page_idx in range(doc.page_count):
                yield (doc_idx, page_idx, doc.get_page(page_idx), path)
            doc.close()


def process_document(
    document: Document,
    extractor: Any,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **extract_kwargs,
) -> "DocumentResult":
    """
    Process all pages of a single document.

    Args:
        document: Document instance
        extractor: Initialized extractor (any type)
        progress_callback: Optional function(current, total) for progress
        **extract_kwargs: Passed to extractor.extract()

    Returns:
        DocumentResult with page results

    Examples:
        ```python
        from omnidocs import Document
        from omnidocs.batch import process_document

        doc = Document.from_pdf("paper.pdf")
        result = process_document(doc, extractor, output_format="markdown")
        result.save_json("output.json")
        ```
    """
    from .utils.aggregation import DocumentResult

    doc_result = DocumentResult(
        source_path=document.metadata.source_path,
        page_count=document.page_count,
    )

    for page_idx in range(document.page_count):
        if progress_callback:
            progress_callback(page_idx + 1, document.page_count)

        page = document.get_page(page_idx)
        result = extractor.extract(page, **extract_kwargs)
        doc_result.add_page_result(page_idx, result)

    return doc_result


def process_directory(
    directory: str,
    extractor: Any,
    output_dir: Optional[str] = None,
    pattern: str = "*.pdf",
    recursive: bool = False,
    dpi: int = 150,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    **extract_kwargs,
) -> "BatchResult":
    """
    Process all PDFs in a directory.

    Convenience function for common batch processing pattern.

    Args:
        directory: Path to directory with PDFs
        extractor: Initialized extractor instance
        output_dir: Optional directory to save results as JSON
        pattern: Glob pattern for files (default: "*.pdf")
        recursive: Search subdirectories
        dpi: Resolution for page rendering
        progress_callback: Function(filename, current, total) for progress
        **extract_kwargs: Passed to extractor.extract()

    Returns:
        BatchResult with all document results

    Examples:
        ```python
        from omnidocs.batch import process_directory
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

        extractor = QwenTextExtractor(
            backend=QwenTextPyTorchConfig(model="Qwen/Qwen2-VL-7B")
        )

        results = process_directory(
            "pdfs/",
            extractor,
            output_dir="results/",
            output_format="markdown",
        )
        ```
    """
    from .utils.aggregation import BatchResult, DocumentResult

    batch = DocumentBatch.from_directory(
        directory,
        pattern=pattern,
        recursive=recursive,
        dpi=dpi,
    )

    batch_result = BatchResult()

    for i, (doc, path) in enumerate(zip(batch, batch.paths)):
        if progress_callback:
            progress_callback(path.name, i + 1, batch.count)

        doc_result = DocumentResult(
            source_path=str(path),
            page_count=doc.page_count,
        )

        for page_idx in range(doc.page_count):
            page = doc.get_page(page_idx)
            result = extractor.extract(page, **extract_kwargs)
            doc_result.add_page_result(page_idx, result)

        batch_result.add_document_result(path.stem, doc_result)

        # Save individual result if output_dir specified
        if output_dir:
            out_path = Path(output_dir) / f"{path.stem}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            doc_result.save_json(str(out_path))

        doc.close()

    return batch_result
