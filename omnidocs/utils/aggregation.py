"""
Result aggregation utilities for batch processing.

Provides containers and utilities for storing, aggregating, and exporting
results from batch document processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentResult:
    """
    Container for results from processing a single document.

    Stores results by page for easy access and serialization.

    Examples:
        ```python
        doc_result = DocumentResult(source_path="paper.pdf", page_count=10)
        doc_result.add_page_result(0, text_output)
        doc_result.add_page_result(1, text_output)

        # Access results
        all_results = doc_result.all_results
        page_0_result = doc_result.get_page_result(0)

        # Save to file
        doc_result.save_json("paper_result.json")
        ```
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        page_count: int = 0,
    ):
        """
        Initialize DocumentResult.

        Args:
            source_path: Path to source document
            page_count: Total number of pages
        """
        self.source_path = source_path
        self.page_count = page_count
        self._page_results: Dict[int, Any] = {}

    def add_page_result(self, page_num: int, result: Any) -> None:
        """
        Add result for a specific page.

        Args:
            page_num: Page number (0-indexed)
            result: Extraction result (TextOutput, LayoutOutput, etc.)
        """
        self._page_results[page_num] = result

    def get_page_result(self, page_num: int) -> Optional[Any]:
        """
        Get result for a specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Result for the page, or None if not found
        """
        return self._page_results.get(page_num)

    @property
    def all_results(self) -> List[Any]:
        """
        Get all results in page order.

        Returns:
            List of results sorted by page number
        """
        return [self._page_results[i] for i in sorted(self._page_results.keys())]

    @property
    def processed_pages(self) -> int:
        """Number of pages with results."""
        return len(self._page_results)

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        results_dict = {}
        for k, v in self._page_results.items():
            if hasattr(v, "model_dump"):
                results_dict[str(k)] = v.model_dump()
            elif hasattr(v, "to_dict"):
                results_dict[str(k)] = v.to_dict()
            elif hasattr(v, "__dict__"):
                results_dict[str(k)] = v.__dict__
            else:
                results_dict[str(k)] = str(v)

        return {
            "source_path": self.source_path,
            "page_count": self.page_count,
            "processed_pages": self.processed_pages,
            "results": results_dict,
        }

    def save_json(self, path: str) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        return f"DocumentResult(source={self.source_path}, pages={self.processed_pages}/{self.page_count})"


class BatchResult:
    """
    Container for results from processing multiple documents.

    Examples:
        ```python
        batch_result = BatchResult()
        batch_result.add_document_result("doc1", doc_result1)
        batch_result.add_document_result("doc2", doc_result2)

        # Access results
        doc1_result = batch_result.get_document_result("doc1")
        all_ids = batch_result.document_ids

        # Save all results
        batch_result.save_json("all_results.json")
        ```
    """

    def __init__(self):
        """Initialize empty BatchResult."""
        self._document_results: Dict[str, DocumentResult] = {}

    def add_document_result(self, doc_id: str, result: DocumentResult) -> None:
        """
        Add result for a document.

        Args:
            doc_id: Document identifier (usually filename without extension)
            result: DocumentResult instance
        """
        self._document_results[doc_id] = result

    def get_document_result(self, doc_id: str) -> Optional[DocumentResult]:
        """
        Get result for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            DocumentResult or None if not found
        """
        return self._document_results.get(doc_id)

    @property
    def document_ids(self) -> List[str]:
        """List of document IDs."""
        return list(self._document_results.keys())

    @property
    def document_count(self) -> int:
        """Number of documents processed."""
        return len(self._document_results)

    @property
    def total_pages(self) -> int:
        """Total pages across all documents."""
        return sum(r.processed_pages for r in self._document_results.values())

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "document_count": self.document_count,
            "total_pages": self.total_pages,
            "documents": {doc_id: result.to_dict() for doc_id, result in self._document_results.items()},
        }

    def save_json(self, path: str) -> None:
        """
        Save all results to JSON file.

        Args:
            path: Output file path
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        return f"BatchResult(documents={self.document_count}, total_pages={self.total_pages})"

    def __len__(self) -> int:
        return self.document_count

    def __iter__(self):
        return iter(self._document_results.items())


def merge_text_results(results: List[Any], separator: str = "\n\n") -> str:
    """
    Merge multiple TextOutput results into single string.

    Args:
        results: List of TextOutput (or objects with .content attribute)
        separator: String to join pages (default: double newline)

    Returns:
        Combined content string

    Examples:
        ```python
        all_results = doc_result.all_results
        full_text = merge_text_results(all_results)
        full_text_with_dividers = merge_text_results(all_results, separator="\\n\\n---\\n\\n")
        ```
    """
    contents = []
    for r in results:
        if hasattr(r, "content") and r.content:
            contents.append(r.content)
        elif isinstance(r, str) and r:
            contents.append(r)
    return separator.join(contents)
