"""
Tests for batch processing utilities.

Tests DocumentBatch, process_document, process_directory, and result aggregation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from omnidocs import Document, DocumentBatch
from omnidocs.batch import process_directory, process_document
from omnidocs.utils.aggregation import BatchResult, DocumentResult, merge_text_results

# ============= Fixtures =============


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def pdf_dir(fixtures_dir):
    """Path to PDF fixtures directory."""
    return fixtures_dir / "pdfs"


@pytest.fixture
def sample_pdf(pdf_dir):
    """Path to a sample PDF."""
    return pdf_dir / "bank_statement_1.pdf"


@pytest.fixture
def mock_extractor():
    """Mock extractor that returns dummy results."""
    extractor = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Extracted text content"
    mock_result.model_dump.return_value = {"content": "Extracted text content"}
    extractor.extract.return_value = mock_result
    return extractor


@pytest.fixture
def mock_text_output():
    """Mock TextOutput-like object."""
    output = MagicMock()
    output.content = "Page content"
    output.model_dump.return_value = {"content": "Page content", "format": "markdown"}
    return output


# ============= DocumentBatch Tests =============


class TestDocumentBatch:
    """Tests for DocumentBatch class."""

    def test_from_directory(self, pdf_dir):
        """Test loading batch from directory."""
        batch = DocumentBatch.from_directory(str(pdf_dir))

        assert batch.count > 0
        assert len(batch) == batch.count
        assert all(p.suffix == ".pdf" for p in batch.paths)

    def test_from_directory_not_found(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DocumentBatch.from_directory("/nonexistent/directory")

    def test_from_directory_with_pattern(self, pdf_dir):
        """Test loading with specific pattern."""
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_*.pdf")

        assert batch.count == 2  # bank_statement_1.pdf and bank_statement_2.pdf

    def test_from_paths(self, pdf_dir):
        """Test loading from explicit paths."""
        paths = [
            str(pdf_dir / "bank_statement_1.pdf"),
            str(pdf_dir / "bank_statement_2.pdf"),
        ]
        batch = DocumentBatch.from_paths(paths)

        assert batch.count == 2

    def test_iteration(self, pdf_dir):
        """Test iterating over batch."""
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_statement_1.pdf")

        docs = list(batch)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_count > 0

    def test_iter_with_progress(self, pdf_dir):
        """Test iteration with progress callback."""
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_*.pdf")

        progress_calls = []

        def callback(current, total, filename):
            progress_calls.append((current, total, filename))

        docs = list(batch.iter_with_progress(callback))

        assert len(docs) == 2
        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 1
        assert progress_calls[1][0] == 2
        assert progress_calls[0][1] == 2  # total

    def test_iter_all_pages(self, pdf_dir):
        """Test iterating over all pages."""
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_statement_1.pdf")

        pages = list(batch.iter_all_pages())

        assert len(pages) > 0
        # Each item is (doc_idx, page_idx, page_img, path)
        doc_idx, page_idx, page_img, path = pages[0]
        assert doc_idx == 0
        assert page_idx == 0
        assert isinstance(page_img, Image.Image)
        assert isinstance(path, Path)

    def test_with_page_range(self, pdf_dir):
        """Test batch with page range."""
        batch = DocumentBatch.from_directory(
            str(pdf_dir),
            pattern="research_paper_1.pdf",
            page_range=(0, 1),  # First 2 pages only
        )

        for doc in batch:
            assert doc.page_count == 2

    def test_with_custom_dpi(self, pdf_dir):
        """Test batch with custom DPI."""
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_statement_1.pdf", dpi=300)

        for doc in batch:
            # Higher DPI means larger images
            page = doc.get_page(0)
            assert page.size[0] > 0


# ============= DocumentResult Tests =============


class TestDocumentResult:
    """Tests for DocumentResult class."""

    def test_create_and_add_results(self, mock_text_output):
        """Test creating DocumentResult and adding page results."""
        result = DocumentResult(source_path="test.pdf", page_count=3)

        result.add_page_result(0, mock_text_output)
        result.add_page_result(1, mock_text_output)

        assert result.processed_pages == 2
        assert result.page_count == 3
        assert result.get_page_result(0) == mock_text_output
        assert result.get_page_result(2) is None

    def test_all_results(self, mock_text_output):
        """Test getting all results in order."""
        result = DocumentResult(source_path="test.pdf", page_count=3)

        result.add_page_result(2, mock_text_output)
        result.add_page_result(0, mock_text_output)
        result.add_page_result(1, mock_text_output)

        all_results = result.all_results
        assert len(all_results) == 3

    def test_to_dict(self, mock_text_output):
        """Test serialization to dict."""
        result = DocumentResult(source_path="test.pdf", page_count=2)
        result.add_page_result(0, mock_text_output)

        d = result.to_dict()

        assert d["source_path"] == "test.pdf"
        assert d["page_count"] == 2
        assert d["processed_pages"] == 1
        assert "0" in d["results"]

    def test_save_json(self, mock_text_output):
        """Test saving to JSON file."""
        result = DocumentResult(source_path="test.pdf", page_count=1)
        result.add_page_result(0, mock_text_output)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save_json(str(path))

            assert path.exists()

            with open(path) as f:
                data = json.load(f)
                assert data["source_path"] == "test.pdf"

    def test_repr(self):
        """Test string representation."""
        result = DocumentResult(source_path="test.pdf", page_count=10)
        repr_str = repr(result)

        assert "test.pdf" in repr_str
        assert "0/10" in repr_str


# ============= BatchResult Tests =============


class TestBatchResult:
    """Tests for BatchResult class."""

    def test_create_and_add_documents(self):
        """Test creating BatchResult and adding documents."""
        batch_result = BatchResult()

        doc_result1 = DocumentResult(source_path="doc1.pdf", page_count=5)
        doc_result2 = DocumentResult(source_path="doc2.pdf", page_count=3)

        batch_result.add_document_result("doc1", doc_result1)
        batch_result.add_document_result("doc2", doc_result2)

        assert batch_result.document_count == 2
        assert len(batch_result) == 2
        assert "doc1" in batch_result.document_ids
        assert "doc2" in batch_result.document_ids

    def test_get_document_result(self):
        """Test retrieving document results."""
        batch_result = BatchResult()
        doc_result = DocumentResult(source_path="test.pdf", page_count=1)
        batch_result.add_document_result("test", doc_result)

        retrieved = batch_result.get_document_result("test")
        assert retrieved == doc_result

        assert batch_result.get_document_result("nonexistent") is None

    def test_total_pages(self, mock_text_output):
        """Test total pages calculation."""
        batch_result = BatchResult()

        doc1 = DocumentResult(source_path="doc1.pdf", page_count=5)
        doc1.add_page_result(0, mock_text_output)
        doc1.add_page_result(1, mock_text_output)

        doc2 = DocumentResult(source_path="doc2.pdf", page_count=3)
        doc2.add_page_result(0, mock_text_output)

        batch_result.add_document_result("doc1", doc1)
        batch_result.add_document_result("doc2", doc2)

        assert batch_result.total_pages == 3  # 2 + 1 processed pages

    def test_iteration(self):
        """Test iterating over batch result."""
        batch_result = BatchResult()

        doc1 = DocumentResult(source_path="doc1.pdf", page_count=1)
        doc2 = DocumentResult(source_path="doc2.pdf", page_count=1)

        batch_result.add_document_result("doc1", doc1)
        batch_result.add_document_result("doc2", doc2)

        items = list(batch_result)
        assert len(items) == 2
        assert items[0][0] == "doc1"

    def test_save_json(self, mock_text_output):
        """Test saving batch result to JSON."""
        batch_result = BatchResult()

        doc = DocumentResult(source_path="test.pdf", page_count=1)
        doc.add_page_result(0, mock_text_output)
        batch_result.add_document_result("test", doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch_result.json"
            batch_result.save_json(str(path))

            assert path.exists()

            with open(path) as f:
                data = json.load(f)
                assert data["document_count"] == 1
                assert "test" in data["documents"]


# ============= merge_text_results Tests =============


class TestMergeTextResults:
    """Tests for merge_text_results function."""

    def test_merge_with_content_attribute(self):
        """Test merging objects with .content attribute."""
        mock1 = MagicMock()
        mock1.content = "Page 1"
        mock2 = MagicMock()
        mock2.content = "Page 2"

        result = merge_text_results([mock1, mock2])

        assert result == "Page 1\n\nPage 2"

    def test_merge_with_custom_separator(self):
        """Test merging with custom separator."""
        mock1 = MagicMock()
        mock1.content = "Page 1"
        mock2 = MagicMock()
        mock2.content = "Page 2"

        result = merge_text_results([mock1, mock2], separator="\n---\n")

        assert result == "Page 1\n---\nPage 2"

    def test_merge_strings(self):
        """Test merging plain strings."""
        result = merge_text_results(["Page 1", "Page 2"])

        assert result == "Page 1\n\nPage 2"

    def test_merge_empty_content(self):
        """Test handling empty content."""
        mock1 = MagicMock()
        mock1.content = "Page 1"
        mock2 = MagicMock()
        mock2.content = ""  # Empty
        mock3 = MagicMock()
        mock3.content = "Page 3"

        result = merge_text_results([mock1, mock2, mock3])

        assert result == "Page 1\n\nPage 3"  # Empty page skipped


# ============= process_document Tests =============


class TestProcessDocument:
    """Tests for process_document function."""

    def test_process_document(self, sample_pdf, mock_extractor):
        """Test processing a document."""
        doc = Document.from_pdf(str(sample_pdf))

        result = process_document(doc, mock_extractor, output_format="markdown")

        assert isinstance(result, DocumentResult)
        assert result.processed_pages == doc.page_count
        assert mock_extractor.extract.call_count == doc.page_count

    def test_process_document_with_progress(self, sample_pdf, mock_extractor):
        """Test processing with progress callback."""
        doc = Document.from_pdf(str(sample_pdf))

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        process_document(doc, mock_extractor, progress_callback=callback)

        assert len(progress_calls) == doc.page_count
        assert progress_calls[-1][0] == doc.page_count


# ============= process_directory Tests =============


class TestProcessDirectory:
    """Tests for process_directory function."""

    def test_process_directory(self, pdf_dir, mock_extractor):
        """Test processing a directory."""
        result = process_directory(
            str(pdf_dir),
            mock_extractor,
            pattern="bank_*.pdf",
        )

        assert isinstance(result, BatchResult)
        assert result.document_count == 2

    def test_process_directory_with_output(self, pdf_dir, mock_extractor):
        """Test processing with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            process_directory(
                str(pdf_dir),
                mock_extractor,
                output_dir=tmpdir,
                pattern="bank_statement_1.pdf",
            )

            # Check output file was created
            output_files = list(Path(tmpdir).glob("*.json"))
            assert len(output_files) == 1
            assert output_files[0].stem == "bank_statement_1"

    def test_process_directory_with_progress(self, pdf_dir, mock_extractor):
        """Test processing with progress callback."""
        progress_calls = []

        def callback(filename, current, total):
            progress_calls.append((filename, current, total))

        process_directory(
            str(pdf_dir),
            mock_extractor,
            pattern="bank_*.pdf",
            progress_callback=callback,
        )

        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1  # current
        assert progress_calls[0][2] == 2  # total


# ============= Integration Tests =============


class TestBatchIntegration:
    """Integration tests for batch processing."""

    def test_full_batch_workflow(self, pdf_dir, mock_extractor):
        """Test complete batch processing workflow."""
        # Load batch
        batch = DocumentBatch.from_directory(str(pdf_dir), pattern="bank_statement_1.pdf")

        # Process with results
        batch_result = BatchResult()

        for doc in batch:
            doc_result = DocumentResult(
                source_path=str(doc.metadata.source_path),
                page_count=doc.page_count,
            )

            for page_idx in range(doc.page_count):
                page = doc.get_page(page_idx)
                result = mock_extractor.extract(page)
                doc_result.add_page_result(page_idx, result)

            batch_result.add_document_result(Path(doc.metadata.source_path).stem, doc_result)
            doc.close()

        # Verify results
        assert batch_result.document_count == 1
        assert batch_result.total_pages > 0

        # Serialize
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_result.save_json(Path(tmpdir) / "results.json")
            assert (Path(tmpdir) / "results.json").exists()

    def test_import_from_omnidocs(self):
        """Test that batch utilities are importable from omnidocs."""
        from omnidocs import (
            BatchResult,
            DocumentBatch,
            DocumentResult,
            merge_text_results,
            process_directory,
            process_document,
        )

        assert DocumentBatch is not None
        assert process_directory is not None
        assert process_document is not None
        assert BatchResult is not None
        assert DocumentResult is not None
        assert merge_text_results is not None
