"""
Integration tests for table extraction.

These tests run actual model inference and require appropriate backends.
Use pytest markers to select which tests to run based on available hardware.

Usage:
    pytest tests/integration/test_table_extractors.py -m cpu
    pytest tests/integration/test_table_extractors.py -m gpu
"""

import pytest

from tests.utils.evaluation import evaluate_table_extraction


class TestTableFormerExtractor:
    """Tests for TableFormer table extraction."""

    @pytest.mark.integration
    @pytest.mark.table_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_tableformer_cpu(self, table_image):
        """Test TableFormer on CPU."""
        from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

        extractor = TableFormerExtractor(config=TableFormerConfig(device="cpu"))
        result = extractor.extract(table_image["image"])

        assert result is not None
        assert hasattr(result, "cells")
        # Should detect table structure
        assert result.num_rows >= 0
        assert result.num_cols >= 0

    @pytest.mark.integration
    @pytest.mark.table_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_tableformer_gpu(self, table_image):
        """Test TableFormer on GPU."""
        from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

        extractor = TableFormerExtractor(config=TableFormerConfig(device="cuda"))
        result = extractor.extract(table_image["image"])

        assert result is not None
        assert hasattr(result, "cells")

    @pytest.mark.integration
    @pytest.mark.table_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_tableformer_output_formats(self, table_image):
        """Test TableFormer output formats."""
        from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

        extractor = TableFormerExtractor(config=TableFormerConfig(device="cpu"))
        result = extractor.extract(table_image["image"])

        # Test HTML output
        html = result.to_html()
        assert html is not None
        assert "<table" in html.lower() or len(result.cells) == 0

        # Test Markdown output
        md = result.to_markdown()
        assert md is not None

        # Test DataFrame output (if cells exist)
        if result.cells:
            df = result.to_dataframe()
            assert df is not None


class TestTableExtractionQuality:
    """Tests for table extraction quality evaluation."""

    @pytest.mark.integration
    @pytest.mark.table_extraction
    @pytest.mark.cpu
    def test_table_evaluation(self, table_image):
        """Test table extraction quality evaluation."""
        ground_truth = table_image["data"]

        # This test demonstrates how to evaluate table extraction quality
        eval_result = evaluate_table_extraction(
            extracted_cells=ground_truth,  # Perfect match for demo
            ground_truth_cells=ground_truth,
        )

        assert eval_result["cell_accuracy"] == 1.0
        assert eval_result["structure_match"] is True
