"""
TableFormer Table Extraction - CPU

Standalone test script. Run locally on CPU.

Usage:
    python -m tests.standalone.table_extraction.tableformer_cpu path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class TableFormerCPUTest(BaseOmnidocsTest):
    """Test TableFormer with CPU."""

    @property
    def test_name(self) -> str:
        return "tableformer_cpu"

    @property
    def backend_name(self) -> str:
        return "pytorch_cpu"

    @property
    def task_name(self) -> str:
        return "table_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerExtractor

        return TableFormerExtractor(config=TableFormerConfig(device="cpu"))

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_cells": len(result.cells),
            "num_rows": result.num_rows,
            "num_cols": result.num_cols,
        }


if __name__ == "__main__":
    run_standalone_test(TableFormerCPUTest)
