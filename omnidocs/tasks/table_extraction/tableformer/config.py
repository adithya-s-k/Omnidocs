"""
Configuration for TableFormer table structure extractor.

TableFormer uses a dual-decoder transformer architecture with OTSL+ support
for recognizing table structure from images.

Example:
    ```python
    from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

    # Fast mode (default)
    extractor = TableFormerExtractor(config=TableFormerConfig())

    # Accurate mode with GPU
    extractor = TableFormerExtractor(
        config=TableFormerConfig(
            mode="accurate",
            device="cuda",
            do_cell_matching=True,
        )
    )
    ```
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TableFormerMode(str, Enum):
    """TableFormer inference mode."""

    FAST = "fast"  # Faster inference, slightly lower accuracy
    ACCURATE = "accurate"  # Higher accuracy, slower


class TableFormerConfig(BaseModel):
    """
    Configuration for TableFormer table structure extractor.

    TableFormer is a transformer-based model that predicts table structure
    using OTSL (Optimal Table Structure Language) tags and cell bounding boxes.

    Attributes:
        mode: Inference mode - "fast" or "accurate"
        device: Device for inference - "cpu", "cuda", "mps", or "auto"
        num_threads: Number of CPU threads for inference
        do_cell_matching: Whether to match predicted cells with OCR text cells
        artifacts_path: Path to pre-downloaded model artifacts
        repo_id: HuggingFace model repository
        revision: Model revision/tag

    Example:
        ```python
        from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

        # Fast mode
        extractor = TableFormerExtractor(config=TableFormerConfig(mode="fast"))

        # Accurate mode with GPU
        extractor = TableFormerExtractor(
            config=TableFormerConfig(
                mode="accurate",
                device="cuda",
                do_cell_matching=True,
            )
        )
        ```
    """

    model_config = ConfigDict(extra="forbid")

    mode: TableFormerMode = Field(
        default=TableFormerMode.FAST,
        description="Inference mode: 'fast' or 'accurate'",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for inference",
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        description="CPU threads for inference",
    )
    do_cell_matching: bool = Field(
        default=True,
        description="Match predicted cells with OCR text cells",
    )
    correct_overlapping_cells: bool = Field(
        default=False,
        description="Attempt to correct overlapping cell predictions",
    )
    sort_row_col_indexes: bool = Field(
        default=True,
        description="Sort cells by row and column indexes",
    )
    artifacts_path: Optional[str] = Field(
        default=None,
        description="Path to pre-downloaded model artifacts",
    )
    repo_id: str = Field(
        default="ds4sd/docling-models",
        description="HuggingFace model repository",
    )
    revision: str = Field(
        default="v2.1.0",
        description="Model revision/tag",
    )
