"""
Configuration for Table Transformer (TATR) table structure extractor.

TATR is a DETR-based object detection model from Microsoft that detects
table rows, columns, and cell bounding boxes from document images.

Example:
```python
    from omnidocs.tasks.table_extraction.tatr import TATRConfig, TATRVariant

    # PyTorch backend
    from omnidocs.tasks.table_extraction.tatr import TATRPyTorchConfig
    extractor = TATRExtractor(backend=TATRPyTorchConfig(variant=TATRVariant.ALL))

    # ONNX backend
    from omnidocs.tasks.table_extraction.tatr import TATRONNXConfig
    extractor = TATRExtractor(backend=TATRONNXConfig())
```
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TATRVariant(str, Enum):
    """
    TATR model variant.

    PUB  - Trained on PubTables-1M (scientific/academic documents)
    FIN  - Trained on FinTabNet.c (financial documents, dense numeric tables)
    ALL  - Trained on both PubTables-1M + FinTabNet.c (general purpose)
    """

    PUB = "pub"
    FIN = "fin"
    ALL = "all"


_VARIANT_REPO = {
    TATRVariant.PUB: "microsoft/table-transformer-structure-recognition",
    TATRVariant.FIN: "microsoft/table-transformer-structure-recognition-v1.1-fin",
    TATRVariant.ALL: "microsoft/table-transformer-structure-recognition-v1.1-all",
}


class TATRPyTorchConfig(BaseModel):
    """
        PyTorch/HuggingFace backend config for TATR.

        Example:
    ```python
            from omnidocs.tasks.table_extraction import TATRExtractor
            from omnidocs.tasks.table_extraction.tatr import TATRPyTorchConfig, TATRVariant

            extractor = TATRExtractor(backend=TATRPyTorchConfig(
                variant=TATRVariant.ALL,
                device="cuda",
            ))
            result = extractor.extract(table_image)
    ```
    """

    model_config = ConfigDict(extra="forbid")

    variant: TATRVariant = Field(
        default=TATRVariant.ALL,
        description="Model variant: 'pub' (scientific), 'fin' (financial), 'all' (general)",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for inference",
    )
    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )

    @property
    def repo_id(self) -> str:
        return _VARIANT_REPO[self.variant]


class TATRONNXConfig(BaseModel):
    """
        ONNX Runtime backend config for TATR.

        Runs without PyTorch. Uses onnxruntime (CPU) or onnxruntime-gpu (CUDA).
        The ONNX model is exported on first use and cached to disk.

        Example:
    ```python
            from omnidocs.tasks.table_extraction import TATRExtractor
            from omnidocs.tasks.table_extraction.tatr import TATRONNXConfig, TATRVariant

            extractor = TATRExtractor(backend=TATRONNXConfig(
                variant=TATRVariant.FIN,
                use_gpu=False,
            ))
            result = extractor.extract(table_image)
    ```
    """

    model_config = ConfigDict(extra="forbid")

    variant: TATRVariant = Field(
        default=TATRVariant.ALL,
        description="Model variant: 'pub' (scientific), 'fin' (financial), 'all' (general)",
    )
    use_gpu: bool = Field(
        default=False,
        description="Use CUDAExecutionProvider if True, else CPUExecutionProvider",
    )
    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache the exported ONNX model file. "
        "If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )

    @property
    def repo_id(self) -> str:
        return _VARIANT_REPO[self.variant]
