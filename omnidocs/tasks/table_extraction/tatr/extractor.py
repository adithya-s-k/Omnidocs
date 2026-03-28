"""
TATRExtractor — unified entry point that dispatches to the correct backend.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from PIL import Image

from omnidocs.tasks.table_extraction.base import BaseTableExtractor
from omnidocs.tasks.table_extraction.models import TableOutput

from .config import TATRONNXConfig, TATRPyTorchConfig
from .onnx import TATRONNXExtractor
from .pytorch import TATRPyTorchExtractor

if TYPE_CHECKING:
    from omnidocs.tasks.ocr_extraction.models import OCROutput

TATRBackendConfig = Union[TATRPyTorchConfig, TATRONNXConfig]


class TATRExtractor(BaseTableExtractor):
    """
        Table Transformer (TATR) table structure extractor.

        Supports three backends selectable via config type:
          - TATRPyTorchConfig  → PyTorch / HuggingFace transformers (CPU, CUDA, MPS)
          - TATRONNXConfig     → ONNX Runtime (CPU or CUDA, no PyTorch at inference)

        Three model variants:
          - TATRVariant.PUB  — scientific/academic documents (PubTables-1M)
          - TATRVariant.FIN  — financial documents (FinTabNet.c)
          - TATRVariant.ALL  — general purpose (both datasets, recommended default)

        Example:
    ```python
            from omnidocs.tasks.table_extraction import TATRExtractor
            from omnidocs.tasks.table_extraction.tatr import (
                TATRPyTorchConfig,
                TATRONNXConfig,
                TATRVariant,
            )

            # PyTorch (GPU)
            extractor = TATRExtractor(backend=TATRPyTorchConfig(device="cuda"))

            # ONNX (CPU, no PyTorch at inference)
            extractor = TATRExtractor(backend=TATRONNXConfig())

            result = extractor.extract(table_image)
            html = result.to_html()
            df = result.to_dataframe()
    ```
    """

    def __init__(self, backend: TATRBackendConfig):
        self.backend_config = backend
        self._impl: BaseTableExtractor = self._build_impl(backend)

    def _build_impl(self, backend: TATRBackendConfig) -> BaseTableExtractor:
        if isinstance(backend, TATRPyTorchConfig):
            return TATRPyTorchExtractor(backend)
        if isinstance(backend, TATRONNXConfig):
            return TATRONNXExtractor(backend)
        raise TypeError(f"Unknown TATR backend config: {type(backend)}")

    def _load_model(self) -> None:
        pass  # Handled in _build_impl

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        ocr_output: Optional["OCROutput"] = None,
    ) -> TableOutput:
        return self._impl.extract(image, ocr_output=ocr_output)

    def batch_extract(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        ocr_outputs=None,
        progress_callback=None,
    ) -> List[TableOutput]:
        return self._impl.batch_extract(images, ocr_outputs, progress_callback)
