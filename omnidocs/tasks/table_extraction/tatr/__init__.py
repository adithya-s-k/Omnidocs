"""
TATR (Table Transformer) module for table structure extraction.

Provides Microsoft's Table Transformer with two backends: PyTorch and ONNX.
"""

from .config import TATRONNXConfig, TATRPyTorchConfig, TATRVariant
from .extractor import TATRExtractor

__all__ = [
    "TATRExtractor",
    "TATRPyTorchConfig",
    "TATRONNXConfig",
    "TATRVariant",
]
