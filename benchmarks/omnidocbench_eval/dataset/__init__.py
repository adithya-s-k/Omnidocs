from registry.registry import DATASET_REGISTRY

from .detection_dataset import DetectionDataset, DetectionDatasetSimpleFormat
from .end2end_dataset import End2EndDataset
from .md2md_dataset import Md2MdDataset
from .recog_dataset import (
    OmiDocBenchSingleModuleDataset as OmniDocBenchSingleModuleDataset,
)
from .recog_dataset import (
    RecognitionFormulaDataset,
    RecognitionTableDataset,
    RecognitionTextDataset,
)

__all__ = [
    "RecognitionFormulaDataset",
    "End2EndDataset",
    "DetectionDataset",
    "Md2MdDataset",
    "OmniDocBenchSingleModuleDataset",
    "DetectionDatasetSimpleFormat",
]

print("DATASET_REGISTRY: ", DATASET_REGISTRY.list_items())
