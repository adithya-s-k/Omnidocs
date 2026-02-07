"""
TableFormer extractor implementation using PyTorch backend.

Uses the TFPredictor from docling-ibm-models for table structure recognition.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from omnidocs.cache import add_reference, get_cache_key, get_cached, set_cached
from omnidocs.tasks.table_extraction.base import BaseTableExtractor
from omnidocs.tasks.table_extraction.models import (
    BoundingBox,
    CellType,
    TableCell,
    TableOutput,
)
from omnidocs.tasks.table_extraction.tableformer.config import (
    TableFormerConfig,
)
from omnidocs.utils.cache import get_model_cache_dir

if TYPE_CHECKING:
    from omnidocs.tasks.ocr_extraction.models import OCROutput


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return device


class TableFormerExtractor(BaseTableExtractor):
    """
    Table structure extractor using TableFormer model.

    TableFormer is a transformer-based model that predicts table structure
    using OTSL (Optimal Table Structure Language) tags. It can detect:
    - Cell boundaries (bounding boxes)
    - Row and column spans
    - Header cells (column and row headers)
    - Section rows

    Example:
        ```python
        from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

        # Initialize extractor
        extractor = TableFormerExtractor(
            config=TableFormerConfig(mode="fast", device="cuda")
        )

        # Extract table structure
        result = extractor.extract(table_image)

        # Get HTML output
        html = result.to_html()

        # Get DataFrame
        df = result.to_dataframe()
        ```
    """

    def __init__(self, config: TableFormerConfig):
        """
        Initialize TableFormer extractor.

        Args:
            config: TableFormerConfig with model settings
        """
        self.config = config
        self._device = _resolve_device(config.device)
        self._predictor = None
        self._model_config: Optional[Dict] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load TableFormer model.

        Uses unified model cache with reference counting to share models.
        """
        # Check cache first
        cache_key = get_cache_key(self.config)
        self._cache_key = cache_key
        cached = get_cached(cache_key)
        if cached is not None:
            self._predictor, self._model_config = cached
            add_reference(cache_key, self)
            return

        # Lazy import
        try:
            from docling_ibm_models.tableformer.data_management.tf_predictor import (
                TFPredictor,
            )
        except ImportError:
            raise ImportError(
                "docling-ibm-models is required for TableFormerExtractor. Install with: pip install docling-ibm-models"
            )

        # Get artifacts path
        if self.config.artifacts_path:
            artifacts_path = Path(self.config.artifacts_path)
        else:
            from huggingface_hub import snapshot_download

            cache_dir = get_model_cache_dir()
            download_path = snapshot_download(
                repo_id=self.config.repo_id,
                revision=self.config.revision,
                cache_dir=str(cache_dir),
            )
            mode_dir = self.config.mode.value
            artifacts_path = Path(download_path) / "model_artifacts" / "tableformer" / mode_dir

        # Build config for TFPredictor
        save_dir = str(artifacts_path)
        self._model_config = self._build_predictor_config(save_dir)

        # Initialize predictor
        self._predictor = TFPredictor(
            config=self._model_config,
            device=self._device,
            num_threads=self.config.num_threads,
        )

        # Cache the loaded model
        set_cached(cache_key, (self._predictor, self._model_config), owner=self)

    def _build_predictor_config(self, save_dir: str) -> Dict:
        """Build configuration dict for TFPredictor."""
        return {
            "dataset": {
                "type": "TF_prepared",
                "name": "TF",
                "load_cells": True,
                "bbox_format": "5plet",
                "resized_image": 448,
                "keep_AR": False,
                "up_scaling_enabled": True,
                "down_scaling_enabled": True,
                "padding_mode": "null",
                "padding_color": [0, 0, 0],
                "image_normalization": {
                    "state": True,
                    "mean": [0.94247851, 0.94254675, 0.94292611],
                    "std": [0.17910956, 0.17940403, 0.17931663],
                },
                "color_jitter": True,
                "rand_crop": True,
                "rand_pad": True,
                "image_grayscale": False,
            },
            "model": {
                "type": "TableModel04_rs",
                "name": "14_128_256_4_true",
                "save_dir": save_dir,
                "backbone": "resnet18",
                "enc_image_size": 28,
                "tag_embed_dim": 16,
                "hidden_dim": 512,
                "tag_decoder_dim": 512,
                "bbox_embed_dim": 256,
                "tag_attention_dim": 256,
                "bbox_attention_dim": 512,
                "enc_layers": 4,
                "dec_layers": 2,
                "nheads": 8,
                "dropout": 0.1,
                "bbox_classes": 2,
            },
            "train": {
                "disable_cuda": self._device == "cpu",
                "batch_size": 1,
                "bbox": True,
            },
            "predict": {
                "max_steps": 1024,
                "beam_size": 5,
                "bbox": True,
                "pdf_cell_iou_thres": 0.05,
                "padding": False,
                "padding_size": 50,
                "disable_post_process": False,
                "profiling": False,
            },
            "dataset_wordmap": {
                "word_map_tag": {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<start>": 2,
                    "<end>": 3,
                    "ecel": 4,
                    "fcel": 5,
                    "lcel": 6,
                    "ucel": 7,
                    "xcel": 8,
                    "nl": 9,
                    "ched": 10,
                    "rhed": 11,
                    "srow": 12,
                },
                "word_map_cell": {"<pad>": 0, "<unk>": 1},
            },
        }

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        ocr_output: Optional["OCROutput"] = None,
    ) -> TableOutput:
        """
        Extract table structure from an image.

        Args:
            image: Table image (should be cropped to table region)
            ocr_output: Optional OCR results for cell text matching

        Returns:
            TableOutput with cells, structure, and export methods

        Example:
            ```python
            result = extractor.extract(table_image)
            print(f"Table: {result.num_rows}x{result.num_cols}")
            html = result.to_html()
            ```
        """
        # Prepare image
        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Convert to OpenCV format (required by TFPredictor)
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for TableFormerExtractor. Install with: pip install opencv-python-headless"
            )

        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Build iOCR page data
        tokens = self._build_tokens_from_ocr(ocr_output) if ocr_output else []
        iocr_page = {
            "width": width,
            "height": height,
            "image": cv_image,
            "tokens": tokens,
        }

        # Table bbox is the entire image
        table_bbox = [0, 0, width, height]

        # Run prediction
        results = self._predictor.multi_table_predict(
            iocr_page=iocr_page,
            table_bboxes=[table_bbox],
            do_matching=self.config.do_cell_matching,
            correct_overlapping_cells=self.config.correct_overlapping_cells,
            sort_row_col_indexes=self.config.sort_row_col_indexes,
        )

        # Convert results to TableOutput
        return self._convert_results(results, width, height)

    def _build_tokens_from_ocr(self, ocr_output: "OCROutput") -> List[Dict]:
        """Convert OCROutput to token format expected by TFPredictor."""
        tokens = []
        for i, block in enumerate(ocr_output.text_blocks):
            token = {
                "id": i,
                "text": block.text,
                "bbox": {
                    "l": block.bbox.x1,
                    "t": block.bbox.y1,
                    "r": block.bbox.x2,
                    "b": block.bbox.y2,
                },
            }
            tokens.append(token)
        return tokens

    def _convert_results(self, results: List[Dict], image_width: int, image_height: int) -> TableOutput:
        """Convert TFPredictor results to TableOutput."""
        if not results:
            return TableOutput(
                cells=[],
                num_rows=0,
                num_cols=0,
                image_width=image_width,
                image_height=image_height,
                model_name="TableFormer",
            )

        # Process first table result
        table_result = results[0]
        tf_responses = table_result.get("tf_responses", [])

        # Convert cells
        cells = []
        max_row = 0
        max_col = 0

        for cell_data in tf_responses:
            row = cell_data.get("start_row_offset_idx", 0)
            col = cell_data.get("start_col_offset_idx", 0)
            row_span = cell_data.get("row_span", 1)
            col_span = cell_data.get("col_span", 1)

            # Update max row/col
            max_row = max(max_row, row + row_span)
            max_col = max(max_col, col + col_span)

            # Determine cell type
            if cell_data.get("column_header", False):
                cell_type = CellType.COLUMN_HEADER
            elif cell_data.get("row_header", False):
                cell_type = CellType.ROW_HEADER
            elif cell_data.get("row_section", False):
                cell_type = CellType.SECTION_ROW
            else:
                cell_type = CellType.DATA

            # Extract bbox
            bbox_data = cell_data.get("bbox", {})
            bbox = None
            if bbox_data:
                bbox = BoundingBox(
                    x1=bbox_data.get("l", 0),
                    y1=bbox_data.get("t", 0),
                    x2=bbox_data.get("r", 0),
                    y2=bbox_data.get("b", 0),
                )

            # Extract text from matched OCR cells
            text = ""
            text_cell_bboxes = cell_data.get("text_cell_bboxes", [])
            if text_cell_bboxes:
                texts = [tcb.get("text", "") for tcb in text_cell_bboxes if tcb.get("text")]
                text = " ".join(texts)

            cell = TableCell(
                row=row,
                col=col,
                row_span=row_span,
                col_span=col_span,
                text=text,
                cell_type=cell_type,
                bbox=bbox,
            )
            cells.append(cell)

        return TableOutput(
            cells=cells,
            num_rows=max_row,
            num_cols=max_col,
            image_width=image_width,
            image_height=image_height,
            model_name="TableFormer",
        )
