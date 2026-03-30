"""
TATR extractor — PyTorch/HuggingFace Transformers backend.

Uses TableTransformerForObjectDetection from the transformers library.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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
from omnidocs.utils.cache import get_model_cache_dir

from .config import TATRPyTorchConfig

if TYPE_CHECKING:
    from omnidocs.tasks.ocr_extraction.models import OCROutput

# TATR label index → semantic meaning
_TATR_LABELS = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return device


class TATRPyTorchExtractor(BaseTableExtractor):
    """
        TATR table structure extractor — PyTorch backend.

        Uses Microsoft's Table Transformer (TATR) via HuggingFace transformers.
        Detects rows, columns, column headers, spanning cells, and projected
        row headers as object detections, then reconstructs the cell grid.

        Example:
    ```python
            from omnidocs.tasks.table_extraction import TATRExtractor
            from omnidocs.tasks.table_extraction.tatr import TATRPyTorchConfig, TATRVariant

            extractor = TATRExtractor(backend=TATRPyTorchConfig(
                variant=TATRVariant.ALL,
                device="cuda",
            ))
            result = extractor.extract(table_image)
            df = result.to_dataframe()
    ```
    """

    def __init__(self, config: TATRPyTorchConfig):
        self.config = config
        self._device = _resolve_device(config.device)
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        cache_key = get_cache_key(self.config)
        self._cache_key = cache_key
        cached = get_cached(cache_key)
        if cached is not None:
            self._model, self._processor = cached
            add_reference(cache_key, self)
            return

        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        except ImportError:
            raise ImportError(
                "transformers is required for TATRPyTorchExtractor. Install with: pip install transformers"
            )

        cache_dir = str(get_model_cache_dir(self.config.cache_dir))

        # ---------------------------------------------------------------
        # Workaround: huggingface_hub >=0.24 introduced strict dataclass
        # validation. The TATR config.json has "dilation": null which
        # fails bool validation during cls(**config_dict). We patch
        # __strict_setattr__ to coerce None → False for bool fields
        # before ANY from_pretrained call fires the validator.
        # ---------------------------------------------------------------
        try:
            import huggingface_hub.dataclasses as _hf_dc

            _orig_validate_simple = _hf_dc._validate_simple_type

            def _patched_validate_simple(name, value, expected_type):
                if value is None:
                    return  # Allow None through — mirrors Optional[T] semantics
                _orig_validate_simple(name, value, expected_type)

            _hf_dc._validate_simple_type = _patched_validate_simple
        except Exception:
            pass

        self._processor = AutoImageProcessor.from_pretrained(
            self.config.repo_id,
            cache_dir=cache_dir,
        )

        # Workaround: TATR preprocessor_config.json has {"longest_edge": 800, "shortest_edge": null}.
        # transformers>=5.x DetrImageProcessor.resize() requires both to be non-None.
        # Set shortest_edge = longest_edge to preserve original aspect-ratio resize behaviour.
        proc_size = self._processor.size
        if isinstance(proc_size, dict):
            longest = proc_size.get("longest_edge")
            if longest is not None and proc_size.get("shortest_edge") is None:
                self._processor.size = {"shortest_edge": longest, "longest_edge": longest}
        else:
            # SizeDict object — set attributes directly
            if (
                getattr(proc_size, "longest_edge", None) is not None
                and getattr(proc_size, "shortest_edge", None) is None
            ):
                proc_size.shortest_edge = proc_size.longest_edge
        self._model = TableTransformerForObjectDetection.from_pretrained(
            self.config.repo_id,
            cache_dir=cache_dir,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        set_cached(cache_key, (self._model, self._processor), owner=self)

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        ocr_output: Optional["OCROutput"] = None,
    ) -> TableOutput:
        import torch

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.config.detection_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results["labels"].cpu().tolist()

        return _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=width,
            image_height=height,
            ocr_output=ocr_output,
            model_name=f"TATR-{self.config.variant.value}",
            label_map=self._model.config.id2label,
        )


def _detections_to_table_output(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    image_width: int,
    image_height: int,
    ocr_output: Optional["OCROutput"],
    model_name: str,
    label_map: dict,
) -> TableOutput:
    """
    Convert TATR object detections into a structured TableOutput.

    TATR detects rows and columns as separate bounding boxes.
    We reconstruct cells by finding all (row, col) intersections.
    """
    rows: List[Tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)
    cols: List[Tuple[float, float, float, float]] = []
    col_headers: List[bool] = []  # parallel to rows — is this row a col header?

    for box, score, label_idx in zip(boxes, scores, labels):
        label_str = label_map.get(label_idx, "")
        x1, y1, x2, y2 = box
        if label_str == "table row":
            rows.append((x1, y1, x2, y2))
        elif label_str == "table column":
            cols.append((x1, y1, x2, y2))
        elif label_str == "table column header":
            rows.append((x1, y1, x2, y2))
            col_headers.append(True)
        # spanning cells and projected row headers are noted but cell
        # reconstruction handles them via overlap logic below

    if not rows or not cols:
        return TableOutput(
            cells=[],
            num_rows=0,
            num_cols=0,
            image_width=image_width,
            image_height=image_height,
            model_name=model_name,
        )

    # Sort rows top-to-bottom, cols left-to-right
    rows.sort(key=lambda b: b[1])
    cols.sort(key=lambda b: b[0])

    # Build a set of column-header row y-ranges for fast lookup
    header_y_ranges = set()
    for box, score, label_idx in zip(boxes, scores, labels):
        label_str = label_map.get(label_idx, "")
        if label_str == "table column header":
            header_y_ranges.add((box[1], box[3]))

    cells: List[TableCell] = []

    # Build OCR word lookup if provided
    ocr_words = []
    if ocr_output:
        for block in ocr_output.text_blocks:
            ocr_words.append((block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2, block.text))

    for row_idx, row_box in enumerate(rows):
        rx1, ry1, rx2, ry2 = row_box
        is_header = any(abs(ry1 - hy1) < 5 and abs(ry2 - hy2) < 5 for hy1, hy2 in header_y_ranges)
        cell_type = CellType.COLUMN_HEADER if is_header else CellType.DATA

        for col_idx, col_box in enumerate(cols):
            cx1, cy1, cx2, cy2 = col_box

            # Cell bbox is the intersection of row and column
            cell_x1 = max(rx1, cx1)
            cell_y1 = max(ry1, cy1)
            cell_x2 = min(rx2, cx2)
            cell_y2 = min(ry2, cy2)

            if cell_x2 <= cell_x1 or cell_y2 <= cell_y1:
                continue

            bbox = BoundingBox(x1=cell_x1, y1=cell_y1, x2=cell_x2, y2=cell_y2)

            # Match OCR words whose center falls within this cell
            text = ""
            if ocr_words:
                matched = []
                for wx1, wy1, wx2, wy2, wtext in ocr_words:
                    wcy = (wy1 + wy2) / 2
                    wcx = (wx1 + wx2) / 2
                    if cell_x1 <= wcx <= cell_x2 and cell_y1 <= wcy <= cell_y2:
                        matched.append((wx1, wtext))
                matched.sort(key=lambda x: x[0])
                text = " ".join(t for _, t in matched)

            cells.append(
                TableCell(
                    row=row_idx,
                    col=col_idx,
                    row_span=1,
                    col_span=1,
                    text=text,
                    cell_type=cell_type,
                    bbox=bbox,
                )
            )

    return TableOutput(
        cells=cells,
        num_rows=len(rows),
        num_cols=len(cols),
        image_width=image_width,
        image_height=image_height,
        model_name=model_name,
    )
