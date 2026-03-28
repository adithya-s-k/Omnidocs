"""
Tests for TATR (Table Transformer) extractor configuration and internals.

Covers:
  - TATRVariant enum
  - TATRPyTorchConfig
  - TATRONNXConfig
  - TATRExtractor backend dispatch
  - _detections_to_table_output grid reconstruction logic
  - ONNX preprocessing helpers
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from omnidocs.tasks.table_extraction.models import CellType
from omnidocs.tasks.table_extraction.tatr import (
    TATRExtractor,
    TATRONNXConfig,
    TATRPyTorchConfig,
    TATRVariant,
)
from omnidocs.tasks.table_extraction.tatr.onnx import (
    _box_cxcywh_to_xyxy,
    _preprocess,
    _sigmoid,
)
from omnidocs.tasks.table_extraction.tatr.pytorch import (
    _detections_to_table_output,
    _resolve_device,
)

# ===========================================================================
# TATRVariant
# ===========================================================================


class TestTATRVariant:
    """Tests for TATRVariant enum."""

    def test_variant_values(self):
        """Test that TATRVariant has expected values."""
        assert TATRVariant.PUB.value == "pub"
        assert TATRVariant.FIN.value == "fin"
        assert TATRVariant.ALL.value == "all"

    def test_variant_is_string_enum(self):
        """Test that TATRVariant can be used as string."""
        assert str(TATRVariant.ALL.value) == "all"

    def test_variant_from_string(self):
        """Test constructing variant from plain string."""
        assert TATRVariant("pub") == TATRVariant.PUB
        assert TATRVariant("fin") == TATRVariant.FIN
        assert TATRVariant("all") == TATRVariant.ALL


# ===========================================================================
# TATRPyTorchConfig
# ===========================================================================


class TestTATRPyTorchConfig:
    """Tests for TATRPyTorchConfig model."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = TATRPyTorchConfig()
        assert config.variant == TATRVariant.ALL
        assert config.device == "auto"
        assert config.detection_threshold == 0.5
        assert config.cache_dir is None

    def test_repo_id_pub(self):
        """Test repo_id for PUB variant."""
        config = TATRPyTorchConfig(variant=TATRVariant.PUB)
        assert config.repo_id == "microsoft/table-transformer-structure-recognition"

    def test_repo_id_fin(self):
        """Test repo_id for FIN variant."""
        config = TATRPyTorchConfig(variant=TATRVariant.FIN)
        assert config.repo_id == "microsoft/table-transformer-structure-recognition-v1.1-fin"

    def test_repo_id_all(self):
        """Test repo_id for ALL variant."""
        config = TATRPyTorchConfig(variant=TATRVariant.ALL)
        assert config.repo_id == "microsoft/table-transformer-structure-recognition-v1.1-all"

    def test_valid_devices(self):
        """Test that all allowed device values are accepted."""
        for device in ["cpu", "cuda", "mps", "auto"]:
            config = TATRPyTorchConfig(device=device)
            assert config.device == device

    def test_invalid_device_raises(self):
        """Test that an invalid device raises a ValidationError."""
        with pytest.raises(ValidationError):
            TATRPyTorchConfig(device="tpu")

    def test_detection_threshold_bounds(self):
        """Test detection_threshold must be in [0, 1]."""
        TATRPyTorchConfig(detection_threshold=0.0)
        TATRPyTorchConfig(detection_threshold=1.0)

        with pytest.raises(ValidationError):
            TATRPyTorchConfig(detection_threshold=-0.1)

        with pytest.raises(ValidationError):
            TATRPyTorchConfig(detection_threshold=1.1)

    def test_cache_dir_optional(self):
        """Test that cache_dir can be set or left as None."""
        config = TATRPyTorchConfig(cache_dir="/tmp/models")
        assert config.cache_dir == "/tmp/models"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            TATRPyTorchConfig(unknown_param="x")

    def test_variant_from_string(self):
        """Test that variant can be supplied as a plain string."""
        config = TATRPyTorchConfig(variant="pub")
        assert config.variant == TATRVariant.PUB

    def test_full_config(self):
        """Test creating config with all fields specified."""
        config = TATRPyTorchConfig(
            variant=TATRVariant.FIN,
            device="cuda",
            detection_threshold=0.7,
            cache_dir="/custom/cache",
        )
        assert config.variant == TATRVariant.FIN
        assert config.device == "cuda"
        assert config.detection_threshold == 0.7
        assert config.cache_dir == "/custom/cache"


# ===========================================================================
# TATRONNXConfig
# ===========================================================================


class TestTATRONNXConfig:
    """Tests for TATRONNXConfig model."""

    def test_default_config(self):
        """Test creating ONNX config with defaults."""
        config = TATRONNXConfig()
        assert config.variant == TATRVariant.ALL
        assert config.use_gpu is False
        assert config.detection_threshold == 0.5
        assert config.cache_dir is None

    def test_repo_id_variants(self):
        """Test repo_id property for all variants."""
        assert "structure-recognition" in TATRONNXConfig(variant=TATRVariant.PUB).repo_id
        assert "fin" in TATRONNXConfig(variant=TATRVariant.FIN).repo_id
        assert "all" in TATRONNXConfig(variant=TATRVariant.ALL).repo_id

    def test_use_gpu_flag(self):
        """Test toggling use_gpu."""
        assert TATRONNXConfig(use_gpu=True).use_gpu is True
        assert TATRONNXConfig(use_gpu=False).use_gpu is False

    def test_detection_threshold_bounds(self):
        """Test detection_threshold must be in [0, 1]."""
        TATRONNXConfig(detection_threshold=0.0)
        TATRONNXConfig(detection_threshold=1.0)

        with pytest.raises(ValidationError):
            TATRONNXConfig(detection_threshold=1.5)

        with pytest.raises(ValidationError):
            TATRONNXConfig(detection_threshold=-0.5)

    def test_cache_dir(self):
        """Test cache_dir field."""
        config = TATRONNXConfig(cache_dir="/onnx/cache")
        assert config.cache_dir == "/onnx/cache"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            TATRONNXConfig(does_not_exist=True)

    def test_full_config(self):
        """Test full ONNX config construction."""
        config = TATRONNXConfig(
            variant=TATRVariant.PUB,
            use_gpu=True,
            detection_threshold=0.3,
            cache_dir="/some/path",
        )
        assert config.variant == TATRVariant.PUB
        assert config.use_gpu is True
        assert config.detection_threshold == 0.3
        assert config.cache_dir == "/some/path"


# ===========================================================================
# TATRExtractor — backend dispatch
# ===========================================================================


class TestTATRExtractorDispatch:
    """Tests for TATRExtractor backend selection."""

    def test_pytorch_backend_selected(self):
        """Test that TATRPyTorchConfig routes to TATRPyTorchExtractor."""
        from omnidocs.tasks.table_extraction.tatr.pytorch import TATRPyTorchExtractor

        config = TATRPyTorchConfig()
        with patch.object(TATRPyTorchExtractor, "_load_model", return_value=None):
            extractor = TATRExtractor(backend=config)
            assert isinstance(extractor._impl, TATRPyTorchExtractor)

    def test_onnx_backend_selected(self):
        """Test that TATRONNXConfig routes to TATRONNXExtractor."""
        from omnidocs.tasks.table_extraction.tatr.onnx import TATRONNXExtractor

        config = TATRONNXConfig()
        with patch.object(TATRONNXExtractor, "_load_model", return_value=None):
            extractor = TATRExtractor(backend=config)
            assert isinstance(extractor._impl, TATRONNXExtractor)

    def test_unknown_backend_raises(self):
        """Test that an unrecognised backend type raises TypeError."""
        from omnidocs.tasks.table_extraction.tatr.pytorch import TATRPyTorchExtractor

        config = TATRPyTorchConfig()
        with patch.object(TATRPyTorchExtractor, "_load_model", return_value=None):
            ext = TATRExtractor(backend=config)
            with pytest.raises(TypeError, match="Unknown TATR backend"):
                ext._build_impl("not-a-config")

    def test_backend_config_stored(self):
        """Test that backend_config is stored on the extractor."""
        from omnidocs.tasks.table_extraction.tatr.pytorch import TATRPyTorchExtractor

        config = TATRPyTorchConfig(variant=TATRVariant.FIN)
        with patch.object(TATRPyTorchExtractor, "_load_model", return_value=None):
            extractor = TATRExtractor(backend=config)
            assert extractor.backend_config is config


# ===========================================================================
# _resolve_device helper
# ===========================================================================


class TestResolveDevice:
    """Tests for _resolve_device helper function."""

    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_explicit_mps(self):
        assert _resolve_device("mps") == "mps"

    def test_auto_falls_back_to_cpu_when_torch_absent(self):
        """When torch is not importable, auto should resolve to cpu."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _resolve_device("auto")
        assert result == "cpu"

    def test_auto_resolves_to_cuda_when_available(self):
        """When CUDA is available, auto should resolve to cuda."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_device("auto")
        assert result == "cuda"

    def test_auto_resolves_to_mps_when_cuda_unavailable(self):
        """When CUDA is unavailable but MPS is, auto should resolve to mps."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_device("auto")
        assert result == "mps"


# ===========================================================================
# _detections_to_table_output (unit — no model weights)
# ===========================================================================

_LABEL_MAP = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}


class TestDetectionsToTableOutput:
    """Tests for the TATR grid reconstruction helper."""

    def _boxes_scores_labels(self, rows, cols, header_rows=None):
        """Build raw detection lists for given row/col bboxes."""
        boxes, scores, labels = [], [], []
        for r in rows:
            boxes.append(list(r))
            scores.append(0.9)
            labels.append(2)  # "table row"
        for c in cols:
            boxes.append(list(c))
            scores.append(0.9)
            labels.append(1)  # "table column"
        if header_rows:
            for hr in header_rows:
                boxes.append(list(hr))
                scores.append(0.9)
                labels.append(3)  # "table column header"
        return boxes, scores, labels

    def test_empty_detections_returns_empty_table(self):
        """No rows or columns → empty TableOutput."""
        result = _detections_to_table_output(
            boxes=[],
            scores=[],
            labels=[],
            image_width=800,
            image_height=600,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 0
        assert result.num_cols == 0
        assert result.cell_count == 0

    def test_no_rows_returns_empty_table(self):
        """Columns but no rows → empty TableOutput."""
        boxes = [[0, 0, 100, 200]]
        scores = [0.9]
        labels = [1]  # column only
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=800,
            image_height=600,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 0
        assert result.num_cols == 0

    def test_no_cols_returns_empty_table(self):
        """Rows but no columns → empty TableOutput."""
        boxes = [[0, 0, 800, 50]]
        scores = [0.9]
        labels = [2]  # row only
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=800,
            image_height=600,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 0
        assert result.num_cols == 0

    def test_2x2_grid_produces_4_cells(self):
        """Two rows × two columns → 4 cells."""
        rows = [(0, 0, 200, 50), (0, 50, 200, 100)]
        cols = [(0, 0, 100, 100), (100, 0, 200, 100)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=100,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 2
        assert result.num_cols == 2
        assert result.cell_count == 4

    def test_3x3_grid_cell_indices_correct(self):
        """Verify row/col indices are assigned correctly for a 3×3 grid."""
        rows = [(0, 0, 300, 50), (0, 50, 300, 100), (0, 100, 300, 150)]
        cols = [(0, 0, 100, 150), (100, 0, 200, 150), (200, 0, 300, 150)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=300,
            image_height=150,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 3
        assert result.num_cols == 3
        assert result.cell_count == 9

        top_left = result.get_cell(0, 0)
        assert top_left is not None
        assert top_left.row == 0 and top_left.col == 0

        bottom_right = result.get_cell(2, 2)
        assert bottom_right is not None
        assert bottom_right.row == 2 and bottom_right.col == 2

    def test_column_header_row_sets_cell_type(self):
        """Rows detected as column headers should produce COLUMN_HEADER cells."""
        boxes = [
            [0, 0, 200, 40],  # label 3 → column header
            [0, 40, 200, 80],  # label 2 → regular row
            [0, 0, 200, 80],  # label 1 → column
        ]
        scores = [0.9, 0.9, 0.9]
        labels = [3, 2, 1]

        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=80,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        header_cells = [c for c in result.cells if c.cell_type == CellType.COLUMN_HEADER]
        assert len(header_cells) > 0

    def test_model_name_stored(self):
        """Verify model_name is propagated to TableOutput."""
        rows = [(0, 0, 100, 50)]
        cols = [(0, 0, 100, 50)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=100,
            image_height=50,
            ocr_output=None,
            model_name="TATR-all",
            label_map=_LABEL_MAP,
        )
        assert result.model_name == "TATR-all"

    def test_image_dimensions_stored(self):
        """Verify image dimensions are propagated to TableOutput."""
        rows = [(0, 0, 640, 50)]
        cols = [(0, 0, 640, 50)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=640,
            image_height=480,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.image_width == 640
        assert result.image_height == 480

    def test_non_overlapping_row_col_skipped(self):
        """A row/col pair that doesn't overlap should not produce a cell."""
        row = (100, 0, 200, 50)  # x: 100–200
        col = (0, 0, 50, 50)  # x: 0–50 — no overlap
        boxes, scores, labels = self._boxes_scores_labels([row], [col])
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=50,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.cell_count == 0

    def test_ocr_text_matched_to_cell(self):
        """Words whose centres fall inside a cell are assigned to it."""
        from omnidocs.tasks.table_extraction.models import BoundingBox

        row = (0, 0, 200, 50)
        col = (0, 0, 200, 50)
        boxes, scores, labels = self._boxes_scores_labels([row], [col])

        word_bbox = BoundingBox(x1=80, y1=15, x2=120, y2=35)
        block = MagicMock()
        block.bbox = word_bbox
        block.text = "Hello"

        ocr_output = MagicMock()
        ocr_output.text_blocks = [block]

        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=50,
            ocr_output=ocr_output,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.cell_count == 1
        assert result.cells[0].text == "Hello"

    def test_rows_sorted_top_to_bottom(self):
        """Rows should be sorted by y1 after grid reconstruction."""
        rows = [(0, 100, 200, 150), (0, 0, 200, 50), (0, 50, 200, 100)]
        cols = [(0, 0, 200, 150)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=150,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_rows == 3
        row_indices = [c.row for c in result.cells]
        assert row_indices == sorted(row_indices)

    def test_cols_sorted_left_to_right(self):
        """Columns should be sorted by x1 after grid reconstruction."""
        cols = [(200, 0, 300, 100), (0, 0, 100, 100), (100, 0, 200, 100)]
        rows = [(0, 0, 300, 100)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=300,
            image_height=100,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.num_cols == 3
        col_indices = [c.col for c in result.cells]
        assert col_indices == sorted(col_indices)

    def test_default_cell_type_is_data(self):
        """Non-header rows should produce DATA cells."""
        rows = [(0, 0, 200, 50)]
        cols = [(0, 0, 200, 50)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=50,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.cells[0].cell_type == CellType.DATA

    def test_cell_spans_are_1_by_default(self):
        """All cells should have row_span=1 and col_span=1 from grid logic."""
        rows = [(0, 0, 200, 50), (0, 50, 200, 100)]
        cols = [(0, 0, 100, 100), (100, 0, 200, 100)]
        boxes, scores, labels = self._boxes_scores_labels(rows, cols)
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=100,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        for cell in result.cells:
            assert cell.row_span == 1
            assert cell.col_span == 1

    def test_cell_bbox_is_intersection(self):
        """Cell bbox should equal the intersection of its row and column boxes."""
        row = (10, 20, 200, 70)  # x: 10–200, y: 20–70
        col = (50, 0, 150, 100)  # x: 50–150, y: 0–100
        boxes, scores, labels = self._boxes_scores_labels([row], [col])
        result = _detections_to_table_output(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_width=200,
            image_height=100,
            ocr_output=None,
            model_name="test",
            label_map=_LABEL_MAP,
        )
        assert result.cell_count == 1
        bbox = result.cells[0].bbox
        assert bbox.x1 == 50  # max(10, 50)
        assert bbox.y1 == 20  # max(20, 0)
        assert bbox.x2 == 150  # min(200, 150)
        assert bbox.y2 == 70  # min(70, 100)


# ===========================================================================
# ONNX preprocessing helpers
# ===========================================================================


class TestONNXPreprocess:
    """Tests for ONNX preprocessing utilities."""

    def test_preprocess_output_shape(self):
        """_preprocess should return (1, 3, 800, 800) float32."""
        img = Image.new("RGB", (640, 480), color=(128, 64, 32))
        arr = _preprocess(img)
        assert arr.shape == (1, 3, 800, 800)
        assert arr.dtype == np.float32

    def test_preprocess_normalised_range(self):
        """After ImageNet normalisation, black pixels should be negative."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        arr = _preprocess(img)
        assert arr.min() < 0

    def test_sigmoid_zero(self):
        """sigmoid(0) should be ~0.5."""
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-6

    def test_sigmoid_large_positive(self):
        """sigmoid(100) should be ~1.0."""
        result = _sigmoid(np.array([100.0]))
        assert result[0] > 0.999

    def test_sigmoid_large_negative(self):
        """sigmoid(-100) should be ~0.0."""
        result = _sigmoid(np.array([-100.0]))
        assert result[0] < 0.001

    def test_sigmoid_batch(self):
        """sigmoid applied to a batch preserves shape and is monotonically increasing."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _sigmoid(x)
        assert result.shape == x.shape
        assert list(result) == sorted(result)

    def test_box_cxcywh_to_xyxy_basic(self):
        """Test center-format to corner-format conversion: cx=50,cy=50,w=100,h=80."""
        boxes = np.array([[50.0, 50.0, 100.0, 80.0]])
        result = _box_cxcywh_to_xyxy(boxes)
        assert result.shape == (1, 4)
        assert abs(result[0, 0] - 0.0) < 1e-5  # x1 = cx - w/2
        assert abs(result[0, 1] - 10.0) < 1e-5  # y1 = cy - h/2
        assert abs(result[0, 2] - 100.0) < 1e-5  # x2 = cx + w/2
        assert abs(result[0, 3] - 90.0) < 1e-5  # y2 = cy + h/2

    def test_box_cxcywh_to_xyxy_batch_shape(self):
        """Batch conversion preserves shape."""
        boxes = np.random.rand(5, 4).astype(np.float32)
        result = _box_cxcywh_to_xyxy(boxes)
        assert result.shape == (5, 4)

    def test_box_cxcywh_to_xyxy_width_height_preserved(self):
        """Converted box width and height should equal original w/h."""
        boxes = np.array([[100.0, 100.0, 60.0, 40.0]])
        result = _box_cxcywh_to_xyxy(boxes)
        w = result[0, 2] - result[0, 0]
        h = result[0, 3] - result[0, 1]
        assert abs(w - 60.0) < 1e-5
        assert abs(h - 40.0) < 1e-5

    def test_box_cxcywh_to_xyxy_center_preserved(self):
        """The centre of the converted box should equal the original cx/cy."""
        boxes = np.array([[200.0, 150.0, 80.0, 60.0]])
        result = _box_cxcywh_to_xyxy(boxes)
        cx = (result[0, 0] + result[0, 2]) / 2
        cy = (result[0, 1] + result[0, 3]) / 2
        assert abs(cx - 200.0) < 1e-5
        assert abs(cy - 150.0) < 1e-5
