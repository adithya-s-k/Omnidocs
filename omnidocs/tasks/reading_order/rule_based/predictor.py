"""
Rule-based reading order predictor.

Uses spatial analysis and R-tree indexing to determine the logical
reading sequence of document elements. Self-contained implementation
without external dependencies on docling-ibm-models.

Based on the algorithm from docling-ibm-models, adapted for omnidocs.
"""

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from rtree import index as rtree_index

from omnidocs.tasks.reading_order.base import BaseReadingOrderPredictor
from omnidocs.tasks.reading_order.models import (
    BoundingBox,
    ElementType,
    OrderedElement,
    ReadingOrderOutput,
)

if TYPE_CHECKING:
    from omnidocs.tasks.layout_extraction.models import LayoutOutput
    from omnidocs.tasks.ocr_extraction.models import OCROutput

_log = logging.getLogger(__name__)


# Mapping from layout labels to reading order element types
LABEL_TO_ELEMENT_TYPE: Dict[str, ElementType] = {
    "title": ElementType.TITLE,
    "text": ElementType.TEXT,
    "list": ElementType.LIST,
    "figure": ElementType.FIGURE,
    "table": ElementType.TABLE,
    "caption": ElementType.CAPTION,
    "formula": ElementType.FORMULA,
    "footnote": ElementType.FOOTNOTE,
    "page_header": ElementType.PAGE_HEADER,
    "page_footer": ElementType.PAGE_FOOTER,
    "code": ElementType.CODE,
    "abandon": ElementType.OTHER,
    "unknown": ElementType.OTHER,
}


@dataclass
class _PageElement:
    """
    Internal page element for reading order algorithm.

    Uses bottom-left coordinate origin for compatibility with the algorithm.
    """

    cid: int  # Original element ID
    text: str
    page_no: int
    page_width: float
    page_height: float
    label: ElementType
    # Coordinates (bottom-left origin)
    left: float
    bottom: float
    right: float
    top: float

    eps: float = 1.0e-3

    def __lt__(self, other: "_PageElement") -> bool:
        """Compare for sorting - by page, then vertical position, then horizontal."""
        if self.page_no == other.page_no:
            if self.overlaps_horizontally(other):
                return self.bottom > other.bottom  # Higher bottom = higher on page
            else:
                return self.left < other.left  # Leftmost first
        else:
            return self.page_no < other.page_no

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def overlaps_horizontally(self, other: "_PageElement") -> bool:
        """Check if two elements overlap horizontally."""
        return not (self.right <= other.left or other.right <= self.left)

    def overlaps_vertically(self, other: "_PageElement") -> bool:
        """Check if two elements overlap vertically."""
        return not (self.top <= other.bottom or other.top <= self.bottom)

    def overlaps(self, other: "_PageElement") -> bool:
        """Check if two elements overlap in both dimensions."""
        return self.overlaps_horizontally(other) and self.overlaps_vertically(other)

    def is_strictly_above(self, other: "_PageElement") -> bool:
        """Check if self is strictly above other (no vertical overlap)."""
        return self.bottom >= other.top - self.eps

    def is_strictly_left_of(self, other: "_PageElement") -> bool:
        """Check if self is strictly to the left of other."""
        return self.right <= other.left + self.eps

    def overlaps_vertically_with_iou(self, other: "_PageElement", min_iou: float) -> bool:
        """Check if vertical overlap meets minimum IoU threshold."""
        if not self.overlaps_vertically(other):
            return False

        overlap_bottom = max(self.bottom, other.bottom)
        overlap_top = min(self.top, other.top)
        overlap_height = overlap_top - overlap_bottom

        union_bottom = min(self.bottom, other.bottom)
        union_top = max(self.top, other.top)
        union_height = union_top - union_bottom

        if union_height <= 0:
            return False

        return (overlap_height / union_height) >= min_iou


@dataclass
class _ReadingOrderState:
    """State container for reading order prediction on a single page."""

    h2i_map: Dict[int, int] = field(default_factory=dict)  # cid -> index
    i2h_map: Dict[int, int] = field(default_factory=dict)  # index -> cid
    l2r_map: Dict[int, int] = field(default_factory=dict)  # left-to-right links
    r2l_map: Dict[int, int] = field(default_factory=dict)  # right-to-left links
    up_map: Dict[int, List[int]] = field(default_factory=dict)  # elements above
    dn_map: Dict[int, List[int]] = field(default_factory=dict)  # elements below
    heads: List[int] = field(default_factory=list)  # starting elements


class RuleBasedReadingOrderPredictor(BaseReadingOrderPredictor):
    """
    Rule-based reading order predictor using spatial analysis.

    Uses R-tree spatial indexing and rule-based algorithms to determine
    the logical reading sequence of document elements. This is a CPU-only
    implementation that doesn't require GPU resources.

    Features:
    - Multi-column layout detection
    - Header/footer separation
    - Caption-to-figure/table association
    - Footnote linking
    - Element merge suggestions

    Example:
        ```python
        from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

        # Initialize components
        layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
        ocr = EasyOCR(config=EasyOCRConfig())
        predictor = RuleBasedReadingOrderPredictor()

        # Process document
        layout = layout_extractor.extract(image)
        ocr_result = ocr.extract(image)
        reading_order = predictor.predict(layout, ocr_result)

        # Get text in reading order
        text = reading_order.get_full_text()
        ```
    """

    def __init__(self):
        """Initialize the reading order predictor."""
        self.dilated_page_element = True
        # Apply horizontal dilation only if less than this page-width normalized threshold
        self._horizontal_dilation_threshold_norm = 0.15

    def predict(
        self,
        layout: "LayoutOutput",
        ocr: Optional["OCROutput"] = None,
        page_no: int = 0,
    ) -> ReadingOrderOutput:
        """
        Predict reading order for a single page.

        Args:
            layout: Layout detection results with bounding boxes
            ocr: Optional OCR results for text content
            page_no: Page number (for multi-page documents)

        Returns:
            ReadingOrderOutput with ordered elements and associations
        """
        page_width = layout.image_width
        page_height = layout.image_height

        # Build text map from OCR if available
        text_map: Dict[int, str] = {}
        if ocr:
            text_map = self._build_text_map(layout, ocr)

        # Convert layout boxes to internal PageElements
        page_elements: List[_PageElement] = []
        for i, box in enumerate(layout.bboxes):
            label_str = box.label.value.lower()
            element_type = LABEL_TO_ELEMENT_TYPE.get(label_str, ElementType.OTHER)

            # Convert from top-left origin to bottom-left origin
            elem = _PageElement(
                cid=i,
                text=text_map.get(i, ""),
                page_no=page_no,
                page_width=page_width,
                page_height=page_height,
                label=element_type,
                left=box.bbox.x1,
                bottom=page_height - box.bbox.y2,  # Convert y2 to bottom
                right=box.bbox.x2,
                top=page_height - box.bbox.y1,  # Convert y1 to top
            )
            page_elements.append(elem)

        # Run reading order prediction
        sorted_elements = self._predict_reading_order(page_elements)

        # Get caption associations
        caption_map = self._find_to_captions(sorted_elements)

        # Get footnote associations
        footnote_map = self._find_to_footnotes(sorted_elements)

        # Get merge suggestions
        merge_map = self._predict_merges(sorted_elements)

        # Convert to OrderedElements
        ordered_elements: List[OrderedElement] = []
        for idx, elem in enumerate(sorted_elements):
            # Convert back from bottom-left to top-left origin
            bbox = BoundingBox(
                x1=elem.left,
                y1=page_height - elem.top,
                x2=elem.right,
                y2=page_height - elem.bottom,
            )

            confidence = 1.0
            if elem.cid < len(layout.bboxes):
                confidence = layout.bboxes[elem.cid].confidence

            ordered_elem = OrderedElement(
                index=idx,
                element_type=elem.label,
                bbox=bbox,
                text=elem.text,
                confidence=confidence,
                page_no=page_no,
                original_id=elem.cid,
            )
            ordered_elements.append(ordered_elem)

        return ReadingOrderOutput(
            ordered_elements=ordered_elements,
            caption_map=caption_map,
            footnote_map=footnote_map,
            merge_map=merge_map,
            image_width=page_width,
            image_height=page_height,
            model_name="RuleBasedReadingOrderPredictor",
        )

    def _predict_reading_order(self, page_elements: List[_PageElement]) -> List[_PageElement]:
        """Predict reading order across all page elements."""
        page_nos: Set[int] = {elem.page_no for elem in page_elements}

        # Separate headers, footers, and main content
        page_to_elems: Dict[int, List[_PageElement]] = {p: [] for p in page_nos}
        page_to_headers: Dict[int, List[_PageElement]] = {p: [] for p in page_nos}
        page_to_footers: Dict[int, List[_PageElement]] = {p: [] for p in page_nos}

        for elem in page_elements:
            if elem.label == ElementType.PAGE_HEADER:
                page_to_headers[elem.page_no].append(elem)
            elif elem.label == ElementType.PAGE_FOOTER:
                page_to_footers[elem.page_no].append(elem)
            else:
                page_to_elems[elem.page_no].append(elem)

        # Process each category separately
        for page_no in page_nos:
            page_to_headers[page_no] = self._predict_page(page_to_headers[page_no])
            page_to_elems[page_no] = self._predict_page(page_to_elems[page_no])
            page_to_footers[page_no] = self._predict_page(page_to_footers[page_no])

        # Combine in order: headers, content, footers
        sorted_elements: List[_PageElement] = []
        for page_no in sorted(page_nos):
            sorted_elements.extend(page_to_headers[page_no])
            sorted_elements.extend(page_to_elems[page_no])
            sorted_elements.extend(page_to_footers[page_no])

        return sorted_elements

    def _predict_page(self, page_elements: List[_PageElement]) -> List[_PageElement]:
        """Reorder elements on a single page into reading order."""
        if not page_elements:
            return []

        state = _ReadingOrderState()

        # Initialize maps
        self._init_h2i_map(page_elements, state)
        self._init_l2r_map(page_elements, state)
        self._init_ud_maps(page_elements, state)

        # Apply horizontal dilation for better column detection
        if self.dilated_page_element:
            dilated_elements = copy.deepcopy(page_elements)
            dilated_elements = self._do_horizontal_dilation(page_elements, dilated_elements, state)
            self._init_ud_maps(dilated_elements, state)

        # Find head elements (no predecessors)
        self._find_heads(page_elements, state)

        # Sort children by position
        self._sort_ud_maps(page_elements, state)

        # Build final order via DFS
        order = self._find_order(page_elements, state)

        return [page_elements[i] for i in order]

    def _init_h2i_map(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> None:
        """Initialize cid-to-index and index-to-cid maps."""
        state.h2i_map = {}
        state.i2h_map = {}
        for i, elem in enumerate(page_elems):
            state.h2i_map[elem.cid] = i
            state.i2h_map[i] = elem.cid

    def _init_l2r_map(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> None:
        """Initialize left-to-right links (currently disabled)."""
        state.l2r_map = {}
        state.r2l_map = {}

    def _init_ud_maps(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> None:
        """Initialize up/down maps using R-tree spatial indexing."""
        state.up_map = {i: [] for i in range(len(page_elems))}
        state.dn_map = {i: [] for i in range(len(page_elems))}

        # Build R-tree spatial index
        spatial_idx = rtree_index.Index()
        for i, elem in enumerate(page_elems):
            spatial_idx.insert(i, (elem.left, elem.bottom, elem.right, elem.top))

        for j, elem_j in enumerate(page_elems):
            if j in state.r2l_map:
                i = state.r2l_map[j]
                state.dn_map[i] = [j]
                state.up_map[j] = [i]
                continue

            # Find elements above that might precede in reading order
            query_bbox = (elem_j.l - 0.1, elem_j.t, elem_j.r + 0.1, float("inf"))
            candidates = list(spatial_idx.intersection(query_bbox))

            for i in candidates:
                if i == j:
                    continue

                elem_i = page_elems[i]

                # Check spatial relationship
                if not (elem_i.is_strictly_above(elem_j) and elem_i.overlaps_horizontally(elem_j)):
                    continue

                # Check for interrupting elements
                if not self._has_sequence_interruption(spatial_idx, page_elems, i, j, elem_i, elem_j):
                    # Follow left-to-right mapping
                    while i in state.l2r_map:
                        i = state.l2r_map[i]

                    state.dn_map[i].append(j)
                    state.up_map[j].append(i)

    def _has_sequence_interruption(
        self,
        spatial_idx: rtree_index.Index,
        page_elems: List[_PageElement],
        i: int,
        j: int,
        elem_i: _PageElement,
        elem_j: _PageElement,
    ) -> bool:
        """Check if any element interrupts the reading sequence between i and j."""
        x_min = min(elem_i.l, elem_j.l) - 1.0
        x_max = max(elem_i.r, elem_j.r) + 1.0
        y_min = elem_j.t
        y_max = elem_i.b

        candidates = list(spatial_idx.intersection((x_min, y_min, x_max, y_max)))

        for w in candidates:
            if w in (i, j):
                continue

            elem_w = page_elems[w]

            if (
                (elem_i.overlaps_horizontally(elem_w) or elem_j.overlaps_horizontally(elem_w))
                and elem_i.is_strictly_above(elem_w)
                and elem_w.is_strictly_above(elem_j)
            ):
                return True

        return False

    def _do_horizontal_dilation(
        self,
        page_elems: List[_PageElement],
        dilated_elems: List[_PageElement],
        state: _ReadingOrderState,
    ) -> List[_PageElement]:
        """Apply horizontal dilation for better column detection."""
        if not page_elems:
            return dilated_elems

        th = self._horizontal_dilation_threshold_norm * page_elems[0].page_width

        for i, elem_i in enumerate(dilated_elems):
            x0, y0, x1, y1 = elem_i.l, elem_i.b, elem_i.r, elem_i.t

            if i in state.up_map and state.up_map[i]:
                elem_up = page_elems[state.up_map[i][0]]
                x0_dil = min(x0, elem_up.l)
                x1_dil = max(x1, elem_up.r)
                if (x0 - x0_dil) <= th and (x1_dil - x1) <= th:
                    x0, x1 = x0_dil, x1_dil

            if i in state.dn_map and state.dn_map[i]:
                elem_dn = page_elems[state.dn_map[i][0]]
                x0_dil = min(x0, elem_dn.l)
                x1_dil = max(x1, elem_dn.r)
                if (x0 - x0_dil) <= th and (x1_dil - x1) <= th:
                    x0, x1 = x0_dil, x1_dil

            # Check for overlaps before applying dilation
            overlaps = any(
                page_elems[k].overlaps(
                    _PageElement(
                        cid=-1,
                        text="",
                        page_no=0,
                        page_width=0,
                        page_height=0,
                        label=ElementType.OTHER,
                        left=x0,
                        bottom=y0,
                        right=x1,
                        top=y1,
                    )
                )
                for k in range(len(page_elems))
                if k != i
            )

            if not overlaps:
                dilated_elems[i].l = x0
                dilated_elems[i].r = x1

        return dilated_elems

    def _find_heads(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> None:
        """Find head elements (those with no predecessors)."""
        head_elems = [page_elems[key] for key, vals in state.up_map.items() if not vals]
        head_elems = sorted(head_elems)  # Uses __lt__
        state.heads = [state.h2i_map[elem.cid] for elem in head_elems]

    def _sort_ud_maps(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> None:
        """Sort children in down_map by position."""
        for ind_i, vals in state.dn_map.items():
            child_elems = sorted([page_elems[j] for j in vals])
            state.dn_map[ind_i] = [state.h2i_map[c.cid] for c in child_elems]

    def _find_order(self, page_elems: List[_PageElement], state: _ReadingOrderState) -> List[int]:
        """Find final reading order via depth-first search."""
        order: List[int] = []
        visited = [False] * len(page_elems)

        for j in state.heads:
            if not visited[j]:
                order.append(j)
                visited[j] = True
                self._dfs_downwards(j, order, visited, state)

        if len(order) != len(page_elems):
            _log.warning(f"Reading order incomplete: {len(order)}/{len(page_elems)}")

        return order

    def _dfs_upwards(self, j: int, visited: List[bool], state: _ReadingOrderState) -> int:
        """Depth-first search upwards to find unvisited ancestors."""
        k = j
        while True:
            found = False
            for ind in state.up_map[k]:
                if not visited[ind]:
                    k = ind
                    found = True
                    break
            if not found:
                return k

    def _dfs_downwards(
        self,
        j: int,
        order: List[int],
        visited: List[bool],
        state: _ReadingOrderState,
    ) -> None:
        """Depth-first search downwards (non-recursive)."""
        stack: List[Tuple[List[int], int]] = [(state.dn_map[j], 0)]

        while stack:
            inds, offset = stack[-1]
            found = False

            if offset < len(inds):
                for new_offset, i in enumerate(inds[offset:], offset):
                    k = self._dfs_upwards(i, visited, state)
                    if not visited[k]:
                        order.append(k)
                        visited[k] = True
                        stack[-1] = (inds, new_offset + 1)
                        stack.append((state.dn_map[k], 0))
                        found = True
                        break

            if not found:
                stack.pop()

    def _find_to_captions(self, page_elements: List[_PageElement]) -> Dict[int, List[int]]:
        """Find caption associations for figures/tables."""
        to_captions: Dict[int, List[int]] = {}
        from_captions: Dict[int, Tuple[List[int], List[int]]] = {}

        # Initialize from_captions for each caption
        for elem in page_elements:
            if elem.label == ElementType.CAPTION:
                from_captions[elem.cid] = ([], [])

        # Find preceding and following figures/tables for each caption
        target_labels = {ElementType.TABLE, ElementType.FIGURE, ElementType.CODE}

        for ind, elem in enumerate(page_elements):
            if elem.label == ElementType.CAPTION:
                # Look backwards
                ind_m1 = ind - 1
                while ind_m1 >= 0 and page_elements[ind_m1].label in target_labels:
                    from_captions[elem.cid][0].append(page_elements[ind_m1].cid)
                    ind_m1 -= 1

                # Look forwards
                ind_p1 = ind + 1
                while ind_p1 < len(page_elements) and page_elements[ind_p1].label in target_labels:
                    from_captions[elem.cid][1].append(page_elements[ind_p1].cid)
                    ind_p1 += 1

        # Assign captions to targets
        assigned_cids: Set[int] = set()

        # First pass: unambiguous assignments
        for cid_i, (preceding, following) in from_captions.items():
            if not preceding and following:
                for cid_j in following:
                    if cid_j not in to_captions:
                        to_captions[cid_j] = []
                    if cid_i not in to_captions[cid_j]:
                        to_captions[cid_j].append(cid_i)
                    assigned_cids.add(cid_j)
            elif preceding and not following:
                for cid_j in preceding:
                    if cid_j not in to_captions:
                        to_captions[cid_j] = []
                    if cid_i not in to_captions[cid_j]:
                        to_captions[cid_j].append(cid_i)
                    assigned_cids.add(cid_j)

        # Remove duplicates: keep only closest caption per target
        def _remove_overlapping(mapping: Dict[int, List[int]]) -> Dict[int, List[int]]:
            used: Set[int] = set()
            result: Dict[int, List[int]] = {}
            for key, values in sorted(mapping.items()):
                valid = [v for v in sorted(values, key=lambda v: abs(v - key)) if v not in used]
                if valid:
                    result[key] = [valid[0]]
                    used.add(valid[0])
            return result

        return _remove_overlapping(to_captions)

    def _find_to_footnotes(self, page_elements: List[_PageElement]) -> Dict[int, List[int]]:
        """Find footnote associations for figures/tables."""
        to_footnotes: Dict[int, List[int]] = {}
        target_labels = {ElementType.TABLE, ElementType.FIGURE}

        for ind, elem in enumerate(page_elements):
            if elem.label in target_labels:
                ind_p1 = ind + 1
                while ind_p1 < len(page_elements) and page_elements[ind_p1].label == ElementType.FOOTNOTE:
                    if elem.cid not in to_footnotes:
                        to_footnotes[elem.cid] = []
                    to_footnotes[elem.cid].append(page_elements[ind_p1].cid)
                    ind_p1 += 1

        return to_footnotes

    def _predict_merges(self, sorted_elements: List[_PageElement]) -> Dict[int, List[int]]:
        """Predict which elements should be merged (split paragraphs)."""
        merges: Dict[int, List[int]] = {}
        skip_labels = {
            ElementType.PAGE_HEADER,
            ElementType.PAGE_FOOTER,
            ElementType.TABLE,
            ElementType.FIGURE,
            ElementType.CAPTION,
            ElementType.FOOTNOTE,
        }

        curr_ind = -1
        for ind, elem in enumerate(sorted_elements):
            if ind <= curr_ind:
                continue

            if elem.label == ElementType.TEXT:
                ind_p1 = ind + 1
                while ind_p1 < len(sorted_elements) and sorted_elements[ind_p1].label in skip_labels:
                    ind_p1 += 1

                if ind_p1 < len(sorted_elements):
                    next_elem = sorted_elements[ind_p1]
                    if next_elem.label == elem.label and (
                        elem.page_no != next_elem.page_no or elem.is_strictly_left_of(next_elem)
                    ):
                        # Check for sentence continuation patterns
                        m1 = re.fullmatch(r".+([a-z,\-])(\s*)", elem.text)
                        m2 = re.fullmatch(r"(\s*[a-z])(.+)", next_elem.text)
                        if m1 and m2:
                            merges[elem.cid] = [next_elem.cid]
                            curr_ind = ind_p1

        return merges

    def _build_text_map(self, layout: "LayoutOutput", ocr: "OCROutput") -> Dict[int, str]:
        """Build a map from layout element IDs to text content."""
        text_map: Dict[int, str] = {}

        for i, box in enumerate(layout.bboxes):
            matched_texts = []
            for text_block in ocr.text_blocks:
                if self._boxes_overlap(box.bbox, text_block.bbox):
                    matched_texts.append(text_block.text)

            if matched_texts:
                text_map[i] = " ".join(matched_texts)

        return text_map

    def _boxes_overlap(self, box1, box2, min_overlap: float = 0.5) -> bool:
        """Check if two boxes have significant overlap."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        if box2_area <= 0:
            return False

        return intersection / box2_area >= min_overlap
