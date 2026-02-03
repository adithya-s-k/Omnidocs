"""
Evaluation Utilities for OmniDocs Testing.

Provides metrics and comparison functions for evaluating extraction quality.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple


@dataclass
class TextEvaluationResult:
    """Result of text extraction evaluation."""

    character_accuracy: float  # 0-1, ratio of matching characters
    word_accuracy: float  # 0-1, ratio of matching words
    line_accuracy: float  # 0-1, ratio of matching lines
    edit_distance: int  # Levenshtein distance
    similarity_ratio: float  # SequenceMatcher ratio
    missing_words: List[str]  # Words in ground truth but not in extracted
    extra_words: List[str]  # Words in extracted but not in ground truth

    @property
    def is_passing(self) -> bool:
        """Check if extraction passes minimum quality threshold."""
        return self.character_accuracy >= 0.9 and self.word_accuracy >= 0.85


@dataclass
class LayoutEvaluationResult:
    """Result of layout extraction evaluation."""

    precision: float  # Correctly detected / total detected
    recall: float  # Correctly detected / total ground truth
    f1_score: float  # Harmonic mean of precision and recall
    iou_scores: List[float]  # IoU for each matched box
    mean_iou: float  # Average IoU
    matched_boxes: int
    total_detected: int
    total_ground_truth: int

    @property
    def is_passing(self) -> bool:
        """Check if layout detection passes minimum quality threshold."""
        return self.f1_score >= 0.8 and self.mean_iou >= 0.7


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Convert to lowercase
    - Remove extra whitespace
    - Normalize line endings
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def evaluate_text_extraction(
    extracted: str,
    ground_truth: str,
    normalize: bool = True,
) -> TextEvaluationResult:
    """
    Evaluate text extraction quality against ground truth.

    Args:
        extracted: Extracted text from the model
        ground_truth: Known correct text
        normalize: Whether to normalize text before comparison

    Returns:
        TextEvaluationResult with detailed metrics
    """
    if normalize:
        extracted = normalize_text(extracted)
        ground_truth = normalize_text(ground_truth)

    # Character-level accuracy
    matcher = SequenceMatcher(None, ground_truth, extracted)
    similarity_ratio = matcher.ratio()

    # Calculate character accuracy
    matching_blocks = matcher.get_matching_blocks()
    matching_chars = sum(block.size for block in matching_blocks)
    max_len = max(len(ground_truth), len(extracted), 1)
    character_accuracy = matching_chars / max_len

    # Word-level analysis
    gt_words = set(ground_truth.split())
    ext_words = set(extracted.split())

    common_words = gt_words & ext_words
    missing_words = list(gt_words - ext_words)
    extra_words = list(ext_words - gt_words)

    word_accuracy = len(common_words) / max(len(gt_words), 1)

    # Line-level analysis
    gt_lines = [line.strip() for line in ground_truth.split("\n") if line.strip()]
    ext_lines = [line.strip() for line in extracted.split("\n") if line.strip()]

    matching_lines = sum(1 for line in gt_lines if line in ext_lines)
    line_accuracy = matching_lines / max(len(gt_lines), 1)

    # Edit distance
    edit_distance = levenshtein_distance(ground_truth, extracted)

    return TextEvaluationResult(
        character_accuracy=character_accuracy,
        word_accuracy=word_accuracy,
        line_accuracy=line_accuracy,
        edit_distance=edit_distance,
        similarity_ratio=similarity_ratio,
        missing_words=missing_words,
        extra_words=extra_words,
    )


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    Calculate Intersection over Union for two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) for first box
        box2: (x1, y1, x2, y2) for second box

    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def evaluate_layout_extraction(
    detected_boxes: List[Tuple[float, float, float, float]],
    ground_truth_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> LayoutEvaluationResult:
    """
    Evaluate layout detection quality against ground truth.

    Args:
        detected_boxes: List of detected bounding boxes (x1, y1, x2, y2)
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: Minimum IoU to consider a match

    Returns:
        LayoutEvaluationResult with detailed metrics
    """
    if not ground_truth_boxes:
        return LayoutEvaluationResult(
            precision=1.0 if not detected_boxes else 0.0,
            recall=1.0,
            f1_score=1.0 if not detected_boxes else 0.0,
            iou_scores=[],
            mean_iou=0.0,
            matched_boxes=0,
            total_detected=len(detected_boxes),
            total_ground_truth=0,
        )

    if not detected_boxes:
        return LayoutEvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            iou_scores=[],
            mean_iou=0.0,
            matched_boxes=0,
            total_detected=0,
            total_ground_truth=len(ground_truth_boxes),
        )

    # Calculate IoU matrix
    iou_matrix = []
    for det_box in detected_boxes:
        row = [calculate_iou(det_box, gt_box) for gt_box in ground_truth_boxes]
        iou_matrix.append(row)

    # Greedy matching
    matched_gt: Set[int] = set()
    matched_det: Set[int] = set()
    iou_scores = []

    # Sort by IoU and greedily match
    matches = []
    for i, row in enumerate(iou_matrix):
        for j, iou in enumerate(row):
            if iou >= iou_threshold:
                matches.append((iou, i, j))

    matches.sort(reverse=True)

    for iou, det_idx, gt_idx in matches:
        if det_idx not in matched_det and gt_idx not in matched_gt:
            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            iou_scores.append(iou)

    # Calculate metrics
    matched_boxes = len(matched_det)
    precision = matched_boxes / len(detected_boxes) if detected_boxes else 0.0
    recall = matched_boxes / len(ground_truth_boxes) if ground_truth_boxes else 0.0

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    return LayoutEvaluationResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou_scores=iou_scores,
        mean_iou=mean_iou,
        matched_boxes=matched_boxes,
        total_detected=len(detected_boxes),
        total_ground_truth=len(ground_truth_boxes),
    )


def evaluate_ocr_extraction(
    detected_blocks: List[Dict[str, Any]],
    ground_truth_blocks: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate OCR extraction quality.

    Args:
        detected_blocks: List of dicts with 'text' and 'bbox' keys
        ground_truth_blocks: List of dicts with 'text' and 'bbox' keys
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Dictionary with layout metrics and text accuracy for matched boxes
    """
    # Extract boxes
    det_boxes = [b["bbox"] for b in detected_blocks]
    gt_boxes = [b["bbox"] for b in ground_truth_blocks]

    # Layout evaluation
    layout_result = evaluate_layout_extraction(det_boxes, gt_boxes, iou_threshold)

    # Text evaluation for matched boxes
    text_accuracies = []

    # Re-do matching to get text pairs
    for det_block in detected_blocks:
        best_iou = 0.0
        best_gt = None

        for gt_block in ground_truth_blocks:
            iou = calculate_iou(det_block["bbox"], gt_block["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_block

        if best_iou >= iou_threshold and best_gt:
            text_result = evaluate_text_extraction(
                det_block["text"],
                best_gt["text"],
            )
            text_accuracies.append(text_result.character_accuracy)

    return {
        "layout": layout_result,
        "text_accuracies": text_accuracies,
        "mean_text_accuracy": sum(text_accuracies) / len(text_accuracies) if text_accuracies else 0.0,
    }


def evaluate_table_extraction(
    extracted_cells: List[List[str]],
    ground_truth_cells: List[List[str]],
) -> Dict[str, Any]:
    """
    Evaluate table extraction quality.

    Args:
        extracted_cells: 2D list of extracted cell contents
        ground_truth_cells: 2D list of ground truth cell contents

    Returns:
        Dictionary with table extraction metrics
    """
    if not ground_truth_cells:
        return {
            "cell_accuracy": 1.0 if not extracted_cells else 0.0,
            "row_accuracy": 1.0 if not extracted_cells else 0.0,
            "structure_match": not extracted_cells,
        }

    gt_rows = len(ground_truth_cells)
    gt_cols = len(ground_truth_cells[0]) if ground_truth_cells else 0

    ext_rows = len(extracted_cells)
    ext_cols = len(extracted_cells[0]) if extracted_cells else 0

    # Structure match
    structure_match = (gt_rows == ext_rows) and (gt_cols == ext_cols)

    # Cell-level accuracy
    correct_cells = 0
    total_cells = gt_rows * gt_cols

    for i in range(min(gt_rows, ext_rows)):
        for j in range(min(gt_cols, ext_cols)):
            gt_cell = normalize_text(ground_truth_cells[i][j])
            ext_cell = normalize_text(extracted_cells[i][j]) if j < len(extracted_cells[i]) else ""

            if gt_cell == ext_cell:
                correct_cells += 1

    cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0

    # Row accuracy
    correct_rows = 0
    for i in range(min(gt_rows, ext_rows)):
        gt_row = [normalize_text(c) for c in ground_truth_cells[i]]
        ext_row = [normalize_text(c) for c in extracted_cells[i]] if i < len(extracted_cells) else []

        if gt_row == ext_row:
            correct_rows += 1

    row_accuracy = correct_rows / gt_rows if gt_rows > 0 else 0.0

    return {
        "cell_accuracy": cell_accuracy,
        "row_accuracy": row_accuracy,
        "structure_match": structure_match,
        "gt_shape": (gt_rows, gt_cols),
        "ext_shape": (ext_rows, ext_cols),
    }
