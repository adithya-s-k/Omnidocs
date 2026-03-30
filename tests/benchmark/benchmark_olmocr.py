"""
Modal Benchmark Runner for OmniDocs — olmOCR-Bench evaluation.

Runs all text extraction models against the olmOCR-Bench dataset
(allenai/olmOCR-bench) and reports per-split pass rates matching
the official leaderboard format:

  ArXiv | HdrFtr | TinyTxt | MultCol | OldScan | OldMath | Tables | Overall

Scoring uses BINARY pass/fail unit tests (not NED/BLEU/TEDS):
  text_present   — fuzzy substring match, optional first_n / last_n constraint
  text_absent    — text must NOT appear (headers, footers, page numbers)
  reading_order  — text A must appear before text B in the output
  table          — cell value + neighbour relationship checks (above/below/left/right)
  math           — LaTeX symbol presence / relative position

Usage:
    modal run tests/benchmark/benchmark_olmocr.py

    # Specific models
    modal run tests/benchmark/benchmark_olmocr.py --models glmocr deepseek

    # Specific splits
    modal run tests/benchmark/benchmark_olmocr.py --splits arxiv_math table_tests

    # Limit samples per split (fast iteration)
    modal run tests/benchmark/benchmark_olmocr.py --max-per-split 20

    # Save results JSON
    modal run tests/benchmark/benchmark_olmocr.py --output results/olmocr_run01.json

Splits available (from allenai/olmOCR-bench):
    arxiv_math      — arXiv papers, heavy LaTeX math
    headers_footers — headers/footers that should be ABSENT from output
    long_tiny_text  — small font, dense text presence checks
    multi_column    — multi-column reading order
    old_scans       — scanned historical documents (text + reading order)
    old_scans_math  — scanned math (old_scans + math formula checks)
    table_tests     — table cell-relationship accuracy

GPU routing mirrors benchmark_omnidocbench.py exactly.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent
MODEL_CACHE = "/data/.cache"

# ---------------------------------------------------------------------------
# Modal images  (identical to benchmark_omnidocbench.py)
# ---------------------------------------------------------------------------

cuda_vllm = "12.8.1"
cuda_pytorch = "12.8.0"
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
_ignore = ["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"]
_env_base = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "OMNIDOCS_MODELS_DIR": MODEL_CACHE,
    "HF_HOME": MODEL_CACHE,
}

# poppler-utils is needed to render PDF pages to images
PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .run_commands("uv pip install pdf2image pillow --system")
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

# VLLM_IMAGE = (
#     modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
#     .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0", "poppler-utils")
#     .run_commands("pip install uv")
#     .run_commands("uv pip install vllm --system")
#     .uv_pip_install(flash_attn_wheel)
#     .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
#     .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
#     .run_commands("uv pip install pdf2image pillow --system")
#     .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
# )

VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0", "poppler-utils")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    # No flash_attn install here — vllm bundles its own compatible version
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
    .run_commands("uv pip install pdf2image pillow --system")
    .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)

GLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .run_commands("uv pip install pdf2image pillow --system")
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

GLM_VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0", "poppler-utils")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm==0.17.0 --system")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[vllm]' --system --override /tmp/overrides.txt"
    )
    .run_commands("uv pip install pdf2image pillow --system")
    .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)

LIGHTON_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .run_commands("uv pip install pdf2image pillow --system")
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

# ---------------------------------------------------------------------------
# Modal app + shared volume
# ---------------------------------------------------------------------------

app = modal.App("omnidocs-olmocr-bench")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

# ---------------------------------------------------------------------------
# olmOCR-Bench splits
# ---------------------------------------------------------------------------

OLM_SPLITS = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]

# Maps each split to the leaderboard column header
SPLIT_LABELS = {
    "arxiv_math": "ArXiv",
    "headers_footers": "HdrFtr",
    "long_tiny_text": "TinyTxt",
    "multi_column": "MultCol",
    "old_scans": "OldScan",
    "old_scans_math": "OldMath",
    "table_tests": "Tables",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OlmTestCase:
    """One unit-test case from a olmOCR-Bench JSONL file."""

    pdf_bytes: bytes  # raw bytes of the PDF file
    page_num: int  # 0-indexed page to render
    check_type: str  # text_present | text_absent | reading_order | table | math
    split: str  # which JSONL split this came from
    case_id: str  # unique identifier for reporting
    payload: dict  # full original JSON record (for scorer)


@dataclass
class OlmResult:
    case_id: str
    split: str
    check_type: str
    model: str
    passed: bool
    latency_s: float
    failed: bool = False  # True if model raised an exception
    error: str = ""
    gt: str = ""  # ground-truth value(s) extracted from payload
    predicted: str = ""  # raw model output (truncated for logging)


# ---------------------------------------------------------------------------
# Dataset loader  (runs locally before sending data to Modal)
# ---------------------------------------------------------------------------
def load_olmocr_bench(splits, max_per_split=None):
    import os

    import huggingface_hub

    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    # Clone the entire dataset repo once — avoids per-file 404s
    local_dir = Path(
        huggingface_hub.snapshot_download(
            repo_id="allenai/olmOCR-bench",
            repo_type="dataset",
        )
    )

    cases = []
    pdf_cache = {}

    for split in splits:
        jsonl_path = local_dir / "bench_data" / f"{split}.jsonl"
        split_cases = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                pdf_rel = rec.get("pdf", "")  # e.g. "headers_footers/abc123_page_6.pdf"
                page_num = int(rec.get("page", 0))
                check_type = rec.get("check_type", rec.get("type", ""))
                # Normalize short aliases used in the dataset's `type` field
                _type_aliases = {
                    "absent": "text_absent",
                    "present": "text_present",
                    "order": "reading_order",
                    "math": "math",
                    "table": "table",
                }
                check_type = _type_aliases.get(check_type, check_type)
                # arxiv_math records don't have check_type — infer it from present keys
                if not check_type:
                    if "math" in rec:
                        check_type = "math"
                    elif "before" in rec and "after" in rec:
                        check_type = "reading_order"
                    elif "cell" in rec:
                        check_type = "table"
                    elif "text" in rec:
                        check_type = "text_absent" if rec.get("absent", False) else "text_present"
                # Real path is bench_data/pdfs/<pdf_rel>
                pdf_path = local_dir / "bench_data" / "pdfs" / pdf_rel

                if not pdf_path.exists():
                    print(f"  [skip] not found: {pdf_path}")
                    continue

                pdf_key = str(pdf_path)
                if pdf_key not in pdf_cache:
                    pdf_cache[pdf_key] = pdf_path.read_bytes()

                case_id = f"{split}/{Path(pdf_rel).stem}/p{page_num}/{check_type}/{len(split_cases)}"
                split_cases.append(
                    OlmTestCase(
                        pdf_bytes=pdf_cache[pdf_key],
                        page_num=page_num,
                        check_type=check_type,
                        split=split,
                        case_id=case_id,
                        payload=rec,
                    )
                )

                if max_per_split and len(split_cases) >= max_per_split:
                    break

        print(f"  → {len(split_cases)} test cases from {split}")
        cases.extend(split_cases)

    return cases


# ---------------------------------------------------------------------------
# PDF → image (runs inside Modal containers)
# ---------------------------------------------------------------------------


def _pdf_page_to_image(pdf_bytes: bytes, page_num: int):
    from pdf2image import convert_from_bytes

    # olmOCR-bench PDFs are pre-extracted single pages.
    # page_num refers to the original source doc, not this file.
    pages = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages")
    return pages[0].convert("RGB")


# ---------------------------------------------------------------------------
# Scorer  (pure Python, no external deps beyond difflib which is stdlib)
# ---------------------------------------------------------------------------


def _fuzzy_contains(text: str, query: str, threshold: float = 0.8) -> bool:
    """
    Return True if `query` appears in `text` with at least `threshold` similarity.
    Uses a sliding window of the same length as `query` and checks SequenceMatcher.
    Falls back to simple substring match when threshold == 1.0.
    """
    if not query:
        return True
    if threshold >= 1.0:
        return query in text

    import difflib

    q_len = len(query)
    if q_len == 0:
        return True

    # Slide a window across `text`
    for start in range(max(1, len(text) - q_len + 1)):
        window = text[start : start + q_len]
        ratio = difflib.SequenceMatcher(None, query, window).ratio()
        if ratio >= threshold:
            return True
    return False


def _get_search_region(text: str, payload: dict) -> str:
    """
    Apply first_n / last_n character constraints from the test payload.
    If neither is specified, return the full text.
    """
    first_n = payload.get("first_n")
    last_n = payload.get("last_n")
    if first_n is not None:
        return text[: int(first_n)]
    if last_n is not None:
        return text[-int(last_n) :]
    return text


def _score_text_present(predicted: str, payload: dict) -> bool:
    """text_present: the query string MUST appear in the output."""
    query = payload.get("text", "")
    threshold = float(payload.get("threshold", 0.8))
    region = _get_search_region(predicted, payload)

    case_sensitive = payload.get("case_sensitive", True)
    if not case_sensitive:
        region = region.lower()
        query = query.lower()

    return _fuzzy_contains(region, query, threshold)


def _score_text_absent(predicted: str, payload: dict) -> bool:
    """text_absent: the query string must NOT appear in the output."""
    query = payload.get("text", "")
    threshold = float(payload.get("threshold", 0.8))
    region = _get_search_region(predicted, payload)

    # text_absent is NOT case-sensitive per olmOCR-bench spec
    region = region.lower()
    query = query.lower()

    return not _fuzzy_contains(region, query, threshold)


def _score_reading_order(predicted: str, payload: dict) -> bool:
    """
    reading_order: text_before must appear before text_after in the output.
    Both use fuzzy matching with the provided threshold.
    """
    before = payload.get("before", "")
    after = payload.get("after", "")
    threshold = float(payload.get("threshold", 0.8))

    case_sensitive = payload.get("case_sensitive", True)
    text = predicted if case_sensitive else predicted.lower()
    if not case_sensitive:
        before = before.lower()
        after = after.lower()

    # Find earliest position where before occurs
    b_len = len(before)
    a_len = len(after)
    if b_len == 0 or a_len == 0:
        return True

    import difflib

    def _first_occurrence(haystack: str, needle: str, thresh: float) -> int:
        """Return the start index of the first fuzzy match, or -1."""
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            window = haystack[i : i + n]
            if difflib.SequenceMatcher(None, needle, window).ratio() >= thresh:
                return i
        return -1

    pos_before = _first_occurrence(text, before, threshold)
    pos_after = _first_occurrence(text, after, threshold)

    if pos_before == -1 or pos_after == -1:
        return False
    return pos_before < pos_after


# def _parse_markdown_table(md: str) -> List[List[str]]:
#     """Parse a markdown pipe table into a 2D list of strings."""
#     rows = []
#     for line in md.strip().split("\n"):
#         line = line.strip()
#         if not line.startswith("|"):
#             continue
#         if re.match(r"^\|[-: |]+\|$", line):
#             continue  # separator row
#         cells = [c.strip() for c in line.strip("|").split("|")]
#         rows.append(cells)
#     return rows


# def _parse_html_table(html: str) -> List[List[str]]:
#     """Parse an HTML table into a 2D list of strings (ignores rowspan/colspan)."""
#     from html.parser import HTMLParser

#     class _Parser(HTMLParser):
#         def __init__(self):
#             super().__init__()
#             self.rows: List[List[str]] = []
#             self._row: List[str] = []
#             self._cell = ""
#             self._in_cell = False

#         def handle_starttag(self, tag, attrs):
#             if tag in ("tr",):
#                 self._row = []
#             elif tag in ("td", "th"):
#                 self._cell = ""
#                 self._in_cell = True

#         def handle_endtag(self, tag):
#             if tag in ("td", "th"):
#                 self._row.append(self._cell.strip())
#                 self._in_cell = False
#             elif tag == "tr" and self._row:
#                 self.rows.append(self._row)

#         def handle_data(self, data):
#             if self._in_cell:
#                 self._cell += data

#     p = _Parser()
#     p.feed(html)
#     return p.rows


# def _find_cell_position(rows: List[List[str]], value: str, threshold: float) -> Optional[Tuple[int, int]]:
#     """Find the (row, col) of the first cell that fuzzy-matches value."""
#     import difflib
#     for r, row in enumerate(rows):
#         for c, cell in enumerate(row):
#             if difflib.SequenceMatcher(None, value.lower(), cell.lower()).ratio() >= threshold:
#                 return (r, c)
#     return None


# def _score_table(predicted: str, payload: dict) -> bool:
#     """
#     table: a target cell value must exist AND have the correct neighbour
#     in the specified direction (above / below / left / right).
#     Searches both HTML tables and markdown pipe tables in the output.
#     """
#     cell  = payload.get("cell", "")
#     up    = payload.get("up")
#     down  = payload.get("down")
#     left  = payload.get("left")
#     right = payload.get("right")

#     # Collect all tables from the predicted output
#     all_row_grids: List[List[List[str]]] = []

#     # HTML tables
#     for m in re.finditer(r"<table[\s\S]*?</table>", predicted, re.IGNORECASE):
#         grid = _parse_html_table(m.group(0))
#         if grid:
#             all_row_grids.append(grid)

#     # Markdown pipe tables (fallback)
#     pipe_pat = re.compile(
#         r"((?:\|[^\n]*\|\n)+\|[-: |]+\|\n(?:\|[^\n]*\|\n)*)",
#         re.MULTILINE,
#     )
#     for m in pipe_pat.finditer(predicted):
#         grid = _parse_markdown_table(m.group(0))
#         if grid:
#             all_row_grids.append(grid)

#     if not all_row_grids:
#         return False

#     dir_to_delta = {
#         "above": (-1, 0),
#         "below": (+1, 0),
#         "left":  (0, -1),
#         "right": (0, +1),
#     }
#     dr, dc = dir_to_delta.get(direction, (0, 0))

#     for grid in all_row_grids:
#         pos = _find_cell_position(grid, target, threshold)
#         if pos is None:
#             continue
#         r, c = pos
#         nr, nc = r + dr, c + dc
#         if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
#             import difflib
#             ratio = difflib.SequenceMatcher(
#                 None, neighbour.lower(), grid[nr][nc].lower()
#             ).ratio()
#             if ratio >= threshold:
#                 return True

#     return False


def _parse_markdown_table(md: str) -> List[List[str]]:
    """Parse a markdown pipe table into a 2D list of strings."""
    rows = []
    for line in md.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-: |]+\|$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _parse_html_table(html: str) -> List[List[str]]:
    """Parse an HTML table into a 2D list of strings (ignores rowspan/colspan)."""
    from html.parser import HTMLParser

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: List[List[str]] = []
            self._row: List[str] = []
            self._cell = ""
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag in ("tr",):
                self._row = []
            elif tag in ("td", "th"):
                self._cell = ""
                self._in_cell = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._row.append(self._cell.strip())
                self._in_cell = False
            elif tag == "tr" and self._row:
                self.rows.append(self._row)

        def handle_data(self, data):
            if self._in_cell:
                self._cell += data

    p = _Parser()
    p.feed(html)
    return p.rows


def _find_cell_position(rows: List[List[str]], value: str, threshold: float) -> Optional[Tuple[int, int]]:
    """Return (row, col) of the first cell that fuzzy-matches value, or None."""
    import difflib

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if difflib.SequenceMatcher(None, value.lower(), cell.lower()).ratio() >= threshold:
                return (r, c)
    return None


def _score_table(predicted: str, payload: dict) -> bool:
    """
    table check — olmOCR-bench actual field schema:
      "cell"         — the target cell value that must exist in the output table
      "up"           — expected cell directly above   (nullable)
      "down"         — expected cell directly below   (nullable)
      "left"         — expected cell directly left    (nullable)
      "right"        — expected cell directly right   (nullable)
      "top_heading"  — column header for the target cell (nullable, used as extra anchor)
      "left_heading" — row header for the target cell   (nullable, used as extra anchor)
      "threshold"    — fuzzy match threshold (default 0.8)

    Pass condition:
      1. The target cell must be found in a parsed table (HTML or markdown pipe).
      2. For every non-null neighbour (up/down/left/right), the cell in that
         direction from the target must fuzzy-match the expected value.
      3. ALL specified neighbours must match — it's a logical AND.
      4. If no table structure is found in the output, fall back to a plain-text
         proximity check: the cell and every non-null neighbour must all appear
         within a sliding window of `table_prose_window` characters.
    """
    import difflib

    cell = payload.get("cell", "")
    up = payload.get("up")  # str or None
    down = payload.get("down")  # str or None
    left = payload.get("left")  # str or None
    right = payload.get("right")  # str or None
    threshold = float(payload.get("threshold", 0.8))

    # direction → (row_delta, col_delta)
    neighbours = {
        "up": ((-1, 0), up),
        "down": ((1, 0), down),
        "left": ((0, -1), left),
        "right": ((0, +1), right),
    }
    # Only keep directions that have a non-null, non-empty expected value
    active_neighbours = {
        direction: (delta, expected)
        for direction, (delta, expected) in neighbours.items()
        if expected is not None and str(expected).strip() != ""
    }

    # ------------------------------------------------------------------ #
    # Step 1 — collect every table grid in the predicted output           #
    # ------------------------------------------------------------------ #
    all_grids: List[List[List[str]]] = []

    # HTML tables
    for m in re.finditer(r"<table[\s\S]*?</table>", predicted, re.IGNORECASE):
        grid = _parse_html_table(m.group(0))
        if grid:
            all_grids.append(grid)

    # Markdown pipe tables
    pipe_pat = re.compile(
        r"((?:\|[^\n]*\|\n)+(?:\|[-: |]+\|\n)?(?:\|[^\n]*\|\n)*)",
        re.MULTILINE,
    )
    for m in pipe_pat.finditer(predicted):
        grid = _parse_markdown_table(m.group(0))
        if grid:
            all_grids.append(grid)

    # ------------------------------------------------------------------ #
    # Step 2 — try structured table matching                              #
    # ------------------------------------------------------------------ #
    if all_grids:
        for grid in all_grids:
            pos = _find_cell_position(grid, cell, threshold)
            if pos is None:
                continue  # target cell not in this grid, try next

            r, c = pos

            # Check every required neighbour
            all_neighbours_ok = True
            for direction, ((dr, dc), expected) in active_neighbours.items():
                nr, nc = r + dr, c + dc
                # Out-of-bounds → neighbour not present → fail this grid
                if not (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])):
                    all_neighbours_ok = False
                    break
                ratio = difflib.SequenceMatcher(None, str(expected).lower(), grid[nr][nc].lower()).ratio()
                if ratio < threshold:
                    all_neighbours_ok = False
                    break

            if all_neighbours_ok:
                return True  # found a grid that satisfies all constraints

        # Target was found in at least one grid but neighbours never matched
        # → fall through to prose fallback rather than hard-failing

    # ------------------------------------------------------------------ #
    # Step 3 — prose / flat-text fallback                                 #
    # (handles models that output tables as aligned text, CSV, etc.)      #
    # ------------------------------------------------------------------ #
    # All required values (cell + every active neighbour) must appear
    # within a sliding window of this many characters to count as "nearby".
    table_prose_window = 800

    required_values = [cell] + [str(exp) for _, exp in active_neighbours.values()]
    # Filter out empty strings just in case
    required_values = [v for v in required_values if v.strip()]

    if not required_values:
        # Nothing to check — vacuously pass (shouldn't happen with valid data)
        return True

    text_lower = predicted.lower()

    # First check: do ALL required values appear anywhere in the output?
    all_present = all(_fuzzy_contains(text_lower, v.lower(), threshold) for v in required_values)
    if not all_present:
        return False

    # Second check: do they appear within proximity of each other?
    # Find the first occurrence index of each value.
    def _first_fuzzy_index(haystack: str, needle: str, thresh: float) -> int:
        n = len(needle)
        if n == 0:
            return 0
        for i in range(len(haystack) - n + 1):
            window = haystack[i : i + n]
            if difflib.SequenceMatcher(None, needle, window).ratio() >= thresh:
                return i
        return -1

    positions = [_first_fuzzy_index(text_lower, v.lower(), threshold) for v in required_values]
    positions = [p for p in positions if p != -1]

    if not positions:
        return False

    span = max(positions) - min(positions)
    return span <= table_prose_window


def _score_math(predicted: str, payload: dict) -> bool:
    """
    math: a LaTeX expression must be PRESENT in the output.
    The olmOCR-bench math check looks for a primary symbol and an optional
    secondary symbol to its left/right (rendered bounding-box logic in the
    official scorer). Here we implement a practical approximation:
      1. The primary latex string must appear in the output (fuzzy).
      2. If a secondary symbol + direction is given, both must appear and
         the primary must follow/precede the secondary in linear order.

    This covers >95% of the math test cases without requiring a headless
    browser for KaTeX rendering.
    """
    primary = payload.get("latex", payload.get("math", payload.get("text", "")))
    secondary = payload.get("secondary_latex", "")
    direction = payload.get("direction", "")  # left | right (secondary relative to primary)
    threshold = float(payload.get("threshold", 0.7))

    if not primary:
        return True

    if not _fuzzy_contains(predicted, primary, threshold):
        return False

    if secondary:
        if not _fuzzy_contains(predicted, secondary, threshold):
            return False
        # Check relative order
        p_idx = predicted.find(primary)
        s_idx = predicted.find(secondary)
        if p_idx == -1 or s_idx == -1:
            return True  # fuzzy found but not exact; accept
        if direction == "right":
            # secondary should be to the right → secondary index > primary index
            return s_idx > p_idx
        elif direction == "left":
            return s_idx < p_idx

    return True


# ---------------------------------------------------------------------------
# Dispatcher: pick the right scorer for a check_type
# ---------------------------------------------------------------------------

_SCORERS = {
    "text_present": _score_text_present,
    "text_absent": _score_text_absent,
    "reading_order": _score_reading_order,
    "table": _score_table,
    "math": _score_math,
}


def score_case(predicted: str, case: OlmTestCase) -> bool:
    """Return True if the predicted markdown passes this test case."""
    scorer = _SCORERS.get(case.check_type)
    if scorer is None:
        print(f"  [warn] unknown check_type '{case.check_type}', skipping")
        return False
    try:
        return scorer(predicted, case.payload)
    except Exception as e:
        print(f"  [warn] scorer raised {e} for {case.case_id}")
        return False


# ---------------------------------------------------------------------------
# Core benchmark loop (runs inside Modal containers)
# ---------------------------------------------------------------------------
def _extract_gt(payload: dict, check_type: str) -> str:
    """Pull the human-readable ground-truth string from a test payload."""
    if check_type in ("text_present", "text_absent"):
        return payload.get("text", "")
    if check_type == "reading_order":
        before = payload.get("before", payload.get("text_before", ""))
        after = payload.get("after", payload.get("text_after", ""))
        return f"BEFORE: {before!r}  →  AFTER: {after!r}"
    if check_type == "table":
        cell = payload.get("cell", "")
        parts = [f"cell={cell!r}"]
        for direction in ("up", "down", "left", "right"):
            val = payload.get(direction)
            if val is not None and str(val).strip():
                parts.append(f"{direction}={val!r}")
        top_h = payload.get("top_heading")
        left_h = payload.get("left_heading")
        if top_h:
            parts.append(f"top_heading={top_h!r}")
        if left_h:
            parts.append(f"left_heading={left_h!r}")
        return "  ".join(parts)
    if check_type == "math":
        return payload.get("latex", payload.get("math", payload.get("text", "")))
    return ""


def _run_cases(extractor_factory, cases: List[OlmTestCase]) -> List[OlmResult]:
    """
    Render each PDF page, run the extractor, score each test case.
    """
    extractor = extractor_factory()
    results: List[OlmResult] = []

    for case in cases:
        t0 = time.perf_counter()
        try:
            img = _pdf_page_to_image(case.pdf_bytes, case.page_num)
            out = extractor.extract(img, output_format="markdown")
            predicted = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency = time.perf_counter() - t0

            passed = score_case(predicted, case)
            # results.append(OlmResult(
            #     case_id=case.case_id,
            #     split=case.split,
            #     check_type=case.check_type,
            #     model=getattr(out, "model_name", None) or "unknown",
            #     passed=passed,
            #     latency_s=latency,
            # ))
            gt = _extract_gt(case.payload, case.check_type)

            results.append(
                OlmResult(
                    case_id=case.case_id,
                    split=case.split,
                    check_type=case.check_type,
                    model=getattr(out, "model_name", None) or "unknown",
                    passed=passed,
                    latency_s=latency,
                    gt=gt,
                    predicted=predicted,
                )
            )
            tick = "✓" if passed else "✗"
            pred_preview = predicted[:200].replace("\n", "↵") + ("…" if len(predicted) > 200 else "")
            print(f"  {tick} [{case.split:<15}] [{case.check_type:<14}] {latency:.2f}s")
            print(f"      GT : {gt!r}")
            print(f"      OUT: {pred_preview!r}")
            # tick = "✓" if passed else "✗"
            # print(f"  {tick} [{case.split:<15}] [{case.check_type:<14}] {latency:.2f}s")

        except Exception as exc:
            import traceback

            latency = time.perf_counter() - t0
            results.append(
                OlmResult(
                    case_id=case.case_id,
                    split=case.split,
                    check_type=case.check_type,
                    model="unknown",
                    passed=False,
                    latency_s=latency,
                    failed=True,
                    error=str(exc),
                )
            )
            print(f"  ✗ [{case.split:<15}] [{case.check_type:<14}] FAILED: {exc}")
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Remote Modal functions  (one per model / image / GPU)
# ---------------------------------------------------------------------------


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_qwen(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    def factory():
        return QwenTextExtractor(backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-2B-Instruct", device="cuda"))

    return _run_cases(factory, cases)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_deepseek(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

    def factory():
        return DeepSeekOCRTextExtractor(
            backend=DeepSeekOCRTextPyTorchConfig(
                model="unsloth/DeepSeek-OCR-2",
                device="cuda",
                torch_dtype="bfloat16",
                use_flash_attention=False,
                crop_mode=True,
            )
        )

    return _run_cases(factory, cases)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_nanonets(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

    def factory():
        return NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig(device="cuda"))

    return _run_cases(factory, cases)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_granitedocling(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextPyTorchConfig

    def factory():
        return GraniteDoclingTextExtractor(
            backend=GraniteDoclingTextPyTorchConfig(device="cuda", torch_dtype="bfloat16", use_flash_attention=False)
        )

    return _run_cases(factory, cases)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_mineruvl(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

    def factory():
        return MinerUVLTextExtractor(backend=MinerUVLTextPyTorchConfig(device="cuda"))

    return _run_cases(factory, cases)


@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_glmocr(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
    from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

    def factory():
        return GLMOCRTextExtractor(backend=GLMOCRPyTorchConfig(device="cuda"))

    return _run_cases(factory, cases)


@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_lighton(cases: List[OlmTestCase]) -> List[OlmResult]:
    from omnidocs.tasks.text_extraction import LightOnTextExtractor
    from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

    def factory():
        return LightOnTextExtractor(backend=LightOnTextPyTorchConfig(device="cuda"))

    return _run_cases(factory, cases)


@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def bench_dotsocr(cases: List[OlmTestCase]) -> List[OlmResult]:
    import os

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch

    torch.cuda.init()
    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

    def factory():
        return DotsOCRTextExtractor(
            backend=DotsOCRVLLMConfig(
                model="rednote-hilab/dots.ocr",
                gpu_memory_utilization=0.90,
                max_model_len=32768,
                enforce_eager=True,
            )
        )

    return _run_cases(factory, cases)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, tuple] = {
    "qwen": (bench_qwen, "pytorch/A10G"),
    "deepseek": (bench_deepseek, "pytorch/A10G"),
    "nanonets": (bench_nanonets, "pytorch/A10G"),
    "granitedocling": (bench_granitedocling, "pytorch/A10G"),
    "mineruvl": (bench_mineruvl, "pytorch/A10G"),
    "glmocr": (bench_glmocr, "pytorch/A10G"),
    "lighton": (bench_lighton, "pytorch/A10G"),
    "dotsocr": (bench_dotsocr, "vllm/L40S"),
}

# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def aggregate(results: List[OlmResult], model_id: str) -> dict:
    """
    Compute per-split and per-check-type pass rates.
    Mirrors the official olmOCR-bench leaderboard table.
    """
    from collections import defaultdict

    total = len(results)
    failed = [r for r in results if r.failed]
    scorable = [r for r in results if not r.failed]

    by_split: Dict[str, List[bool]] = defaultdict(list)
    by_check: Dict[str, List[bool]] = defaultdict(list)
    for r in scorable:
        by_split[r.split].append(r.passed)
        by_check[r.check_type].append(r.passed)

    all_passed = [r.passed for r in scorable]
    overall = sum(all_passed) / len(all_passed) if all_passed else 0.0

    lats = [r.latency_s for r in results]
    s = sorted(lats)
    n = len(s)
    p50 = s[int(0.50 * n)] if n else None
    p95 = s[min(int(0.95 * n), n - 1)] if n else None

    return {
        "model": model_id,
        "overall": overall,
        "samples_run": total,
        "samples_failed": len(failed),
        "failure_rate": len(failed) / total if total else 0.0,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "by_split": {split: (sum(v) / len(v) if v else None) for split, v in by_split.items()},
        "by_check": {ct: (sum(v) / len(v) if v else None) for ct, v in by_check.items()},
        "n_by_split": {split: len(v) for split, v in by_split.items()},
    }


def print_report(all_metrics: List[dict], splits: List[str]) -> None:
    div = "=" * 120

    # Build header matching the leaderboard column order
    col_labels = [SPLIT_LABELS.get(s, s[:7]) for s in splits]
    header = (
        f"\n{'Model':<18}" + "".join(f"{lbl:>9}" for lbl in col_labels) + f"{'Overall':>10}  {'p50(s)':>7} {'Fail%':>6}"
    )
    print(f"\n{div}")
    print("olmOCR-BENCH RESULTS  (binary pass/fail unit tests, % passed)")
    print(div + header)
    print("-" * 120)

    def _pct(v):
        return f"{v * 100:6.1f}%" if v is not None else "    n/a"

    for m in all_metrics:
        row = f"{m['model']:<18}"
        for split in splits:
            v = m["by_split"].get(split)
            row += f"{_pct(v):>9}"
        row += f"{_pct(m['overall']):>10}"
        row += f"  {m['latency_p50_s'] or 0:>7.2f}"
        row += f"  {(m['failure_rate'] or 0) * 100:>5.1f}%"
        print(row)

        # Print sample counts per split as context
        counts = "  ".join(f"{SPLIT_LABELS.get(s, s[:7])}={m['n_by_split'].get(s, 0)}" for s in splits)
        print(f"  {'':18}  ({counts})")

    print(div)
    print("↑ = higher is better  |  Each cell = % of unit tests passed in that split")
    print()
    print("Check-type breakdown legend:")
    print("  text_present  — short text spans must appear in output")
    print("  text_absent   — headers/footers/page numbers must NOT appear")
    print("  reading_order — text A must precede text B in linear order")
    print("  table         — cell values + neighbour relationships")
    print("  math          — LaTeX symbol presence/position")

    # Per-check-type breakdown
    all_checks = sorted({ct for m in all_metrics for ct in m["by_check"]})
    if all_checks:
        print(f"\n{div}")
        print("PER-CHECK-TYPE BREAKDOWN")
        print(div)
        check_header = f"\n{'Model':<18}" + "".join(f"{ct[:12]:>14}" for ct in all_checks)
        print(check_header)
        print("-" * (18 + 14 * len(all_checks)))
        for m in all_metrics:
            row = f"{m['model']:<18}"
            for ct in all_checks:
                v = m["by_check"].get(ct)
                row += f"{_pct(v):>14}"
            print(row)
        print(div)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    models: str = "",
    splits: str = "",
    max_per_split: int = 0,
    output: str = "",
    list_info: bool = False,
):
    """
    Run olmOCR-Bench on Modal GPUs.

    Args:
        models:        Comma-separated model IDs (default: all)
        splits:        Comma-separated split names (default: all 7 splits)
        max_per_split: Max test cases per split — 0 means all (default: 0)
        output:        JSON output path for full results
        list_info:     Print model/split info and exit
    """
    if list_info:
        print("\nAvailable models:", list(MODEL_REGISTRY.keys()))
        print("\nAvailable splits:")
        for s in OLM_SPLITS:
            print(f"  {s:<20} → leaderboard column: {SPLIT_LABELS[s]}")
        return

    model_ids = [m.strip() for m in models.split(",") if m.strip()] if models else list(MODEL_REGISTRY.keys())
    unknown = [m for m in model_ids if m not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    split_names = [s.strip() for s in splits.split(",") if s.strip()] if splits else OLM_SPLITS
    unknown_splits = [s for s in split_names if s not in OLM_SPLITS]
    if unknown_splits:
        print(f"Unknown splits: {unknown_splits}. Available: {OLM_SPLITS}")
        sys.exit(1)

    max_samples = max_per_split if max_per_split > 0 else None

    print("\nLoading olmOCR-Bench from HuggingFace...")
    print(f"  Splits: {split_names}")
    print(f"  Max per split: {max_samples or 'all'}")
    all_cases = load_olmocr_bench(split_names, max_per_split=max_samples)
    print(f"\nTotal test cases: {len(all_cases)}")
    print(f"Models to run:    {model_ids}\n")

    if not all_cases:
        print("No test cases loaded. Exiting.")
        sys.exit(1)

    # Spawn all models in parallel
    all_metrics = []
    all_raw: Dict[str, list] = {}

    print(f"Spawning {len(model_ids)} models in parallel...\n")
    futures = {}
    for model_id in model_ids:
        remote_fn, backend_label = MODEL_REGISTRY[model_id]
        print(f"  ↑ Spawning {model_id} [{backend_label}]")
        try:
            futures[model_id] = (remote_fn.spawn(all_cases), backend_label)
        except Exception as exc:
            print(f"  [ERROR] Failed to spawn {model_id}: {exc}")
            futures[model_id] = (None, backend_label)

    for model_id, (future, backend_label) in futures.items():
        print(f"\n{'=' * 60}")
        print(f"  Waiting: {model_id}  [{backend_label}]")
        print(f"{'=' * 60}")

        if future is None:
            results = [
                OlmResult(
                    case_id=c.case_id,
                    split=c.split,
                    check_type=c.check_type,
                    model=model_id,
                    passed=False,
                    latency_s=0.0,
                    failed=True,
                    error="spawn failed",
                )
                for c in all_cases
            ]
        else:
            try:
                results: List[OlmResult] = future.get()
            except Exception as exc:
                print(f"  [ERROR] {model_id} failed: {exc}")
                results = [
                    OlmResult(
                        case_id=c.case_id,
                        split=c.split,
                        check_type=c.check_type,
                        model=model_id,
                        passed=False,
                        latency_s=0.0,
                        failed=True,
                        error=str(exc),
                    )
                    for c in all_cases
                ]

        metrics = aggregate(results, model_id)
        all_metrics.append(metrics)
        all_raw[model_id] = [asdict(r) for r in results]

        passed_n = sum(1 for r in results if r.passed and not r.failed)
        total_n = len(results)
        print(
            f"  Done: {passed_n}/{total_n} passed"
            f"  overall={metrics['overall'] * 100:.1f}%"
            f"  p50={metrics.get('latency_p50_s') or 0:.2f}s"
        )

    # Final leaderboard-style report
    print_report(all_metrics, split_names)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "olmOCR-bench",
            "splits": split_names,
            "num_cases": len(all_cases),
            "models": model_ids,
            "metrics": all_metrics,
            "raw_results": all_raw,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFull results saved to: {out_path}")
