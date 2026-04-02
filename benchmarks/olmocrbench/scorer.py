"""
benchmarks/olmocrbench/scorer.py

Pure-Python implementation of the five olmOCR-bench check types.
No external dependencies beyond stdlib difflib.

Check types:
    text_present   — fuzzy substring match (optional first_n / last_n constraint)
    text_absent    — text must NOT appear in output
    reading_order  — text A must appear before text B
    table          — cell value + neighbour relationship checks
    math           — LaTeX symbol presence / relative position

All functions follow the signature: fn(predicted: str, payload: dict) -> bool
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------


def _fuzzy_contains(text: str, query: str, threshold: float = 0.8) -> bool:
    """
    Return True if `query` appears somewhere in `text` with at least
    `threshold` SequenceMatcher similarity.
    Falls back to a simple substring check when threshold >= 1.0.
    """
    if not query:
        return True
    if threshold >= 1.0:
        return query in text

    import difflib

    q_len = len(query)
    if q_len == 0:
        return True

    for start in range(max(1, len(text) - q_len + 1)):
        window = text[start : start + q_len]
        if difflib.SequenceMatcher(None, query, window).ratio() >= threshold:
            return True
    return False


def _get_search_region(text: str, payload: dict) -> str:
    """Apply first_n / last_n constraints from the payload."""
    first_n = payload.get("first_n")
    last_n = payload.get("last_n")
    if first_n is not None:
        return text[: int(first_n)]
    if last_n is not None:
        return text[-int(last_n) :]
    return text


def _first_fuzzy_index(haystack: str, needle: str, threshold: float) -> int:
    """Return the start index of the first fuzzy match, or -1."""
    import difflib

    n = len(needle)
    if n == 0:
        return 0
    for i in range(len(haystack) - n + 1):
        window = haystack[i : i + n]
        if difflib.SequenceMatcher(None, needle, window).ratio() >= threshold:
            return i
    return -1


# ---------------------------------------------------------------------------
# text_present
# ---------------------------------------------------------------------------


def score_text_present(predicted: str, payload: dict) -> bool:
    """The query string MUST appear in the output."""
    query = payload.get("text", "")
    threshold = float(payload.get("threshold", 0.8))
    region = _get_search_region(predicted, payload)

    if not payload.get("case_sensitive", True):
        region = region.lower()
        query = query.lower()

    return _fuzzy_contains(region, query, threshold)


# ---------------------------------------------------------------------------
# text_absent
# ---------------------------------------------------------------------------


def score_text_absent(predicted: str, payload: dict) -> bool:
    """The query string must NOT appear in the output."""
    query = payload.get("text", "")
    threshold = float(payload.get("threshold", 0.8))
    region = _get_search_region(predicted, payload)

    # text_absent is not case-sensitive per olmOCR-bench spec
    return not _fuzzy_contains(region.lower(), query.lower(), threshold)


# ---------------------------------------------------------------------------
# reading_order
# ---------------------------------------------------------------------------


def score_reading_order(predicted: str, payload: dict) -> bool:
    """text_before must appear before text_after in the output."""
    before = payload.get("before", "")
    after = payload.get("after", "")
    threshold = float(payload.get("threshold", 0.8))

    if not before or not after:
        return True

    text = predicted
    if not payload.get("case_sensitive", True):
        text = text.lower()
        before = before.lower()
        after = after.lower()

    pos_before = _first_fuzzy_index(text, before, threshold)
    pos_after = _first_fuzzy_index(text, after, threshold)

    if pos_before == -1 or pos_after == -1:
        return False
    return pos_before < pos_after


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def _parse_markdown_table(md: str) -> List[List[str]]:
    """Parse a markdown pipe table into a 2-D list of cell strings."""
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
    """Parse an HTML table into a 2-D list of cell strings."""

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: List[List[str]] = []
            self._row: List[str] = []
            self._cell = ""
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
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
    """Return (row, col) of the first fuzzy-matching cell, or None."""
    import difflib

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if difflib.SequenceMatcher(None, value.lower(), cell.lower()).ratio() >= threshold:
                return (r, c)
    return None


def score_table(predicted: str, payload: dict) -> bool:
    """
    Pass if the target cell is found in a parsed table AND every specified
    neighbour (up/down/left/right) matches.  Falls back to a flat-text
    proximity check for models that don't emit structured tables.
    """
    import difflib

    cell = payload.get("cell", "")
    up = payload.get("up")
    down = payload.get("down")
    left = payload.get("left")
    right = payload.get("right")
    threshold = float(payload.get("threshold", 0.8))

    neighbours = {
        "up": ((-1, 0), up),
        "down": ((1, 0), down),
        "left": ((0, -1), left),
        "right": ((0, +1), right),
    }
    active = {d: (delta, exp) for d, (delta, exp) in neighbours.items() if exp is not None and str(exp).strip() != ""}

    # Collect all grids (HTML + markdown)
    grids: List[List[List[str]]] = []
    for m in re.finditer(r"<table[\s\S]*?</table>", predicted, re.IGNORECASE):
        g = _parse_html_table(m.group(0))
        if g:
            grids.append(g)

    pipe_pat = re.compile(
        r"((?:\|[^\n]*\|\n)+(?:\|[-: |]+\|\n)?(?:\|[^\n]*\|\n)*)",
        re.MULTILINE,
    )
    for m in pipe_pat.finditer(predicted):
        g = _parse_markdown_table(m.group(0))
        if g:
            grids.append(g)

    if grids:
        for grid in grids:
            pos = _find_cell_position(grid, cell, threshold)
            if pos is None:
                continue
            r, c = pos
            ok = True
            for _d, ((dr, dc), exp) in active.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])):
                    ok = False
                    break
                if difflib.SequenceMatcher(None, str(exp).lower(), grid[nr][nc].lower()).ratio() < threshold:
                    ok = False
                    break
            if ok:
                return True

    # Prose fallback: all required values must appear within 800 chars of each other
    window = 800
    required = [cell] + [str(exp) for _, exp in active.values()]
    required = [v for v in required if v.strip()]
    if not required:
        return True

    text_lower = predicted.lower()
    if not all(_fuzzy_contains(text_lower, v.lower(), threshold) for v in required):
        return False

    positions = [_first_fuzzy_index(text_lower, v.lower(), threshold) for v in required]
    positions = [p for p in positions if p != -1]
    if not positions:
        return False

    return (max(positions) - min(positions)) <= window


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def score_math(predicted: str, payload: dict) -> bool:
    """
    The primary LaTeX expression must appear in the output.
    If a secondary symbol + direction is given, both must appear and
    their relative order must match.
    """
    primary = payload.get("latex", payload.get("math", payload.get("text", "")))
    secondary = payload.get("secondary_latex", "")
    direction = payload.get("direction", "")
    threshold = float(payload.get("threshold", 0.7))

    if not primary:
        return True

    if not _fuzzy_contains(predicted, primary, threshold):
        return False

    if secondary:
        if not _fuzzy_contains(predicted, secondary, threshold):
            return False
        p_idx = predicted.find(primary)
        s_idx = predicted.find(secondary)
        if p_idx != -1 and s_idx != -1:
            if direction == "right":
                return s_idx > p_idx
            elif direction == "left":
                return s_idx < p_idx

    return True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SCORERS = {
    "text_present": score_text_present,
    "text_absent": score_text_absent,
    "reading_order": score_reading_order,
    "table": score_table,
    "math": score_math,
}


def score_case(predicted: str, check_type: str, payload: dict, case_id: str = "") -> bool:
    """Dispatch to the correct scorer for this check_type."""
    scorer = _SCORERS.get(check_type)
    if scorer is None:
        print(f"  [warn] unknown check_type '{check_type}', skipping" + (f" ({case_id})" if case_id else ""))
        return False
    try:
        return scorer(predicted, payload)
    except Exception as exc:
        print(f"  [warn] scorer raised {exc}" + (f" for {case_id}" if case_id else ""))
        return False
