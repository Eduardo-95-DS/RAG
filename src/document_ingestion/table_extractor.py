"""Table-to-prose conversion for financial PDF tables.

PyPDFLoader extracts financial tables as raw whitespace-separated text, which
breaks both FAISS (semantic embedding mismatch) and BM25 (no lexical overlap)
for queries that use different vocabulary than the table (e.g. "R&D spending"
vs. "research and development expenses $ 12,914"). This module detects table
regions on a page (via pdfplumber) and converts each row into a natural
language sentence that both retrievers can match against.

Design notes (see reference/rag_production_2_plan.md for the full writeup):
- Headers (e.g. "Jan 26, 2025") and units (e.g. "$ in millions") are detected
  once per table, not per row, since financial statement tables can run
  20+ rows under a single header block.
- A table can switch units partway through (e.g. an income statement table
  that ends in per-share and share-count rows). Sub-header rows with no
  numeric values (e.g. "Net income per share:") are used as unit-mode
  switches for the rows that follow.
- Original number formatting (commas, decimals) is preserved rather than
  normalized, since retrieval eval keywords match on exact strings like
  "12,914".
- If a table or row can't be confidently parsed, it's skipped rather than
  guessing -- the caller is expected to append this prose to the original
  extracted text, not replace it, so nothing is lost on a parse failure.
"""

import re
from typing import List, Optional

DATE_RE = re.compile(r"[A-Z][a-z]{2}\.?\s+\d{1,2},\s+\d{4}")
MILLIONS_RE = re.compile(r"in\s+millions", re.IGNORECASE)
BILLIONS_RE = re.compile(r"in\s+billions", re.IGNORECASE)
THOUSANDS_RE = re.compile(r"in\s+thousands", re.IGNORECASE)
PERCENT_TABLE_RE = re.compile(r"(as a |expressed as a )?percentage of (net )?revenue", re.IGNORECASE)
# Require a leading digit so a bare comma/paren in a label (e.g. "Sales,") can't
# be mistaken for a value token.
VALUE_RE = re.compile(r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?\s*%?|—")

UNIT_SCALE_PATTERNS = [
    (THOUSANDS_RE, "thousand"),
    (MILLIONS_RE, "million"),
    (BILLIONS_RE, "billion"),
]


def _clean_value(token: str):
    """Strip $ / parens / % but preserve comma and decimal formatting.

    Returns (value_str, is_percent) or (None, False) if not a real value.
    """
    token = token.strip()
    if token in ("—", ""):
        return None, False
    is_pct = "%" in token
    is_neg = token.startswith("(") and token.endswith(")")
    num = re.sub(r"[^\d,.]", "", token)
    if not num:
        return None, False
    return ("-" if is_neg else "") + num, is_pct


def _parse_row(row_text: str):
    """Split a flattened table row into (label, [(value, is_percent), ...])."""
    tokens = VALUE_RE.findall(row_text)
    if not tokens:
        return row_text.strip(), []
    label = row_text
    for t in reversed(tokens):
        idx = label.rfind(t)
        if idx == -1:
            break
        label = label[:idx]
    label = label.strip().rstrip("$").strip()
    values = []
    for t in tokens:
        v, is_pct = _clean_value(t)
        if v is not None:
            values.append((v, is_pct))
    return label, values


def _detect_unit_scale(text: str) -> str:
    for pattern, scale in UNIT_SCALE_PATTERNS:
        if pattern.search(text):
            return scale
    return ""


def _row_to_prose(label, values, headers, unit_mode, unit_suffix) -> Optional[str]:
    """unit_mode: 'money' | 'percent' | 'per_share' | 'shares'"""
    if not label or not values:
        return None
    hdrs = headers or [f"period {i + 1}" for i in range(len(values))]
    parts = []
    for (val, row_is_pct), header in zip(values, hdrs):
        if row_is_pct or unit_mode == "percent":
            parts.append(f"{val}% in the period ended {header}")
        elif unit_mode == "per_share":
            parts.append(f"${val} per share in the period ended {header}")
        elif unit_mode == "shares":
            scale = f"{unit_suffix} " if unit_suffix else ""
            parts.append(f"{val} {scale}shares in the period ended {header}")
        else:  # money
            suffix = f" {unit_suffix}" if unit_suffix else ""
            parts.append(f"${val}{suffix} in the period ended {header}")
    if not parts:
        return None
    return f"{label} was " + ", ".join(parts) + "."


def _table_header_context(page, region_top: float, region_bottom: float) -> str:
    """Text strictly between two y-coordinates on the page (the header zone above a table)."""
    if region_bottom <= region_top:
        return ""
    cropped = page.within_bbox((0, region_top, page.width, region_bottom))
    return cropped.extract_text() or ""


def tables_to_prose(page) -> str:
    """Detect all tables on a pdfplumber page and return their prose conversion.

    Returns an empty string if no tables are found or none could be parsed.
    Caller is responsible for appending this to the page's original extracted
    text -- this function never replaces or judges the raw text.
    """
    found = page.find_tables()
    if not found:
        return ""

    full_page_text = page.extract_text() or ""
    prose_lines: List[str] = []
    prev_bottom = 0.0

    for table in found:
        _, top, _, bottom = table.bbox
        header_ctx = _table_header_context(page, prev_bottom, top)
        prev_bottom = bottom

        rows = table.extract()
        if not rows:
            continue

        first_cell = rows[0][0] if rows[0] else ""
        cutoff = full_page_text.find(first_cell) if first_cell else -1
        wide_ctx = full_page_text[:cutoff] if cutoff != -1 else full_page_text
        scan_text = f"{header_ctx}\n{wide_ctx}"

        headers = DATE_RE.findall(header_ctx)
        base_unit_suffix = _detect_unit_scale(scan_text)
        table_is_pct = bool(PERCENT_TABLE_RE.search(scan_text))

        unit_mode = "percent" if table_is_pct else "money"
        unit_suffix = base_unit_suffix

        for row in rows:
            row_text = " ".join(c for c in row if c).strip()
            if not row_text:
                continue
            label, values = _parse_row(row_text)

            if not values:
                # Sub-header row (no numbers) -- use it to update the unit mode
                # for the rows that follow, e.g. "Net income per share:" or
                # "Weighted average shares used in per share computation:".
                low = label.lower()
                if "weighted average shares" in low or "shares used" in low:
                    unit_mode, unit_suffix = "shares", base_unit_suffix
                elif "per share" in low:
                    unit_mode, unit_suffix = "per_share", ""
                else:
                    unit_mode = "percent" if table_is_pct else "money"
                    unit_suffix = base_unit_suffix
                continue

            prose = _row_to_prose(label, values, headers, unit_mode, unit_suffix)
            if prose:
                prose_lines.append(prose)

    return "\n".join(prose_lines)
