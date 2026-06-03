"""Offline ingest: PDF -> table-aware chunks -> embeddings -> index/.

The brochure's accuracy-critical content (entry/maturity age, policy & premium
terms, premium-rate grids, benefit illustrations) lives in TABLES. A raw text
dump scrambles row/column relationships and yields wrong numbers, so tables are
extracted structurally and each row is linearized into a self-describing
sentence that inlines its column headers and the table caption. Prose is chunked
separately. Run once: ``uv run src/ingest.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The brochure contains ✓, –, • and other non-ASCII glyphs; make stdout UTF-8 so
# the spot-check prints don't crash on the Windows console (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pdfplumber

from rag import INDEX_DIR, embed_texts

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "flexi-term-pro.pdf"

PROSE_CHUNK_SIZE = 700
PROSE_OVERLAP = 120


def _clean(cell) -> str:
    return re.sub(r"\s+", " ", (cell or "").strip())


def find_caption(page, bbox, pageno: int) -> str:
    """Text line immediately above the table — used to make rows self-describing."""
    _x0, top, _x1, _bottom = bbox
    words = [w for w in page.extract_words() if w["bottom"] <= top + 1]
    if not words:
        return f"Table on page {pageno}"
    nearest = max(w["bottom"] for w in words)
    line = sorted(
        (w for w in words if nearest - w["bottom"] <= 4), key=lambda w: w["x0"]
    )
    caption = _clean(" ".join(w["text"] for w in line))
    return caption or f"Table on page {pageno}"


def extract_table_chunks(page, pageno: int) -> tuple[list[dict], list[tuple]]:
    """Linearize each table row into one retrievable chunk. Returns (chunks, bboxes)."""
    chunks: list[dict] = []
    tables = page.find_tables()
    for table in tables:
        rows = [[_clean(c) for c in row] for row in (table.extract() or [])]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        caption = find_caption(page, table.bbox, pageno)
        # Treat the first row as a header if it has multiple labeled columns.
        header = rows[0]
        has_header = len(rows) > 1 and sum(1 for c in header if c) >= 2
        data_rows = rows[1:] if has_header else rows
        for row in data_rows:
            parts = []
            for i, cell in enumerate(row):
                if not cell:
                    continue
                col = header[i] if has_header and i < len(header) else ""
                parts.append(f"{col}: {cell}" if col and col != cell else cell)
            if not parts:
                continue
            body = "; ".join(parts)
            text = f"[{caption}] {body}" if caption else body
            chunks.append(
                {"text": text, "page": pageno, "type": "table", "caption": caption}
            )
    return chunks, [t.bbox for t in tables]


def prose_outside_tables(page, table_bboxes: list[tuple]) -> str:
    """Extract page text, excluding anything inside a detected table region."""
    if not table_bboxes:
        return page.extract_text() or ""

    def keep(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(
            x0 <= cx <= x1 and top <= cy <= bottom
            for (x0, top, x1, bottom) in table_bboxes
        )

    return page.filter(keep).extract_text() or ""


def chunk_prose(text: str, pageno: int) -> list[dict]:
    """Paragraph-aware chunking with overlap for continuity."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= PROSE_CHUNK_SIZE:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= PROSE_CHUNK_SIZE:
            buf = para
        else:
            step = PROSE_CHUNK_SIZE - PROSE_OVERLAP
            for i in range(0, len(para), step):
                chunks.append(para[i : i + PROSE_CHUNK_SIZE])
            buf = ""
    if buf:
        chunks.append(buf)
    return [
        {"text": c, "page": pageno, "type": "prose", "caption": ""} for c in chunks
    ]


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    chunks: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            table_chunks, bboxes = extract_table_chunks(page, pageno)
            chunks.extend(table_chunks)
            chunks.extend(chunk_prose(prose_outside_tables(page, bboxes), pageno))

    chunks = [c for c in chunks if c["text"].strip()]
    if not chunks:
        raise RuntimeError("No text extracted from PDF — check the source file.")

    print(f"Embedding {len(chunks)} chunks with the multilingual model...")
    emb = embed_texts([c["text"] for c in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_table = sum(1 for c in chunks if c["type"] == "table")
    n_prose = len(chunks) - n_table
    print(f"Wrote {len(chunks)} chunks ({n_prose} prose, {n_table} table) -> {INDEX_DIR}")

    print("\n--- sample TABLE chunks (spot-check against the PDF) ---")
    for c in [c for c in chunks if c["type"] == "table"][:6]:
        print(f"[p{c['page']}] {c['text'][:180]}")
    print("\n--- sample PROSE chunks ---")
    for c in [c for c in chunks if c["type"] == "prose"][:3]:
        print(f"[p{c['page']}] {c['text'][:160]}")


if __name__ == "__main__":
    main()
