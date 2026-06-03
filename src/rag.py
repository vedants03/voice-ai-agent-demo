"""Local vector RAG over the precomputed PDF index.

The corpus embeddings are built once, offline, by ``ingest.py``. At runtime we
only embed the (short) user query and do an in-memory cosine search, so per-turn
latency stays minimal. No external embedding service is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

# Multilingual embeddings so a Hindi query can match the English document
# (cross-lingual retrieval). Small + fast; no query/passage prefixes needed.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

_model: TextEmbedding | None = None
_chunks: list[dict] | None = None
_emb: np.ndarray | None = None


def get_model() -> TextEmbedding:
    """Load the embedding model once and reuse it across turns."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize, so cosine similarity is a plain dot product."""
    vecs = np.asarray(list(get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def load_index() -> tuple[list[dict], np.ndarray]:
    """Load chunks + embeddings once (lazily)."""
    global _chunks, _emb
    if _chunks is None or _emb is None:
        if not EMB_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Index not found in {INDEX_DIR}. Run: uv run src/ingest.py"
            )
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        _emb = np.load(EMB_PATH)
    return _chunks, _emb


def warmup() -> None:
    """Pre-load model + index so the first user turn isn't slow."""
    load_index()
    embed_texts(["warmup"])


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return the top-k most relevant chunks for ``query``."""
    chunks, emb = load_index()
    qv = embed_texts([query])[0]
    sims = emb @ qv
    top = np.argsort(-sims)[:k]
    return [{**chunks[i], "score": float(sims[i])} for i in top]


def format_context(results: list[dict]) -> str:
    """Render retrieved chunks into a compact block for the LLM prompt."""
    lines = []
    for r in results:
        tag = f"(page {r.get('page', '?')}, {r.get('type', 'prose')})"
        lines.append(f"- {tag} {r['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone sanity check: prints top chunks for a few EN/HI/table queries.
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    queries = sys.argv[1:] or [
        "What is the minimum and maximum entry age?",
        "policy term and premium payment term options",
        "इस पॉलिसी में मृत्यु पर क्या लाभ मिलता है?",  # death benefit (Hindi)
        "annual premium for a 30 year old",
    ]
    for q in queries:
        print(f"\n### QUERY: {q}")
        for r in retrieve(q, k=3):
            print(f"  [{r['score']:.3f}] p{r['page']} {r['type']}: {r['text'][:160]}")
