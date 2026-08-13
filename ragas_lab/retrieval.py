"""
retrieval.py — chunking, retrieval, and re-ranking.

Everything here is pure standard library. No numpy, no Azure, no network.
That is deliberate: you can read the whole retrieval path in one sitting,
and the mechanics are identical to what Azure AI Search does for you.

WHAT YOU ARE COMPARING
----------------------
CHUNKERS
  fixed_256 / fixed_512 / fixed_1024   the curriculum's three sizes
  structure                            split on the document's own boundaries
                                       (clause numbers, blank lines)

RETRIEVERS
  keyword    term frequency, inverse document frequency. Finds clause numbers
             and exact identifiers. Blind to meaning.
  vector     cosine similarity over a bag-of-words vector. Finds meaning.
             Blind to a clause number it has never seen phrased that way.
  hybrid     both, scores combined. This is what ADR-2 locked in Sprint 0.

RE-RANKER
  A second pass over a wide candidate set. Retrieve wide, pass narrow.

NOTE ON THE "VECTOR" RETRIEVER
------------------------------
A real vector search uses learned embeddings. This one uses TF-IDF cosine,
which is a bag-of-words approximation. It behaves like a vector search for
the purposes of this lab — it rewards overlap in meaning-bearing words and
ignores exact identifiers — but it will not capture synonymy the way a real
embedding model does. Where that matters is called out in the report.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "by", "with", "as", "that", "this",
    "it", "from", "any", "not", "no", "which", "where", "under", "does", "do",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9,\.\-]*", text.lower()) if t not in STOP]


def approx_tokens(text: str) -> int:
    """Rough token count. Real tokenisers differ; this is close enough to
    compare chunk sizes against each other, which is what the lab needs."""
    return max(1, int(len(text) / 4))


@dataclass
class Chunk:
    id: str
    doc: str
    corpus: str
    text: str
    n_tokens: int


# --------------------------------------------------------------------------
# CHUNKERS
# --------------------------------------------------------------------------
def chunk_fixed(doc: str, corpus: str, text: str, target_tokens: int) -> list[Chunk]:
    """Fixed-size chunking with a small overlap. The naive approach — and the
    one that split a clause across a boundary in Session 2."""
    target_chars = target_tokens * 4
    overlap = target_chars // 8
    out, i, n = [], 0, 0
    while i < len(text):
        piece = text[i:i + target_chars]
        out.append(Chunk(f"{doc}#f{target_tokens}-{n}", doc, corpus, piece.strip(),
                         approx_tokens(piece)))
        i += max(1, target_chars - overlap)
        n += 1
    return out


CLAUSE_RE = re.compile(r"^(CLAUSE\s+[\d\.]+|[A-Z][A-Z \-]{4,})\s*$", re.MULTILINE)


def chunk_structure(doc: str, corpus: str, text: str) -> list[Chunk]:
    """Split on the document's own boundaries: clause headings, or blank lines.
    A clause never ends up half in one chunk and half in another."""
    marks = [m.start() for m in CLAUSE_RE.finditer(text)]
    if len(marks) >= 2:
        bounds = marks + [len(text)]
        pieces = [text[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    else:
        pieces = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Chunk(f"{doc}#s-{i}", doc, corpus, p.strip(), approx_tokens(p))
            for i, p in enumerate(pieces) if p.strip()]


CHUNKERS = {
    "fixed_256":  lambda d, c, t: chunk_fixed(d, c, t, 256),
    "fixed_512":  lambda d, c, t: chunk_fixed(d, c, t, 512),
    "fixed_1024": lambda d, c, t: chunk_fixed(d, c, t, 1024),
    "structure":  chunk_structure,
}


def build_chunks(docs: dict[str, tuple[str, str]], strategy: str) -> list[Chunk]:
    fn = CHUNKERS[strategy]
    out: list[Chunk] = []
    for name, (corpus, text) in docs.items():
        out.extend(fn(name, corpus, text))
    return out


# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.toks = [tokenize(c.text) for c in chunks]
        self.tf = [Counter(t) for t in self.toks]
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        n = max(1, len(chunks))
        self.idf = {w: math.log(1 + n / (1 + c)) for w, c in df.items()}
        self._norms = [self._norm(i) for i in range(len(chunks))]

    def _vec(self, i: int) -> dict[str, float]:
        return {w: f * self.idf.get(w, 0.0) for w, f in self.tf[i].items()}

    def _norm(self, i: int) -> float:
        return math.sqrt(sum(v * v for v in self._vec(i).values())) or 1e-9

    # ---- keyword: idf-weighted term overlap, rewards exact identifiers ----
    def keyword_scores(self, query: str) -> list[float]:
        q = tokenize(query)
        qset = Counter(q)
        out = []
        for i in range(len(self.chunks)):
            s = 0.0
            for w, qf in qset.items():
                if w in self.tf[i]:
                    s += qf * self.idf.get(w, 0.0) * (1 + math.log(self.tf[i][w]))
            out.append(s)
        return out

    # ---- vector: TF-IDF cosine, rewards overlap in meaning-bearing words ----
    def vector_scores(self, query: str) -> list[float]:
        q = Counter(tokenize(query))
        qv = {w: f * self.idf.get(w, 0.0) for w, f in q.items()}
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1e-9
        out = []
        for i in range(len(self.chunks)):
            dv = self._vec(i)
            dot = sum(v * dv.get(w, 0.0) for w, v in qv.items())
            out.append(dot / (qn * self._norms[i]))
        return out

    def search(self, query: str, mode: str, top_k: int) -> list[tuple[Chunk, float]]:
        if mode == "keyword":
            sc = self.keyword_scores(query)
        elif mode == "vector":
            sc = self.vector_scores(query)
        elif mode == "hybrid":
            k, v = self.keyword_scores(query), self.vector_scores(query)
            mk, mv = max(k) or 1e-9, max(v) or 1e-9
            sc = [0.5 * (a / mk) + 0.5 * (b / mv) for a, b in zip(k, v)]
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")
        ranked = sorted(zip(self.chunks, sc), key=lambda x: -x[1])
        return ranked[:top_k]


# --------------------------------------------------------------------------
# RE-RANKING — retrieve wide, then pass narrow.
#
# A real re-ranker is a cross-encoder that scores (query, passage) pairs
# together. This one approximates that by rewarding chunks that contain the
# query's rarest terms AND are not mostly filler. Same shape, no model.
# --------------------------------------------------------------------------
def rerank(query: str, candidates: list[tuple[Chunk, float]], idf: dict[str, float],
           keep: int) -> list[tuple[Chunk, float]]:
    q = set(tokenize(query))
    rare = sorted(q, key=lambda w: -idf.get(w, 0.0))[:6]
    out = []
    for ch, base in candidates:
        toks = tokenize(ch.text)
        tset = set(toks)
        covered = sum(1 for w in rare if w in tset) / max(1, len(rare))
        density = sum(1 for w in toks if w in q) / max(1, len(toks))
        out.append((ch, 0.5 * base + 0.35 * covered + 0.15 * min(1.0, density * 8)))
    return sorted(out, key=lambda x: -x[1])[:keep]


# --------------------------------------------------------------------------
# QUERY DECOMPOSITION
#
# A multi-part question retrieved as one string gets a candidate set that is
# a compromise between its parts, and often good for neither. Split it, run
# each sub-query, then merge.
# --------------------------------------------------------------------------
SPLIT_RE = re.compile(r"\s*(?:,\s*and\s+|\s+and\s+|;\s*)", re.IGNORECASE)


def decompose(question: str) -> list[str]:
    parts = [p.strip(" ?.") for p in SPLIT_RE.split(question) if p.strip(" ?.")]
    if len(parts) < 2:
        return [question]
    head = parts[0]
    # Carry the interrogative from the first part into later fragments that
    # lost it in the split ("does the supplementary work need...").
    out = [head]
    for p in parts[1:]:
        out.append(p if len(p.split()) > 3 else f"{head} {p}")
    return out


def retrieve(index: Index, question: str, mode: str, top_k: int,
             use_rerank: bool = False, use_decomp: bool = False,
             wide_k: int | None = None) -> list[Chunk]:
    wide = wide_k or (top_k * 3 if use_rerank else top_k)
    queries = decompose(question) if use_decomp else [question]

    pool: dict[str, tuple[Chunk, float]] = {}
    for q in queries:
        for ch, sc in index.search(q, mode, wide):
            prev = pool.get(ch.id)
            if prev is None or sc > prev[1]:
                pool[ch.id] = (ch, sc)

    cands = sorted(pool.values(), key=lambda x: -x[1])[:max(wide, top_k)]
    if use_rerank:
        cands = rerank(question, cands, index.idf, top_k)
    return [c for c, _ in cands[:top_k]]
