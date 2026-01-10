from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RagHit:
    source: str
    snippet: str
    score: float


def _load_kb_docs(kb_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in sorted(kb_dir.glob("*.md")):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs


def retrieve_guidance(query: str, kb_dir: str = "docs/survey_methodology_kb", top_k: int = 3) -> List[RagHit]:
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return []

    docs = _load_kb_docs(kb_path)
    if not docs:
        return []

    names = [d[0] for d in docs]
    texts = [d[1] for d in docs]

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    q = vec.transform([query])

    scores = (X @ q.T).toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]

    hits: list[RagHit] = []
    for i in top_idx:
        snippet_lines = texts[i].strip().splitlines()
        snippet = "\n".join(snippet_lines[:18])
        hits.append(RagHit(source=names[i], snippet=snippet, score=float(scores[i])))

    return hits
