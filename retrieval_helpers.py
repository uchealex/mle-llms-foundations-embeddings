"""Shared local retrieval helpers for the notebook exercises."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z]+(?:'[A-Za-z]+)?  # Words and contractions
    |\d+(?:\.\d+)?            # Integers and decimals
    |[^\w\s]                  # Punctuation and symbols
    """,
    re.VERBOSE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "whether",
    "why",
    "with",
}

SparseVector = dict[str, float]


@dataclass(frozen=True)
class CostEstimate:
    """Request-level cost estimate split by input and output tokens."""

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass(frozen=True)
class EmbeddedCorpus:
    """Fitted sparse vectors plus the source records they represent."""

    records: list[dict[str, str]]
    text_field: str
    idf: dict[str, float]
    vectors: list[SparseVector]


@dataclass(frozen=True)
class SearchResult:
    """Readable retrieval candidate returned by the local sparse index."""

    rank: int
    score: float
    lesson_id: str
    title: str
    module: str
    description: str


def simple_tokenize(text: str) -> list[str]:
    """Split text into approximate tokens."""
    return TOKEN_PATTERN.findall(text)


def count_tokens(text: str) -> int:
    """Return the approximate token count used by the local exercises."""
    return len(simple_tokenize(text))


def estimate_request_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> CostEstimate:
    """Estimate request cost from token counts and per-million-token prices."""
    input_cost = input_tokens / 1_000_000 * input_price_per_million
    output_cost = output_tokens / 1_000_000 * output_price_per_million
    return CostEstimate(input_tokens, output_tokens, input_cost, output_cost)


def format_usd(value: float) -> str:
    """Format small request costs consistently for comparison tables."""
    return f"${value:.6f}"


def content_tokens(text: str) -> list[str]:
    """Return comparable lowercase tokens after removing low-signal words."""
    tokens = [token.lower() for token in simple_tokenize(text)]
    return [
        token
        for token in tokens
        if token not in STOPWORDS and any(char.isalnum() for char in token)
    ]


def combine_text_fields(record: dict[str, str], fields: list[str]) -> str:
    """Join selected metadata fields into the text that will be vectorized."""
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"Missing expected fields: {missing}")
    return " ".join(record.get(field, "") for field in fields)


def fit_idf(texts: list[str]) -> dict[str, float]:
    """Learn how informative each token is across the corpus."""
    document_count = len(texts)
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update(set(content_tokens(text)))
    return {
        token: math.log((1 + document_count) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> SparseVector:
    """Convert text into a normalized TF-IDF vector for cosine comparison."""
    tokens = content_tokens(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    vector = {
        token: (count / total) * idf.get(token, 0.0)
        for token, count in counts.items()
        if token in idf
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {token: value / norm for token, value in vector.items()}


def embed_corpus(records: list[dict[str, str]], text_field: str) -> EmbeddedCorpus:
    """Fit vector weights once, then vectorize every record with the same weights."""
    if not records:
        raise ValueError("Cannot embed an empty corpus.")
    texts = [record[text_field] for record in records]
    idf = fit_idf(texts)
    vectors = [vectorize(text, idf) for text in texts]
    return EmbeddedCorpus(
        records=records, text_field=text_field, idf=idf, vectors=vectors
    )


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Return a directional similarity score for two sparse vectors."""
    if not left or not right:
        return 0.0
    shared = set(left).intersection(right)
    return sum(left[token] * right[token] for token in shared)


def similarity_scores(corpus: EmbeddedCorpus, query: str) -> list[float]:
    """Score one query against every document vector in the fitted corpus."""
    query_vector = vectorize(query, corpus.idf)
    return [cosine_similarity(query_vector, vector) for vector in corpus.vectors]


class LocalRetrievalIndex:
    """Reusable local sparse retrieval boundary for notebook exercises."""

    def __init__(self, corpus: EmbeddedCorpus):
        self.corpus = corpus

    @classmethod
    def from_csv(
        cls, path: str | Path, text_fields: list[str] | None = None
    ) -> LocalRetrievalIndex:
        with Path(path).open(newline="", encoding="utf-8") as file:
            records = list(csv.DictReader(file))
        fields = text_fields or ["title", "module", "description"]
        for record in records:
            record["search_text"] = combine_text_fields(record, fields)
        return cls(embed_corpus(records, text_field="search_text"))

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Return ranked local sparse-retrieval candidates for a query."""
        scores = similarity_scores(self.corpus, query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:k]
        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            row = self.corpus.records[index]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[index]),
                    lesson_id=row["lesson_id"],
                    title=row["title"],
                    module=row["module"],
                    description=row["description"],
                )
            )
        return results


def relevant_in_top_k(
    results: list[SearchResult], expected_topic: str, k: int = 3
) -> bool:
    """Check whether the labeled topic is represented in the first k candidates."""
    expected = expected_topic.lower().strip().replace("_", " ")
    for result in results[:k]:
        text = f"{result.lesson_id} {result.title} {result.module} {result.description}"
        searchable = text.lower().replace("_", " ")
        if expected in searchable:
            return True
    return False
