from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]*|\d+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "in", "into", "is", "it", "its", "me",
    "of", "on", "or", "our", "please", "should", "so", "that", "the",
    "their", "them", "then", "there", "these", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "explain", "describe", "tell", "about",
}


def tokenize(text: str) -> list[str]:
    return [token.lower().strip("._-+") for token in TOKEN_PATTERN.findall(text.lower()) if len(token.strip("._-+")) > 1]


def split_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()
    if not cleaned:
        return []
    if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be >= 200 and overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end
        if hard_end < text_length:
            lower_bound = start + int(chunk_size * 0.65)
            candidates = [
                cleaned.rfind("\n", lower_bound, hard_end),
                cleaned.rfind(". ", lower_bound, hard_end),
                cleaned.rfind("? ", lower_bound, hard_end),
                cleaned.rfind("! ", lower_bound, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if cleaned[boundary] == "\n" else 2)

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_segments(
    segments: list[dict[str, str]],
    *,
    source_name: str,
    chunk_size: int = 1100,
    overlap: int = 180,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for segment in segments:
        locator = segment.get("locator", "document")
        for text in split_text(segment.get("text", ""), chunk_size, overlap):
            results.append({"source": source_name, "locator": locator, "text": text})
    return results


def retrieve_relevant_chunks(
    *,
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank local document chunks using a small BM25-style lexical retriever.

    This intentionally needs no embedding API, network request, or vector database.
    It keeps document upload reliable even when Gemini is unavailable or rate-limited.
    """
    if not chunks:
        return []

    query_tokens = [token for token in tokenize(question) if token not in STOP_WORDS]
    if not query_tokens:
        query_tokens = tokenize(question)
    if not query_tokens:
        return []

    tokenized_docs = [tokenize(str(item.get("text", ""))) for item in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    query_counts = Counter(query_tokens)
    n_docs = len(chunks)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int]] = []

    for index, (item, tokens, doc_length) in enumerate(zip(chunks, tokenized_docs, doc_lengths)):
        frequencies = Counter(tokens)
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1))
            contribution = inverse_document_frequency * (frequency * (k1 + 1) / denominator)
            score += contribution * (1.0 + min(query_frequency - 1, 2) * 0.1)

        normalized_text = str(item.get("text", "")).lower()
        normalized_question = question.strip().lower()
        if len(normalized_question) >= 5 and normalized_question in normalized_text:
            score += 2.0

        if score > 0:
            scored.append((score, index))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results: list[dict[str, Any]] = []
    for score, index in scored[: max(1, top_k)]:
        result = dict(chunks[index])
        result["score"] = float(score)
        results.append(result)
    return results
