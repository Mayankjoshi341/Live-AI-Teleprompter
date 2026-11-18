from rag.retriever import chunk_segments, retrieve_relevant_chunks, split_text


def test_split_text_returns_content_for_short_text():
    text = "Python is a programming language. It is dynamically typed."
    assert split_text(text) == [text]


def test_split_text_creates_overlapping_chunks():
    text = " ".join([f"Sentence number {index}." for index in range(180)])
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert len(chunks) > 1
    assert all(chunks)
    assert chunks[0][-60:] in chunks[1]


def test_chunk_segments_retains_source_and_locator():
    segments = [{"text": "Python is interpreted.", "locator": "page 4"}]
    chunks = chunk_segments(segments, source_name="notes.pdf")
    assert chunks[0]["source"] == "notes.pdf"
    assert chunks[0]["locator"] == "page 4"
    assert chunks[0]["text"] == "Python is interpreted."


def test_retrieval_finds_relevant_chunk():
    chunks = [
        {"source": "python.pdf", "locator": "page 1", "text": "Python is dynamically typed. Variables do not need explicit type declarations."},
        {"source": "django.pdf", "locator": "page 2", "text": "Django is a high-level Python web framework."},
        {"source": "sql.pdf", "locator": "page 3", "text": "SQL is used to query relational databases."},
    ]
    results = retrieve_relevant_chunks(question="Why is Python dynamically typed?", chunks=chunks)
    assert results
    assert results[0]["source"] == "python.pdf"


def test_retrieval_returns_empty_when_no_terms_match():
    chunks = [{"source": "sql.pdf", "locator": "page 1", "text": "Relational databases use tables and SQL."}]
    assert retrieve_relevant_chunks(question="Explain quantum entanglement", chunks=chunks) == []
