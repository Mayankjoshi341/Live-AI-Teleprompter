from types import SimpleNamespace

from rag.gemini_service import generate_answer, generate_answer_stream, transcribe_audio


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePart:
    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str):
        return {"data": data, "mime_type": mime_type}


class FakeTypes:
    GenerateContentConfig = FakeGenerateContentConfig
    Part = FakePart


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.stream_calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        words = self.text.split(" ")
        for index, word in enumerate(words):
            yield SimpleNamespace(text=word + (" " if index < len(words) - 1 else ""))


class FakeUploadedFile:
    uri = "files/fake-audio"
    mime_type = "audio/wav"
    name = "files/fake-audio"


class FakeFiles:
    def __init__(self):
        self.upload_calls = []
        self.delete_calls = []

    def upload(self, **kwargs):
        self.upload_calls.append(kwargs)
        return FakeUploadedFile()

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


class FakeInteractions:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)
        self.files = FakeFiles()
        self.interactions = FakeInteractions(text)


def test_generate_answer_returns_streamed_text(monkeypatch):
    monkeypatch.setattr("rag.gemini_service._load_genai", lambda: (None, FakeTypes))
    client = FakeClient("Python is dynamically typed. [1]")
    answer = generate_answer(
        client=client,
        question="Why is Python dynamically typed?",
        retrieved_chunks=[{"source": "python.txt", "locator": "text file", "text": "Python is dynamically typed."}],
        model="gemini-3.5-flash-lite",
    )
    assert answer.startswith("Python is dynamically typed")
    assert client.models.stream_calls[0]["model"] == "gemini-3.5-flash-lite"
    assert client.models.stream_calls[0]["config"].kwargs["max_output_tokens"] == 360


def test_generate_answer_stream_yields_incremental_text(monkeypatch):
    monkeypatch.setattr("rag.gemini_service._load_genai", lambda: (None, FakeTypes))
    client = FakeClient("First part. Second part.")
    chunks = list(generate_answer_stream(
        client=client,
        question="Explain something.",
        retrieved_chunks=[],
        model="gemini-3.5-flash-lite",
    ))
    assert chunks == ["First ", "part. ", "Second ", "part."]


def test_transcribe_audio_returns_transcript(monkeypatch):
    monkeypatch.setattr("rag.gemini_service._load_genai", lambda: (None, FakeTypes))
    client = FakeClient("What is Python?")
    transcript = transcribe_audio(client=client, audio_bytes=b"fake wav data", mime_type="audio/wav")
    assert transcript == "What is Python?"
    assert client.interactions.calls[0]["model"] == "gemini-3.5-transcribe"
    assert client.interactions.calls[0]["input"][0]["uri"] == "files/fake-audio"
    assert client.files.delete_calls == [{"name": "files/fake-audio"}]


def test_transcribe_audio_keeps_webm_extension(monkeypatch):
    monkeypatch.setattr("rag.gemini_service._load_genai", lambda: (None, FakeTypes))
    client = FakeClient("What is Django middleware?")
    transcript = transcribe_audio(client=client, audio_bytes=b"fake webm data", mime_type="audio/webm;codecs=opus")
    assert transcript == "What is Django middleware?"
    assert client.files.upload_calls[0]["file"].endswith(".webm")
