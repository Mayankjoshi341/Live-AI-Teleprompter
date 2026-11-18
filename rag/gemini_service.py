from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator


class GeminiServiceError(Exception):
    """Raised for Gemini API and setup errors that are safe to display in the UI."""


def _load_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiServiceError(
            "The Google Gen AI SDK is missing or installed in a different Python environment. "
            'Run: python -m pip install --upgrade "google-genai>=3.3.0,<4"'
        ) from exc
    return genai, types


def build_gemini_client(api_key: str) -> Any:
    if not api_key.strip():
        raise GeminiServiceError("Enter your Gemini API key in the sidebar.")
    genai, _ = _load_genai()
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception as exc:
        raise GeminiServiceError(_friendly_api_error(exc, "Could not initialize the Gemini client")) from exc


def _friendly_api_error(exc: Exception, fallback: str) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if any(term in lowered for term in ("api key", "unauthorized", "permission_denied", "forbidden", "401", "403")):
        return "Gemini rejected the API key. Check that it is valid and that the Gemini API is enabled in Google AI Studio."
    if any(term in lowered for term in ("quota", "resource_exhausted", "rate limit", "429")):
        return "Gemini API quota or rate limit reached. Check your Google AI Studio quota and try again."
    if "not found" in lowered or "404" in lowered or "unsupported model" in lowered:
        return f"The requested Gemini model is unavailable. Check the model name and API access. Provider message: {message[:500]}"
    if message:
        return f"{fallback}. Provider message: {message[:700]}"
    return fallback


def _build_answer_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    # Keep retrieval context deliberately small to reduce prompt processing time.
    source_sections = []
    for index, item in enumerate(retrieved_chunks[:3], start=1):
        text = str(item.get("text", ""))[:1800]
        source_sections.append(f"[SOURCE {index}] {item['source']} ({item['locator']})\n{text}")
    context = "\n\n".join(source_sections) if source_sections else (
        "No uploaded document passage matched this question. Answer from reliable general knowledge "
        "instead, and do not invent or cite uploaded sources."
    )

    grounding_instruction = (
        "Use the supplied passages as your primary source. If they do not fully answer the question, "
        "you may add reliable general knowledge, but do not attribute it to the documents. Cite claims "
        "from supplied passages using [1], [2], etc."
        if source_sections else
        "Answer from reliable general knowledge. Do not pretend that an uploaded source supports the answer."
    )

    return f"""You are a clear, accurate speaking assistant. The user will read your answer aloud.

Answer style: ADAPTIVE.
- Start with the direct answer; do not greet or add an introduction.
- For a simple factual question, answer in 2–4 natural sentences.
- For a technical, how/why, comparison, or multi-part question, explain the essential details clearly.
- Use plain fluent English and short speakable sentences.
- Keep the answer focused; avoid filler and repeated conclusions.
- {grounding_instruction}
- Do not mention that you are an AI model.

QUESTION:
{question.strip()}

UPLOADED MATERIAL:
{context}

Write the answer now."""


def generate_answer_stream(
    *,
    client: Any,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str,
) -> Iterator[str]:
    """Yield answer text as Gemini streams it, instead of waiting for full completion."""
    _, types = _load_genai()
    prompt = _build_answer_prompt(question, retrieved_chunks)
    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=360,
                temperature=0.25,
            ),
        )
        emitted = False
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                emitted = True
                yield text
        if not emitted:
            raise GeminiServiceError("Gemini returned an empty answer. Try again or rephrase the question.")
    except GeminiServiceError:
        raise
    except Exception as exc:
        raise GeminiServiceError(_friendly_api_error(exc, "Gemini could not generate an answer")) from exc


def generate_answer(
    *,
    client: Any,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str,
) -> str:
    """Compatibility wrapper that collects the streamed answer into one string."""
    return "".join(
        generate_answer_stream(
            client=client,
            question=question,
            retrieved_chunks=retrieved_chunks,
            model=model,
        )
    ).strip()


def transcribe_audio(
    *,
    client: Any,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    model: str = "gemini-3.5-transcribe",
) -> str:
    """Transcribe recorded-audio fallback using Gemini Files + Interactions APIs."""
    if not audio_bytes:
        raise GeminiServiceError("No microphone recording was received. Please record the question again.")

    normalized_mime = (mime_type or "audio/webm").split(";")[0].strip().lower()
    extension_by_mime = {
        "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm",
        "audio/mp4": ".mp4", "audio/m4a": ".m4a", "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/aac": ".aac",
    }
    suffix = extension_by_mime.get(normalized_mime, ".webm")
    uploaded_file = None
    try:
        with TemporaryDirectory(prefix="live_assistant_audio_") as temporary_dir:
            audio_path = Path(temporary_dir) / f"recording{suffix}"
            audio_path.write_bytes(audio_bytes)
            uploaded_file = client.files.upload(file=str(audio_path))

        interaction = client.interactions.create(
            model=model,
            input=[{
                "type": "audio",
                "uri": uploaded_file.uri,
                "mime_type": getattr(uploaded_file, "mime_type", None) or normalized_mime or "audio/webm",
            }],
        )
        transcript = (getattr(interaction, "output_text", None) or "").strip()
        if not transcript:
            raise GeminiServiceError(
                "Gemini could not hear a clear question. Please record it again, closer to the microphone."
            )
        return transcript
    except GeminiServiceError:
        raise
    except Exception as exc:
        raise GeminiServiceError(_friendly_api_error(exc, "Audio transcription failed")) from exc
    finally:
        uploaded_name = getattr(uploaded_file, "name", None) if uploaded_file is not None else None
        if uploaded_name:
            try:
                client.files.delete(name=uploaded_name)
            except Exception:
                pass
