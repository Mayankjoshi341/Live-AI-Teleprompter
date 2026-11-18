from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from rag.component_protocol import normalize_component_payload
from rag.document_loader import DocumentLoadError, extract_document
from rag.gemini_service import (
    GeminiServiceError,
    build_gemini_client,
    generate_answer_stream,
    transcribe_audio,
)
from rag.retriever import chunk_segments, retrieve_relevant_chunks

APP_TITLE = "Live Answer Assistant"
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"
MODEL_OPTIONS = {
    "Fastest · Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Balanced · Gemini 3.8 Flash": "gemini-3.8-flash",
}
MAX_FILE_SIZE_MB = 20

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
live_speech_component = components.declare_component(
    "live_speech_recognition",
    path=str(BASE_DIR / "components" / "live_speech"),
)

st.markdown(
    """
    <style>
      .block-container { max-width: 960px; padding-top: 1rem; padding-bottom: 2rem; }
      .app-kicker { color: #7290a8; font-size: .70rem; text-transform: uppercase;
                    letter-spacing: .14em; font-weight: 800; margin-bottom: .25rem; }
      .mode-note { color: var(--text-color); opacity: .72; font-size: .94rem; }
      .answer-stream { border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
                       padding: 17px 19px; background: var(--secondary-background-color);
                       color: var(--text-color); font-size: 1.14rem; line-height: 1.6;
                       white-space: pre-wrap; overflow-wrap: anywhere; }
      .question-panel { border: 1px solid rgba(128,128,128,.24); border-radius: 13px;
                        padding: 13px 15px; background: rgba(128,128,128,.055);
                        line-height: 1.55; overflow-wrap: anywhere; }
      div[data-testid="stButton"] button { min-height: 48px; border-radius: 12px; font-weight: 700; }
      @media (max-width: 700px) {
        .block-container { padding-left: .75rem; padding-right: .75rem; padding-top: .7rem; }
        h1 { font-size: 1.85rem !important; line-height: 1.15 !important; }
        h2, h3 { line-height: 1.2 !important; }
        div[data-testid="stButton"] button { min-height: 52px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "knowledge_chunks": [],
        "processed_files": {},
        "last_question": "",
        "last_answer": "",
        "last_sources": [],
        "last_answer_mode": "",
        "last_speech_event_id": "",
        "capture_error": "",
        "capture_warning": "",
        "generation_error": "",
        "generation_notice": "",
        "input_mode": "Live answer",
        "last_generation_seconds": None,
        "last_first_text_seconds": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def get_saved_api_key() -> str:
    try:
        secret_value = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_value = ""
    return str(secret_value or os.getenv("GEMINI_API_KEY", "")).strip()


def fingerprint(name: str, data: bytes) -> str:
    return f"{name}:{hashlib.sha256(data).hexdigest()}"


def add_to_knowledge_base(source_name: str, segments: list[dict[str, str]]) -> int:
    chunks = chunk_segments(segments, source_name=source_name)
    if not chunks:
        raise DocumentLoadError(f"No readable text was extracted from {source_name}.")
    st.session_state.knowledge_chunks.extend(chunks)
    return len(chunks)


def clear_last_answer() -> None:
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.last_sources = []
    st.session_state.last_answer_mode = ""
    st.session_state.generation_error = ""
    st.session_state.generation_notice = ""
    st.session_state.last_generation_seconds = None
    st.session_state.last_first_text_seconds = None


def render_stream_preview(target: Any, text: str) -> None:
    safe_text = html.escape(text)
    target.markdown(
        '<div class="answer-stream" aria-live="polite">' + safe_text + "</div>",
        unsafe_allow_html=True,
    )


def generate_and_store_answer(
    question_text: str,
    api_key_value: str,
    model_name: str,
    *,
    audio_bytes: bytes | None = None,
    audio_mime_type: str = "audio/webm",
    transcript_is_final: bool = True,
    stream_target: Any = None,
) -> tuple[bool, str]:
    """Transcribe fallback audio if needed, retrieve context, and stream Gemini's answer."""
    question_text = question_text.strip()
    if not question_text and not audio_bytes:
        return False, "No speech was captured. Check microphone access and try again."

    # Clear stale output before validating the key so an old answer never looks like
    # the answer to a new question whose generation could not start.
    clear_last_answer()
    if question_text:
        st.session_state.last_question = question_text
    if not api_key_value:
        return False, "Add your Gemini API key in the sidebar before generating an answer."

    overall_started = time.perf_counter()
    first_text_seconds: float | None = None
    answer_parts: list[str] = []
    try:
        client = build_gemini_client(api_key_value)
        with st.status("Preparing your answer…", expanded=False) as status:
            if audio_bytes and not transcript_is_final:
                status.update(label="Transcribing the recording…", state="running", expanded=False)
                question_text = transcribe_audio(
                    client=client,
                    audio_bytes=audio_bytes,
                    mime_type=audio_mime_type,
                ).strip()

            question_text = question_text.strip()
            if not question_text:
                status.update(label="No question detected", state="error", expanded=True)
                return False, "I couldn't detect a clear question. Please speak a little closer to the microphone and try again."

            st.session_state.last_question = question_text
            retrieved = retrieve_relevant_chunks(
                question=question_text,
                chunks=st.session_state.knowledge_chunks,
                top_k=3,
            )
            answer_mode = (
                "Based on your uploaded material" if retrieved
                else "General knowledge · no matching uploaded passage"
            )
            status.update(label="Generating answer…", state="running", expanded=False)
            generation_started = time.perf_counter()

            if stream_target is not None:
                stream_target.markdown("_Generating answer…_", unsafe_allow_html=False)

            last_preview_at = 0
            for piece in generate_answer_stream(
                client=client,
                question=question_text,
                retrieved_chunks=retrieved,
                model=model_name,
            ):
                if first_text_seconds is None:
                    first_text_seconds = time.perf_counter() - generation_started
                answer_parts.append(piece)
                current_answer = "".join(answer_parts)
                st.session_state.last_answer = current_answer
                # Batch small token fragments for fewer UI rerenders while keeping the answer live.
                if stream_target is not None and (
                    last_preview_at == 0 or len(current_answer) - last_preview_at >= 24 or "\n" in piece
                ):
                    render_stream_preview(stream_target, current_answer)
                    last_preview_at = len(current_answer)

            answer = "".join(answer_parts).strip()
            if not answer:
                raise GeminiServiceError("Gemini returned an empty answer. Try asking the question again.")
            st.session_state.last_answer = answer
            st.session_state.last_sources = retrieved
            st.session_state.last_answer_mode = answer_mode
            st.session_state.last_first_text_seconds = first_text_seconds
            st.session_state.last_generation_seconds = time.perf_counter() - overall_started
            if stream_target is not None:
                # Show the final polished teleprompter below after the streamed preview completes.
                stream_target.empty()
            status.update(label="Answer ready", state="complete", expanded=False)
        return True, "Answer generated."
    except GeminiServiceError as exc:
        if answer_parts:
            st.session_state.last_answer = "".join(answer_parts).strip()
        elif stream_target is not None:
            stream_target.empty()
        return False, str(exc)
    except Exception as exc:
        if answer_parts:
            st.session_state.last_answer = "".join(answer_parts).strip()
        elif stream_target is not None:
            stream_target.empty()
        return False, f"Unexpected answer-generation error: {type(exc).__name__}: {exc}"


def render_copy_button(text: str) -> None:
    payload = json.dumps(text, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    components.html(
        f"""
        <div style="font-family:Inter,system-ui,sans-serif;">
          <button id="copyAnswer" style="border:1px solid rgba(128,128,128,.4);border-radius:10px;
            padding:10px 15px;background:transparent;color:inherit;font-weight:700;cursor:pointer;min-height:42px;">
            Copy answer
          </button>
          <span id="copyStatus" style="font-size:12px;margin-left:8px;opacity:.75;"></span>
        </div>
        <script>
          const answerText = {payload};
          const button = document.getElementById('copyAnswer');
          const status = document.getElementById('copyStatus');
          button.addEventListener('click', async () => {{
            try {{
              if (navigator.clipboard && window.isSecureContext) {{
                await navigator.clipboard.writeText(answerText);
              }} else {{
                const area = document.createElement('textarea');
                area.value = answerText;
                area.style.position = 'fixed'; area.style.opacity = '0';
                document.body.appendChild(area); area.select();
                const copied = document.execCommand('copy'); area.remove();
                if (!copied) throw new Error('Clipboard unavailable');
              }}
              status.textContent = 'Copied';
            }} catch (error) {{ status.textContent = 'Copy was blocked by the browser'; }}
          }});
        </script>
        """,
        height=52,
        scrolling=False,
    )


def render_teleprompter(text: str, *, speed: int, auto_scroll: bool) -> None:
    answer_json = (json.dumps(text, ensure_ascii=False)
                   .replace("<", "\\u003c")
                   .replace("\u2028", "\\u2028")
                   .replace("\u2029", "\\u2029"))
    auto_start = "true" if auto_scroll else "false"
    components.html(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
          <style>
            * {{ box-sizing: border-box; }}
            html, body {{ margin:0; padding:0; background:transparent; font-family:Inter,system-ui,-apple-system,sans-serif; }}
            .tele {{ border:1px solid rgba(130,145,165,.32); border-radius:18px; overflow:hidden;
                    background:#111821; color:#f7f9fc; }}
            .bar {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:11px 14px;
                    background:#1b2633; color:#d8e3ef; font-size:12px; font-weight:750; letter-spacing:.04em; }}
            .actions {{ display:flex; gap:7px; }}
            .actions button {{ border:1px solid #536477; background:#273646; color:#fff; border-radius:9px;
                              padding:7px 10px; min-height:34px; font-weight:700; cursor:pointer; }}
            #reader {{ height:340px; overflow-y:auto; overscroll-behavior:contain; padding:24px clamp(18px,4vw,34px);
                       font-size:clamp(23px,3.6vw,32px); line-height:1.58; font-weight:580; white-space:pre-wrap;
                       overflow-wrap:anywhere; scrollbar-width:thin; scroll-behavior:auto; }}
            .foot {{ display:flex; justify-content:space-between; padding:9px 14px; color:#aebccc; font-size:11px;
                     border-top:1px solid rgba(160,175,190,.18); }}
            @media(max-width:600px) {{
              .tele {{ border-radius:13px; }}
              .bar {{ padding:10px 11px; }}
              #reader {{ height: min(54vh, 370px); min-height:230px; padding:19px 17px; font-size:25px; line-height:1.62; }}
            }}
            @media(orientation:landscape) and (max-height:600px) {{
              #reader {{ height:58vh; min-height:180px; font-size:23px; padding:16px 22px; }}
            }}
          </style>
        </head>
        <body>
          <div class="tele">
            <div class="bar">
              <span>TELEPROMPTER · {speed} PX/SEC</span>
              <div class="actions">
                <button id="pauseBtn" type="button">Pause</button>
                <button id="topBtn" type="button">Top</button>
              </div>
            </div>
            <div id="reader" tabindex="0" aria-label="Answer teleprompter"></div>
            <div class="foot"><span id="scrollStatus">{('Auto-scroll on' if auto_scroll else 'Manual scroll')}</span><span>Landscape mode supported</span></div>
          </div>
          <script>
            const answerText = {answer_json};
            const reader = document.getElementById('reader');
            const pauseBtn = document.getElementById('pauseBtn');
            const topBtn = document.getElementById('topBtn');
            const status = document.getElementById('scrollStatus');
            const speed = {int(speed)};
            reader.textContent = answerText;
            let running = {auto_start};
            let lastTime = 0;
            function updateButton() {{
              pauseBtn.textContent = running ? 'Pause' : 'Resume';
              status.textContent = running ? 'Auto-scroll on' : 'Scrolling paused';
            }}
            function animate(timestamp) {{
              if (running) {{
                if (lastTime) reader.scrollTop += (timestamp - lastTime) * speed / 1000;
                if (reader.scrollTop + reader.clientHeight >= reader.scrollHeight - 2) running = false;
                updateButton();
              }}
              lastTime = timestamp;
              requestAnimationFrame(animate);
            }}
            pauseBtn.addEventListener('click', () => {{ running = !running; lastTime = 0; updateButton(); }});
            topBtn.addEventListener('click', () => {{ reader.scrollTop = 0; if ({auto_start}) running = true; lastTime = 0; updateButton(); }});
            reader.addEventListener('touchstart', () => {{ if (running) {{ running = false; updateButton(); }} }}, {{passive:true}});
            updateButton();
            requestAnimationFrame(animate);
          </script>
        </body>
        </html>
        """,
        height=440,
        scrolling=False,
    )


# ----------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.title("Assistant settings")
    st.caption("Manage your knowledge base and reading experience.")

    api_key = st.text_input(
        "Gemini API key",
        value=get_saved_api_key(),
        type="password",
        placeholder="Paste your Gemini API key",
        help="You can also set GEMINI_API_KEY in .streamlit/secrets.toml or your environment.",
        key="gemini_api_key_input",
    ).strip()

    model_label = st.selectbox(
        "Answer speed / model",
        options=list(MODEL_OPTIONS),
        index=0,
        help="Flash-Lite is the fast default. Use Flash for a more capable but potentially slower answer.",
    )
    generation_model = MODEL_OPTIONS[model_label]

    st.divider()
    st.subheader("Teleprompter")
    auto_scroll = st.toggle("Automatic scrolling", value=True, key="teleprompter_auto_scroll")
    scroll_speed = st.slider(
        "Scrolling speed",
        min_value=10,
        max_value=100,
        value=32,
        step=2,
        help="Pixels per second. Increase this if you read quickly; reduce it for slower reading.",
        key="teleprompter_speed",
    )
    st.caption(f"Current speed: {scroll_speed} px/sec")

    st.divider()
    st.subheader("Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILE_SIZE_MB} MB per file. Scanned PDFs and legacy .doc files are not supported.",
        key="document_upload",
    )
    add_files_clicked = st.button(
        "Add selected files",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files),
    )

    st.markdown("**Or paste reference notes**")
    pasted_title = st.text_input("Notes title", value="Pasted notes")
    pasted_text = st.text_area(
        "Reference text",
        height=115,
        placeholder="Paste definitions, documentation or interview notes…",
    )
    add_text_clicked = st.button(
        "Add pasted notes",
        use_container_width=True,
        disabled=not bool(pasted_text.strip()),
    )

    st.divider()
    source_names = sorted({item["source"] for item in st.session_state.knowledge_chunks})
    st.metric("Indexed chunks", len(st.session_state.knowledge_chunks))
    st.caption(f"{len(source_names)} source(s) added")
    if source_names:
        with st.expander("View indexed sources"):
            for source_name in source_names:
                st.write(f"• {source_name}")

    if st.button("Clear knowledge base", use_container_width=True):
        st.session_state.knowledge_chunks = []
        st.session_state.processed_files = {}
        clear_last_answer()
        st.session_state.capture_error = ""
        st.rerun()

# ----------------------------- Header --------------------------------------
st.markdown('<div class="app-kicker">Fast answers · live capture · manual mode</div>', unsafe_allow_html=True)
st.title("Live Answer Assistant")
st.write("Ask by microphone or type a question. Answers are streamed as soon as Gemini returns text.")

mode_left, mode_right = st.columns(2, gap="small")
with mode_left:
    if st.button(
        "🎙 Live answer",
        type="primary" if st.session_state.input_mode == "Live answer" else "secondary",
        use_container_width=True,
        key="switch_live_mode",
    ):
        st.session_state.input_mode = "Live answer"
        st.rerun()
with mode_right:
    if st.button(
        "⌨️ Manual question",
        type="primary" if st.session_state.input_mode == "Manual question" else "secondary",
        use_container_width=True,
        key="switch_manual_mode",
    ):
        st.session_state.input_mode = "Manual question"
        st.rerun()

# ----------------------------- Document processing ------------------------
if add_files_clicked:
    added_files = 0
    added_chunks = 0
    errors: list[str] = []
    for uploaded in uploaded_files or []:
        content = uploaded.getvalue()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{uploaded.name}: exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            continue
        file_id = fingerprint(uploaded.name, content)
        if file_id in st.session_state.processed_files:
            continue
        try:
            segments = extract_document(uploaded.name, content)
            chunk_count = add_to_knowledge_base(uploaded.name, segments)
            st.session_state.processed_files[file_id] = uploaded.name
            added_files += 1
            added_chunks += chunk_count
        except DocumentLoadError as exc:
            errors.append(f"{uploaded.name}: {exc}")
        except Exception as exc:
            errors.append(f"{uploaded.name}: {type(exc).__name__}: {exc}")
    if added_files:
        st.success(f"Added {added_files} file(s) and indexed {added_chunks} text chunk(s).")
    for error in errors:
        st.error(error)

if add_text_clicked:
    title = pasted_title.strip() or "Pasted notes"
    clean_text = pasted_text.strip()
    text_id = "pasted:" + hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    if text_id in st.session_state.processed_files:
        st.info("These exact notes have already been added.")
    else:
        try:
            count = add_to_knowledge_base(title, [{"text": clean_text, "locator": "pasted notes"}])
            st.session_state.processed_files[text_id] = title
            st.success(f"Added {count} chunk(s) from '{title}'.")
        except DocumentLoadError as exc:
            st.error(str(exc))

# ----------------------------- Input modes ---------------------------------
if st.session_state.input_mode == "Live answer":
    st.subheader("Live question capture")
    st.caption("Tap Start when the question begins. Stop when the speaker finishes; the answer starts automatically.")
    if not st.session_state.knowledge_chunks:
        st.info("No reference files are indexed yet. General-knowledge answers still work.")
    if not api_key:
        st.warning("Add your Gemini API key in the sidebar. Transcription can run without it, but answer generation needs it.")

    speech_result_raw = live_speech_component(key="live_speech_recognition", default=None)
    st.caption("Phone access requires HTTPS. If microphone access was previously blocked, allow it in your browser's site settings and reload.")
    speech_result = normalize_component_payload(speech_result_raw)
    stream_placeholder = st.empty()
    if speech_result:
        event_id = str(speech_result.get("event_id", "")).strip()
        if event_id and event_id != st.session_state.last_speech_event_id:
            st.session_state.last_speech_event_id = event_id
            transcript = str(speech_result.get("transcript", "")).strip()
            speech_error = str(speech_result.get("error", "")).strip()
            status = str(speech_result.get("status", ""))
            transcript_is_final = bool(speech_result.get("is_final_transcript", bool(transcript)))
            audio_base64 = str(speech_result.get("audio_base64", "") or "").strip()
            audio_mime_type = str(speech_result.get("audio_mime_type", "audio/webm") or "audio/webm")

            st.session_state.capture_error = ""
            st.session_state.capture_warning = ""
            st.session_state.generation_error = ""

            audio_bytes = None
            if audio_base64:
                try:
                    audio_bytes = base64.b64decode(audio_base64, validate=True)
                except (binascii.Error, ValueError):
                    st.session_state.generation_error = "The microphone audio backup was malformed. Please record the question again."

            if status == "stopped" and (transcript or audio_bytes):
                if speech_error:
                    st.session_state.capture_warning = speech_error
                success, message = generate_and_store_answer(
                    transcript,
                    api_key,
                    generation_model,
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                    transcript_is_final=transcript_is_final,
                    stream_target=stream_placeholder,
                )
                if success:
                    st.session_state.generation_notice = "Your answer is ready."
                else:
                    st.session_state.generation_error = message
            elif status == "permission_error" or speech_error:
                st.session_state.capture_error = speech_error or "Microphone capture failed. Check browser permissions."
            else:
                st.session_state.capture_error = "No clear speech was captured. Please try again."

else:
    st.subheader("Manual question")
    st.caption("Type a question and submit it. The same fast retrieval and streaming answer engine is used.")
    with st.form("manual_question_form", clear_on_submit=False):
        manual_question = st.text_area(
            "Your question",
            placeholder="For example: What is Django middleware and how does it work?",
            height=110,
            key="manual_question_text",
        )
        manual_submit = st.form_submit_button("Generate answer", type="primary", use_container_width=True)
    stream_placeholder = st.empty()
    if manual_submit:
        if not manual_question.strip():
            st.session_state.generation_error = "Enter a question first."
        else:
            st.session_state.generation_error = ""
            success, message = generate_and_store_answer(
                manual_question,
                api_key,
                generation_model,
                stream_target=stream_placeholder,
            )
            if success:
                st.session_state.generation_notice = "Your answer is ready."
            else:
                st.session_state.generation_error = message

# ----------------------------- Results -------------------------------------
if st.session_state.capture_error:
    st.error(st.session_state.capture_error)
if st.session_state.capture_warning:
    st.warning(st.session_state.capture_warning)
if st.session_state.generation_error:
    st.error(st.session_state.generation_error)
if st.session_state.generation_notice:
    st.success(st.session_state.generation_notice)

if st.session_state.last_question:
    st.markdown("**Question captured**")
    st.markdown(
        '<div class="question-panel">' + html.escape(st.session_state.last_question) + "</div>",
        unsafe_allow_html=True,
    )

if st.session_state.last_answer:
    st.divider()
    st.markdown("### Your answer")
    if st.session_state.last_answer_mode:
        st.caption(st.session_state.last_answer_mode)
    if st.session_state.last_first_text_seconds is not None and st.session_state.last_generation_seconds is not None:
        st.caption(
            f"First text: {st.session_state.last_first_text_seconds:.2f}s · "
            f"Complete answer: {st.session_state.last_generation_seconds:.2f}s"
        )
    render_teleprompter(
        st.session_state.last_answer,
        speed=int(scroll_speed),
        auto_scroll=bool(auto_scroll),
    )
    render_copy_button(st.session_state.last_answer)
    if st.session_state.last_sources:
        with st.expander("Sources used", expanded=False):
            for index, item in enumerate(st.session_state.last_sources, start=1):
                st.markdown(f"**[{index}] {item['source']} — {item['locator']}**")
                st.caption(item["text"])

st.divider()
st.caption(
    "Live mode: Start listening → live English transcript → Stop → automatic answer. "
    "Manual mode: type a question → Generate answer. Only explicit microphone captures are processed."
)
