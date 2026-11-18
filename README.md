# Live Answer Assistant — Final Prototype

A mobile-friendly Streamlit assistant with two interchangeable ways to ask a question:

- **Live answer:** Start listening → live English transcript → Stop → automatic answer.
- **Manual question:** type a question → Generate answer.

Both modes use the same local document retrieval and Gemini answer service.

## Final features

- Live microphone capture with explicit browser permission request.
- Interim English transcription while speaking where Web Speech API is supported.
- MediaRecorder backup and Gemini transcription for browsers without live speech recognition.
- Automatic answer generation after Stop in Live mode.
- Manual question mode, available from the two mode-switch buttons at the top.
- Fast default model: `gemini-3.5-flash-lite`; optional balanced model: `gemini-3.8-flash`.
- Streamed Gemini output: the first text appears as soon as the model emits it rather than waiting for the complete response.
- Adaptive spoken-English answer style with concise defaults.
- Local BM25-style search over uploaded PDF, DOCX, TXT and Markdown documents.
- General-knowledge fallback when no document passage matches.
- Mobile-first dark teleprompter with a landscape-friendly layout.
- Sidebar control for auto-scroll and speed (10–100 pixels per second).
- Pause/resume, return-to-top and copy-answer controls.
- Source passages shown when uploaded material was retrieved.

## Important latency note

Streaming reduces perceived wait time because text is displayed as it arrives. It cannot guarantee a complete answer within one second: actual time depends on speech finalization, provider/network latency, quota/load, and answer length. The app records first-text and full-answer timing after a successful generation so performance can be measured on the actual deployment.

For the quickest path, use the browser's live transcript. If the browser has to use the recorded-audio Gemini transcription fallback, the app must first transcribe audio and then generate an answer, which adds another network request.

## Requirements

- Python 3.11 or newer (Python 3.12 recommended)
- Gemini API key: https://aistudio.google.com/app/apikey
- Current Chrome or Edge recommended for live browser speech recognition
- HTTPS for microphone access when opened on a phone

## Run on Windows

Open PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `setup_and_run_windows.bat`.

## Use it

1. Open the sidebar and paste your Gemini API key.
2. Keep **Fastest · Gemini 3.5 Flash-Lite** selected for lower-latency answers. Choose Balanced only when you prefer the larger model.
3. Upload PDF, DOCX, TXT or Markdown files and click **Add selected files**. You can also paste reference notes.
4. Choose **Live answer** or **Manual question** using the two buttons at the top.
5. In Live answer mode, tap **Start listening**, speak the question, and tap **Stop listening**. The app automatically generates the answer.
6. In Manual question mode, type the question and press **Generate answer**.
7. Read the answer in the teleprompter. Adjust scroll speed and auto-scroll from the sidebar. Pause/resume and return to the top from the teleprompter itself.
8. Use **Copy answer** to copy the answer. Expand **Sources used** to inspect matched passages.

## Phone microphone notes

- Open the app in the browser itself, not in an in-app preview.
- Phone access requires an HTTPS URL. A plain LAN address such as `http://192.168.1.5:8501` is not a secure context for microphone capture.
- On Android Chrome, tap the site icon → Site settings → Microphone → Allow, then reload.
- On iPhone Safari, check the page website settings and iOS Settings → Privacy & Security → Microphone.
- Web Speech API availability varies by browser and OS. Some browsers use a remote speech recognition service. When live recognition is unavailable, the recording fallback is transcribed after Stop and is therefore slower.

## Retrieval notes

- Document extraction and retrieval run locally.
- Retrieval uses a lightweight BM25-style keyword ranker, not vector embeddings. It is fast and simple, but may miss semantically related passages when vocabulary differs.
- Scanned/image-only PDFs need OCR, which is not included. Legacy `.doc` files are not supported; save them as `.docx`.
- Knowledge base content is stored in Streamlit session memory. Restarting the app or losing the session means re-uploading files.
- Uploaded text passages and captured questions are sent to Gemini for answer generation. Use only material you are comfortable sending to the provider.

## Troubleshooting

### Gemini SDK import error

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "google-genai>=3.3.0,<4"
.\.venv\Scripts\python.exe -c "from google import genai; print('Gemini SDK import OK')"
```

Do not install `google` using `pip install google`; the required distribution is `google-genai`.

### Microphone prompt doesn't appear on phone

Open the HTTPS deployment in Chrome or Safari. Check the site's microphone permission and reload. Plain HTTP LAN URLs and in-app browsers may block microphone access.

### Transcript appears but answer doesn't

Check that a Gemini API key is entered and that the selected model is available to that key. The application displays provider errors below the capture area. Try the Fastest model option first.

### Answer is slow

Use the Fastest model option. Also check whether the app is using the live browser transcript or the recorded-audio transcription fallback. Streaming only improves time-to-first-text; it cannot guarantee a full response within one second.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Project layout

```text
live_answer_assistant_fixed/
├── app.py
├── components/live_speech/index.html
├── rag/
│   ├── component_protocol.py
│   ├── document_loader.py
│   ├── gemini_service.py
│   └── retriever.py
├── tests/
├── sample_data/
├── requirements.txt
└── setup_and_run_windows.bat
```
