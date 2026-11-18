import json

from rag.component_protocol import normalize_component_payload


def test_native_json_object_is_preserved():
    payload = {"event_id": "evt-1", "status": "stopped", "transcript": "What is Python?"}
    assert normalize_component_payload(payload) == payload


def test_legacy_json_string_is_decoded():
    payload = {"event_id": "evt-2", "status": "stopped", "transcript": "Explain Django."}
    assert normalize_component_payload(json.dumps(payload)) == payload


def test_wrapped_legacy_json_string_is_decoded():
    payload = {"event_id": "evt-3", "status": "stopped", "transcript": "What is SQL?"}
    wrapped = {"value": json.dumps(payload), "dataType": "json"}
    assert normalize_component_payload(wrapped) == payload


def test_invalid_component_payload_is_ignored():
    assert normalize_component_payload("not-json") is None
    assert normalize_component_payload(17) is None


def test_live_component_uses_native_json_value_and_explicit_mic_prompt():
    from pathlib import Path

    component_path = Path(__file__).parents[1] / "components" / "live_speech" / "index.html"
    component_html = component_path.read_text(encoding="utf-8")
    assert "value: payload" in component_html
    assert "value: JSON.stringify(payload)" not in component_html
    assert "navigator.mediaDevices.getUserMedia" in component_html
    assert "color:#f7f9fc" in component_html
