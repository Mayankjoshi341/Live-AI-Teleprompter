from pathlib import Path


def test_app_has_live_and_manual_modes_and_streaming_default():
    app = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert '"Live answer"' in app
    assert '"Manual question"' in app
    assert "generate_answer_stream" in app
    assert '"gemini-3.5-flash-lite"' in app


def test_app_exposes_teleprompter_speed_and_autoscroll_settings():
    app = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert '"Scrolling speed"' in app
    assert '"Automatic scrolling"' in app
    assert "render_teleprompter" in app


def test_teleprompter_supports_landscape_mobile_layout():
    app = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "orientation:landscape" in app
    assert "height: min(54vh, 370px)" in app
