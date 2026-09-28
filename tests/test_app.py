"""Smoke test: the Streamlit script runs to completion without raising (needs streamlit + TensorFlow)."""

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_starts_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    # Either the model exists (tabs are shown) or a clear error message is shown - never a crash.
    assert len(at.tabs) == 3 or len(at.error) >= 1
