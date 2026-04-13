"""Tests for reelcribe.llm module."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelcribe.llm import generate_title, save_title


class TestGenerateTitle:
    def _mock_response(self, body: dict):
        """Build a mock urllib response."""
        raw = json.dumps(body).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_title_from_api(self):
        mock_resp = self._mock_response({"response": "  My Great Reel  "})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            title = generate_title("Some transcript text")
        assert title == "My Great Reel"

    def test_raises_connection_error_on_url_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(ConnectionError, match="Could not reach Ollama"):
                generate_title("transcript")

    def test_raises_runtime_error_on_bad_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Unexpected response"):
                generate_title("transcript")

    def test_raises_runtime_error_on_missing_key(self):
        mock_resp = self._mock_response({"something": "else"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Unexpected response"):
                generate_title("transcript")


class TestSaveTitle:
    def test_writes_file(self, tmp_path):
        out = tmp_path / "title.txt"
        result = save_title("My Title", out)
        assert result == out
        assert out.read_text(encoding="utf-8") == "My Title"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "title.txt"
        save_title("Hello", out)
        assert out.exists()
