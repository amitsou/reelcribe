"""Tests for reelcribe.llm module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelcribe.llm import (
    append_title_line,
    filenames_in_titles_file,
    generate_title,
    save_title,
    verify_ollama_reachable,
)


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

    def test_language_appears_in_request_payload(self):
        mock_resp = self._mock_response({"response": "Titre"})
        captured: dict[str, str] = {}

        def capture_urlopen(req, timeout=None):
            captured["body"] = req.data.decode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            generate_title("hello", language="French")
        payload = json.loads(captured["body"])
        assert "French" in payload["prompt"]


class TestVerifyOllamaReachable:
    def test_ok_when_tags_endpoint_responds(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models":[]}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            base = verify_ollama_reachable("http://localhost:11434/api/generate")
        assert base == "http://localhost:11434"

    def test_raises_on_connection_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(ConnectionError, match="Could not reach Ollama"):
                verify_ollama_reachable("http://127.0.0.1:11434/api/generate")


class TestTitlesFileHelpers:
    def test_append_and_parse_roundtrip(self, tmp_path):
        path = tmp_path / "titles.txt"
        append_title_line(path, "a.mp4", "One")
        append_title_line(path, "b.mp4", "Two: parts")
        assert filenames_in_titles_file(path) == {"a.mp4", "b.mp4"}
        text = path.read_text(encoding="utf-8")
        assert "a.mp4: One" in text
        assert "b.mp4: Two: parts" in text


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
