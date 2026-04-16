"""Tests for reelcribe.thumb_advise (Ollama chat, JSON parse, overlay)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageChops

from reelcribe.thumb_advise import (
    chat_base_url,
    ollama_chat_with_image,
    parse_advice_json,
    render_advice_overlay,
    run_advise_pipeline,
)


def _chat_mock_response(content: str) -> MagicMock:
    raw = json.dumps({"message": {"content": content}}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestParseAdviceJson:
    def test_plain_object(self):
        text = '{"title_anchor_norm":{"x":0.1,"y":0.2},"confidence":0.5}'
        d = parse_advice_json(text)
        assert d["confidence"] == 0.5
        assert d["title_anchor_norm"]["x"] == 0.1

    def test_json_fence(self):
        text = 'Here:\n```json\n{"a": 1}\n```\n'
        assert parse_advice_json(text) == {"a": 1}

    def test_leading_prose(self):
        text = 'Sure! {"notes_el": ["x"], "confidence": 1}'
        d = parse_advice_json(text)
        assert d["notes_el"] == ["x"]

    def test_raises_when_no_object(self):
        with pytest.raises(ValueError, match="No JSON object"):
            parse_advice_json("no braces here")


class TestOllamaChatWithImage:
    def test_returns_message_content(self):
        mock_resp = _chat_mock_response("  hello  ")
        with patch("reelcribe.thumb_advise.urllib.request.urlopen", return_value=mock_resp):
            out = ollama_chat_with_image(
                model="m",
                chat_url="http://localhost:11434/api/chat",
                image_b64="abc",
                system_prompt="s",
                user_prompt="u",
                timeout=5.0,
            )
        assert out == "hello"

    def test_connection_error_on_url_error(self):
        import urllib.error

        with patch(
            "reelcribe.thumb_advise.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(ConnectionError, match="Could not reach Ollama"):
                ollama_chat_with_image(
                    model="m",
                    chat_url="http://localhost:11434/api/chat",
                    image_b64="x",
                    system_prompt="s",
                    user_prompt="u",
                )


class TestRenderAdviceOverlay:
    def test_output_size_matches_input(self):
        im = Image.new("RGB", (80, 120), color=(240, 240, 240))
        advice = {
            "title_anchor_norm": {"x": 0.5, "y": 0.25},
            "logo_anchor_norm": {"x": 0.9, "y": 0.9},
        }
        out = render_advice_overlay(im, advice)
        assert out.size == (80, 120)
        assert out.mode == "RGB"

    def test_overlay_changes_image(self):
        """Markers and labels should alter at least some pixels vs a flat white input."""
        w, h = 50, 40
        im = Image.new("RGB", (w, h), color=(255, 255, 255))
        advice = {
            "title_anchor_norm": {"x": 0.0, "y": 0.0},
            "logo_anchor_norm": {"x": 1.0, "y": 1.0},
        }
        out = render_advice_overlay(im, advice)
        diff = ImageChops.difference(out, im)
        assert diff.getbbox() is not None

    def test_clamps_out_of_range_anchors(self):
        im = Image.new("RGB", (20, 20), color=(255, 255, 255))
        advice = {
            "title_anchor_norm": {"x": -1.0, "y": 2.0},
            "logo_anchor_norm": {"x": "bad", "y": None},
        }
        out = render_advice_overlay(im, advice)
        assert out.size == (20, 20)


class TestRunAdvisePipeline:
    def test_end_to_end_mocked(self, tmp_path: Path):
        p = tmp_path / "thumb.png"
        Image.new("RGB", (10, 10), color="gray").save(p)
        raw = (
            '{"title_anchor_norm":{"x":0.5,"y":0.5},'
            '"logo_anchor_norm":{"x":0.2,"y":0.8},'
            '"title_colors_hex":{"fill":"#000000"},'
            '"font_suggestions":["Inter"],'
            '"logo_placement_note":"corner",'
            '"notes_el":["α"],'
            '"confidence":0.88}'
        )
        with patch("reelcribe.thumb_advise.verify_ollama_reachable"), patch(
            "reelcribe.thumb_advise.ollama_chat_with_image",
            return_value=raw,
        ):
            advice, text = run_advise_pipeline(
                p,
                model="llama3.2-vision",
                chat_url="http://127.0.0.1:11434/api/chat",
                language="Greek",
                max_image_side=1024,
                timeout=30.0,
            )
        assert text == raw
        assert advice["confidence"] == 0.88
        assert advice["notes_el"] == ["α"]


def test_chat_base_url_strips_path():
    assert chat_base_url("http://localhost:11434/api/chat") == "http://localhost:11434"
