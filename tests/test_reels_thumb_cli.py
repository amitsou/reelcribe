"""Tests for reelcribe.reels_thumb_cli."""

from pathlib import Path

from PIL import Image

from reelcribe.reels_thumb_cli import main


def test_default_converts_landscape_to_1080x1920(tmp_path):
    """Non–9:16 input is normalized before analysis."""
    png = tmp_path / "wide.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(png)
    rc = main(["-i", str(png), "--json"])
    assert rc == 0


def test_strict_1080p_fails_on_wrong_size(tmp_path):
    png = tmp_path / "wide.png"
    Image.new("RGB", (1920, 1080), (0, 0, 0)).save(png)
    rc = main(["-i", str(png), "--strict-1080p", "--json"])
    assert rc == 1


def test_ui_returns_1_when_tkinter_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "reelcribe.reels_thumb_cli._tkinter_available",
        lambda: False,
    )
    png = tmp_path / "a.png"
    Image.new("RGB", (1080, 1920), (0, 0, 0)).save(png)
    rc = main(["-i", str(png), "--ui"])
    assert rc == 1
