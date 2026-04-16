"""Guarantee exported canvases are exact 9:16, 4:5, and 3:4 (integer ratios, no float drift)."""

import pytest
from PIL import Image

from reelcribe.social_images import (
    INSTAGRAM_FEED_3_4,
    INSTAGRAM_FEED_4_5,
    VERTICAL_9x16,
    dimensions_match_ratio,
    reframe,
    reframe_contain,
    reframe_cover,
)

# Canonical (width, height) and (num, den) such that w/h == num/den exactly.
_CANVASES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    (VERTICAL_9x16, (9, 16)),
    (INSTAGRAM_FEED_4_5, (4, 5)),
    (INSTAGRAM_FEED_3_4, (3, 4)),
)


class TestDimensionsMatchRatio:
    def test_canonical_constants_are_exact_ratios(self):
        for (w, h), (rw, rh) in _CANVASES:
            assert dimensions_match_ratio(w, h, rw, rh), f"{w}x{h} vs {rw}:{rh}"

    def test_cross_multiply_examples(self):
        assert dimensions_match_ratio(1080, 1920, 9, 16)
        assert dimensions_match_ratio(1080, 1350, 4, 5)
        assert not dimensions_match_ratio(1080, 1920, 4, 5)


@pytest.mark.parametrize("src_size", [(1, 1), (16, 9), (1920, 1080), (4000, 3000), (1080, 1920)])
@pytest.mark.parametrize("fit", ["contain", "cover"])
@pytest.mark.parametrize("canvas,ratio", _CANVASES)
def test_reframe_always_outputs_exact_canvas_size(
    src_size: tuple[int, int],
    fit: str,
    canvas: tuple[int, int],
    ratio: tuple[int, int],
):
    w, h = src_size
    src = Image.new("RGB", (max(1, w), max(1, h)), (1, 2, 3))
    tw, th = canvas
    out = reframe(src, tw, th, fit=fit)
    assert out.size == canvas
    assert dimensions_match_ratio(tw, th, ratio[0], ratio[1])


def test_saved_png_round_trip_dimensions(tmp_path):
    from reelcribe.social_images import save_image

    src = Image.new("RGB", (1280, 720), (9, 9, 9))
    for canvas in (VERTICAL_9x16, INSTAGRAM_FEED_4_5):
        out = reframe_contain(src, *canvas)
        path = tmp_path / f"out_{canvas[0]}x{canvas[1]}.png"
        save_image(out, path)
        assert Image.open(path).size == canvas
