"""Tests for reelcribe.social_images."""

from pathlib import Path

import pytest
from PIL import Image

from reelcribe.social_images import (
    SAFE_ZONE_BBOX_9x16,
    SAFE_ZONE_HEIGHT_PX,
    SAFE_ZONE_TOP_PX,
    VERTICAL_9x16,
    find_image_files,
    instagram_feed_size,
    load_image,
    reframe,
    reframe_contain,
    reframe_cover,
)


class TestSafeZoneConstants:
    def test_bbox_matches_centered_1080x1350_in_1920(self):
        assert VERTICAL_9x16 == (1080, 1920)
        assert SAFE_ZONE_HEIGHT_PX == 1350
        assert SAFE_ZONE_TOP_PX == 285
        left, top, right, bottom = SAFE_ZONE_BBOX_9x16
        assert (left, top, right, bottom) == (0, 285, 1080, 1635)
        assert bottom - top == SAFE_ZONE_HEIGHT_PX


class TestReframeCover:
    def test_output_dimensions(self):
        src = Image.new("RGB", (1920, 1080), color=(128, 64, 32))
        out = reframe_cover(src, 1080, 1920)
        assert out.size == (1080, 1920)

    def test_portrait_source(self):
        src = Image.new("RGB", (1080, 1920), color=(0, 0, 0))
        out = reframe_cover(src, 1080, 1350)
        assert out.size == (1080, 1350)


class TestReframeContain:
    def test_output_dimensions(self):
        src = Image.new("RGB", (1920, 1080), color=(128, 64, 32))
        out = reframe_contain(src, 1080, 1920)
        assert out.size == (1080, 1920)

    def test_landscape_has_letterbox_bars(self):
        """16:9 inside 9:16 canvas leaves black top/bottom."""
        src = Image.new("RGB", (1920, 1080), color=(200, 50, 50))
        out = reframe_contain(src, 1080, 1920, bg=(0, 0, 0))
        px = out.load()
        assert px[0, 0] == (0, 0, 0)
        mid_y = 1920 // 2
        assert px[540, mid_y][0] > 100

    def test_vertical_align_top(self):
        src = Image.new("RGB", (1920, 1080), (255, 0, 0))
        out = reframe_contain(src, 1080, 1920, vertical_align="top")
        px = out.load()
        assert px[540, 0][0] > 200
        assert px[0, 1919] == (0, 0, 0)


class TestReframeDispatch:
    def test_contain_matches_reframe_contain(self):
        src = Image.new("RGB", (1920, 1080), (1, 2, 3))
        a = reframe_contain(src, 1080, 1920)
        b = reframe(src, 1080, 1920, fit="contain")
        assert a.tobytes() == b.tobytes()

    def test_cover_matches_reframe_cover(self):
        src = Image.new("RGB", (1920, 1080), (1, 2, 3))
        a = reframe_cover(src, 1080, 1920)
        b = reframe(src, 1080, 1920, fit="cover")
        assert a.tobytes() == b.tobytes()


class TestInstagramFeedSize:
    def test_45(self):
        assert instagram_feed_size("4:5") == (1080, 1350)

    def test_34(self):
        assert instagram_feed_size("3:4") == (1080, 1440)

    def test_invalid(self):
        with pytest.raises(ValueError):
            instagram_feed_size("16:9")


class TestFindImageFiles:
    def test_finds_png(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "readme.txt").write_text("x")
        found = find_image_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "a.png"


class TestRoundTripFile:
    def test_save_and_dimensions_in_pipeline(self, tmp_path):
        from reelcribe.social_images import save_image

        src_path = tmp_path / "wide.png"
        Image.new("RGB", (1600, 900), (255, 0, 0)).save(src_path)

        im = load_image(src_path)
        cropped = reframe_cover(im, *VERTICAL_9x16)
        out_path = tmp_path / "out" / "vertical.png"
        save_image(cropped, out_path)
        assert out_path.exists()
        assert Image.open(out_path).size == VERTICAL_9x16
