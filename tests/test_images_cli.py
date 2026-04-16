"""Tests for reelcribe.images_cli."""

from pathlib import Path

from PIL import Image

from reelcribe.images_cli import main


class TestImagesCliMain:
    def test_returns_1_if_input_missing(self, tmp_path):
        rc = main(["-i", str(tmp_path / "nope")])
        assert rc == 1

    def test_returns_1_if_input_not_image_file(self, tmp_path):
        bad = tmp_path / "x.txt"
        bad.write_text("nope")
        rc = main(["-i", str(bad)])
        assert rc == 1

    def test_returns_0_if_no_images(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        rc = main(["-i", str(in_dir)])
        assert rc == 0

    def test_writes_two_ratio_subfolders(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        Image.new("RGB", (1920, 1080), (10, 20, 30)).save(in_dir / "thumb.png")
        out_dir = tmp_path / "out"

        rc = main(["-i", str(in_dir), "-o", str(out_dir)])
        assert rc == 0
        assert (out_dir / "9x16" / "thumb.png").exists()
        assert (out_dir / "4x5" / "thumb.png").exists()

        from PIL import Image as PILImage

        assert PILImage.open(out_dir / "9x16" / "thumb.png").size == (1080, 1920)
        assert PILImage.open(out_dir / "4x5" / "thumb.png").size == (1080, 1350)

    def test_verify_flag_passes_when_dimensions_exact(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        Image.new("RGB", (1920, 1080), (10, 20, 30)).save(in_dir / "thumb.png")
        out_dir = tmp_path / "out"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "--verify"])
        assert rc == 0

    def test_fit_cover_writes_same_dimensions(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        Image.new("RGB", (1920, 1080), (10, 20, 30)).save(in_dir / "thumb.png")
        out_dir = tmp_path / "out"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "--fit", "cover"])
        assert rc == 0
        from PIL import Image as PILImage

        assert PILImage.open(out_dir / "9x16" / "thumb.png").size == (1080, 1920)
        assert PILImage.open(out_dir / "4x5" / "thumb.png").size == (1080, 1350)

    def test_feed_3x4_folder(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        Image.new("RGB", (1920, 1080), (0, 0, 0)).save(in_dir / "a.png")
        out_dir = tmp_path / "out"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "--feed-aspect", "3:4"])
        assert rc == 0
        assert (out_dir / "3x4" / "a.png").exists()
        from PIL import Image as PILImage

        assert PILImage.open(out_dir / "3x4" / "a.png").size == (1080, 1440)

    def test_single_file_default_output_next_to_file(self, tmp_path):
        folder = tmp_path / "Thumbnail"
        folder.mkdir()
        png = folder / "episode_thumb.png"
        Image.new("RGB", (1920, 1080), (5, 5, 5)).save(png)

        rc = main(["-i", str(png)])
        assert rc == 0
        assert (folder / "9x16" / "episode_thumb.png").exists()
        assert (folder / "4x5" / "episode_thumb.png").exists()
