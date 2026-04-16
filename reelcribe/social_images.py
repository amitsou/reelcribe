"""Resize still images to target aspect ratios: cover-crop or letterboxed contain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required for social image reframing. Install with: pip install Pillow"
    ) from exc

SUPPORTED_IMAGE_EXTENSIONS: Final = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
)

# Output sizes (width x height), center cover-crop. One vertical size serves
# TikTok, Instagram Reels, YouTube Shorts, etc. (all 9:16 at 1080 wide).
VERTICAL_9x16: Final[tuple[int, int]] = (1080, 1920)
TIKTOK_SIZE = VERTICAL_9x16  # alias for tests / callers
INSTAGRAM_REELS_SIZE = VERTICAL_9x16
INSTAGRAM_FEED_4_5: Final[tuple[int, int]] = (1080, 1350)
INSTAGRAM_FEED_3_4: Final[tuple[int, int]] = (1080, 1440)
# Same pixels as feed 3:4; use this name when checking "profile grid" preview crops.
GRID_PREVIEW_3x4: Final[tuple[int, int]] = INSTAGRAM_FEED_3_4


def dimensions_match_ratio(
    width: int, height: int, ratio_w: int, ratio_h: int
) -> bool:
    """Return True iff *width*/*height* equals *ratio_w*/*ratio_h* as exact rationals.

    Uses integer cross-multiplication (no floating point), so 1080×1920 vs 9:16 is
    exact: ``width * ratio_h == height * ratio_w``.
    """
    if width < 1 or height < 1 or ratio_w < 1 or ratio_h < 1:
        return False
    return width * ratio_h == height * ratio_w


# --- Safe zone (for design / automation), relative to a 1080x1920 canvas ---
# Instagram full-screen Reels are 9:16, but the profile grid often shows a tighter
# vertical crop (e.g. 3:4). Keeping faces and text in a *central* 1080x1350 band
# reduces the chance they are cut off in grid or hidden behind UI (tabs, captions).
# Top/bottom inset: (1920 - 1350) / 2 = 285 px.
SAFE_ZONE_HEIGHT_PX: Final[int] = 1350
SAFE_ZONE_TOP_PX: Final[int] = (VERTICAL_9x16[1] - SAFE_ZONE_HEIGHT_PX) // 2
# Pillow bbox (left, upper, right, lower) for the safe rectangle in pixel coordinates.
SAFE_ZONE_BBOX_9x16: Final[tuple[int, int, int, int]] = (
    0,
    SAFE_ZONE_TOP_PX,
    VERTICAL_9x16[0],
    SAFE_ZONE_TOP_PX + SAFE_ZONE_HEIGHT_PX,
)


def reframe_cover(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale with \"cover\", then center-crop to *target_w* x *target_h*.

    Cuts off edges. Bad for wide YouTube thumbnails with subjects on the left/right.
    """
    if target_w < 1 or target_h < 1:
        raise ValueError("target dimensions must be positive")

    src_w, src_h = image.size
    if src_w < 1 or src_h < 1:
        raise ValueError("source image has invalid size")

    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return resized.crop((left, top, right, bottom))


def reframe_contain(
    image: Image.Image,
    target_w: int,
    target_h: int,
    *,
    bg: tuple[int, int, int] = (0, 0, 0),
    vertical_align: str = "center",
) -> Image.Image:
    """Fit the entire image inside *target_w* x *target_h* with letterboxing.

    Use this for landscape thumbnails with people on opposite sides: nothing is cropped,
    but you may get black (or *bg*) bars top/bottom or left/right.
    """
    if target_w < 1 or target_h < 1:
        raise ValueError("target dimensions must be positive")
    if vertical_align not in ("top", "center", "bottom"):
        raise ValueError("vertical_align must be top, center, or bottom")

    src_w, src_h = image.size
    if src_w < 1 or src_h < 1:
        raise ValueError("source image has invalid size")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if resized.mode not in ("RGB", "RGBA"):
        resized = resized.convert("RGBA")

    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - new_w) // 2
    if vertical_align == "center":
        y = (target_h - new_h) // 2
    elif vertical_align == "top":
        y = 0
    else:
        y = target_h - new_h

    if resized.mode == "RGBA":
        canvas.paste(resized, (x, y), resized)
    else:
        canvas.paste(resized, (x, y))
    return canvas


def reframe(
    image: Image.Image,
    target_w: int,
    target_h: int,
    *,
    fit: str = "contain",
    vertical_align: str = "center",
) -> Image.Image:
    """Dispatch to :func:`reframe_contain` or :func:`reframe_cover` based on *fit*."""
    if fit == "contain":
        return reframe_contain(
            image, target_w, target_h, vertical_align=vertical_align
        )
    if fit == "cover":
        return reframe_cover(image, target_w, target_h)
    raise ValueError(f"fit must be 'contain' or 'cover', not {fit!r}")


def _prepare_rgba_for_export(image: Image.Image, fmt: str) -> Image.Image:
    """Flatten alpha to white if saving as JPEG."""
    if fmt.upper() == "JPEG" and image.mode in ("RGBA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        return background
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA")
    return image


def save_image(image: Image.Image, path: Path) -> None:
    """Write *image* to *path*; format follows the file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        im = _prepare_rgba_for_export(image, "JPEG")
        im.save(path, "JPEG", quality=95, optimize=True)
    elif ext == ".png":
        image.save(path, "PNG", optimize=True)
    elif ext == ".webp":
        im = image.convert("RGBA") if image.mode not in ("RGB", "RGBA") else image
        im.save(path, "WEBP", quality=90)
    elif ext == ".bmp":
        image.convert("RGB").save(path, "BMP")
    else:
        image.save(path, "PNG", optimize=True)


def load_image(path: Path) -> Image.Image:
    """Open an image file (RGB/RGBA as returned by Pillow)."""
    with Image.open(path) as im:
        return im.copy()


def find_image_files(input_dir: Path) -> list[Path]:
    """List supported image files in *input_dir* (non-recursive), sorted by name."""
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    logger.debug("Found %d image file(s) in %s", len(files), input_dir)
    return files


def instagram_feed_size(feed_aspect: str) -> tuple[int, int]:
    """Return (width, height) for Instagram feed crop."""
    if feed_aspect == "4:5":
        return INSTAGRAM_FEED_4_5
    if feed_aspect == "3:4":
        return INSTAGRAM_FEED_3_4
    raise ValueError(f"unsupported feed aspect: {feed_aspect!r}")
