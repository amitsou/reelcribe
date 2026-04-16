"""Face-aware text placement for 1080×1920 Reels/Facebook thumbnails (Meta Suite).

Pure-Python zone scoring + optional MediaPipe face detection. No cloud APIs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Final, Literal

from PIL import Image, ImageDraw, ImageFont

from reelcribe.social_images import VERTICAL_9x16, dimensions_match_ratio

TextStyle = Literal["light_on_dark", "dark_on_light"]

# Normalized (left, top, right, bottom) in [0, 1] relative to full frame.
# Tuned for short titles in corners / upper band; avoids center (logo/mic).
_ZONE_NORM: Final[tuple[tuple[str, tuple[float, float, float, float]], ...]] = (
    ("top_right", (0.52, 0.04, 0.98, 0.30)),
    ("top_left", (0.02, 0.04, 0.48, 0.30)),
    ("mid_right", (0.54, 0.34, 0.98, 0.58)),
    ("mid_left", (0.02, 0.34, 0.46, 0.58)),
    ("bottom_right", (0.50, 0.72, 0.98, 0.96)),
    ("bottom_left", (0.02, 0.72, 0.48, 0.96)),
)

# Reject zone if more than this fraction of the zone overlaps inflated face.
_MAX_ZONE_FACE_OVERLAP: Final[float] = 0.12


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    width: int
    height: int

    def to_ltrb(self) -> tuple[int, int, int, int]:
        return (
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )


@dataclass
class ZoneScore:
    name: str
    bbox: tuple[int, int, int, int]  # left, top, right, bottom
    overlap_ratio: float
    valid: bool
    mean_luminance: float
    text_style: TextStyle


@dataclass
class AnalysisResult:
    canvas_width: int
    canvas_height: int
    faces: list[dict[str, int]]
    zones: list[ZoneScore]
    recommended_zone: str | None
    warnings: list[str]


def _clamp_box(
    l: int, t: int, r: int, b: int, w: int, h: int
) -> tuple[int, int, int, int]:
    return (
        max(0, min(l, w - 1)),
        max(0, min(t, h - 1)),
        max(1, min(r, w)),
        max(1, min(b, h)),
    )


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il = max(al, bl)
    it = max(at, bt)
    ir = min(ar, br)
    ib = min(ab, bb)
    if ir <= il or ib <= it:
        return 0
    return (ir - il) * (ib - it)


def _box_area(box: tuple[int, int, int, int]) -> int:
    l, t, r, b = box
    return max(0, r - l) * max(0, b - t)


def _inflate_face(face: FaceBox, margin: int, iw: int, ih: int) -> tuple[int, int, int, int]:
    l, t, r, b = face.to_ltrb()
    l -= margin
    t -= margin
    r += margin
    b += margin
    return _clamp_box(l, t, r, b, iw, ih)


def mean_luminance_region(im: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Mean grayscale value in *box* (left, top, right, bottom)."""
    l, t, r, b = box
    l, t, r, b = int(l), int(t), int(r), int(b)
    if r <= l or b <= t:
        return 128.0
    crop = im.crop((l, t, r, b)).convert("L")
    hist = crop.histogram()
    n = sum(hist)
    if n == 0:
        return 128.0
    return sum(i * hist[i] for i in range(256)) / n


def text_style_for_luminance(mean_l: float) -> TextStyle:
    """Dark background → light text; light background → dark text."""
    if mean_l < 128:
        return "light_on_dark"
    return "dark_on_light"


def clamp_point_to_zones(px: int, py: int, zones: list[ZoneScore]) -> tuple[int, int]:
    """Snap *px*, *py* to the nearest point inside suggested zone rectangles (valid first)."""
    rects = [z.bbox for z in zones if z.valid]
    if not rects:
        rects = [z.bbox for z in zones]
    if not rects:
        return px, py
    best_px, best_py = px, py
    best_d: float | None = None
    for l, t, r, b in rects:
        cx = max(l, min(px, r - 1))
        cy = max(t, min(py, b - 1))
        d = float((cx - px) ** 2 + (cy - py) ** 2)
        if best_d is None or d < best_d:
            best_d = d
            best_px, best_py = cx, cy
    return best_px, best_py


def zone_for_point(px: int, py: int, zones: list[ZoneScore]) -> ZoneScore | None:
    for z in zones:
        l, t, r, b = z.bbox
        if l <= px < r and t <= py < b:
            return z
    return None


def zone_by_name(name: str | None, zones: list[ZoneScore]) -> ZoneScore | None:
    if not name:
        return None
    for z in zones:
        if z.name == name:
            return z
    return None


# Colored spans: [#RRGGBB]text[/] or [#RRGGBB:#RRGGBB]text[/] (fill : stroke for outline).
_COLORED_SPAN_RE = re.compile(
    r"\[\s*#([0-9a-fA-F]{6})\s*(?::\s*#([0-9a-fA-F]{6}))?\s*\](.*?)\[/\]",
    re.DOTALL,
)


def has_colored_markup(text: str) -> bool:
    return "[#" in text


def parse_colored_markup(
    text: str,
) -> list[tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]]:
    """Split *text* into (chunk, fill_rgb_or_None, stroke_rgb_or_None).

    Untagged runs use (None, None) and inherit defaults when drawing.
    Tagged runs use explicit hex colors from ``[#RRGGBB]...[/]`` or
    ``[#fill_hex:stroke_hex]...[/]`` for per-span outline color.
    """
    out: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ] = []
    pos = 0
    for m in _COLORED_SPAN_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], None, None))
        fr, fg, fb = (int(m.group(1)[i : i + 2], 16) for i in (0, 2, 4))
        fill = (fr, fg, fb)
        stroke: tuple[int, int, int] | None = None
        if m.group(2):
            sr, sg, sb = (int(m.group(2)[i : i + 2], 16) for i in (0, 2, 4))
            stroke = (sr, sg, sb)
        out.append((m.group(3), fill, stroke))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], None, None))
    if not out:
        out.append((text, None, None))
    return out


def plain_text_from_markup(text: str) -> str:
    """Concatenate span contents (for wizards that re-wrap plain text)."""
    return "".join(seg[0] for seg in parse_colored_markup(text))


def _token_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text: str,
) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _line_height(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    return bbox[3] - bbox[1] + max(2, (bbox[3] - bbox[1]) // 8)


def _flatten_segments_to_tokens(
    segments: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ],
) -> list[
    tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
]:
    """Split segments into tokens (words / spaces) keeping colors per token."""
    tokens: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ] = []
    for seg_text, fill, stroke in segments:
        for part in re.split(r"(\s+)", seg_text):
            if part:
                tokens.append((part, fill, stroke))
    return tokens


def _wrap_tokens_to_lines(
    tokens: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[
    list[tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]]
]:
    if max_width < 40:
        return [tokens] if tokens else []
    lines: list[
        list[tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]]
    ] = []
    line: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ] = []
    cur_w = 0
    for tok, tf, ts in tokens:
        w = _token_width(draw, font, tok)
        if line and cur_w + w > max_width:
            lines.append(line)
            line = []
            cur_w = 0
        line.append((tok, tf, ts))
        cur_w += w
    if line:
        lines.append(line)
    return lines


def _line_pixel_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    line: list[
        tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
    ],
) -> int:
    return sum(_token_width(draw, font, t[0]) for t in line)


def draw_multicolor_text_block(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    cx: int,
    cy: int,
    max_width: int,
    default_fill: tuple[int, int, int],
    default_stroke: tuple[int, int, int],
    stroke_width: int,
) -> None:
    """Draw *text* centered at *(cx, cy)* with optional [#hex]…[/] spans."""
    segments = parse_colored_markup(text.strip())
    tokens = _flatten_segments_to_tokens(segments)
    if not tokens:
        return
    lines = _wrap_tokens_to_lines(tokens, draw, font, max_width)
    if not lines:
        return
    lh = _line_height(draw, font)
    line_widths = [_line_pixel_width(draw, font, ln) for ln in lines]
    block_w = max(line_widths)
    block_h = len(lines) * lh
    left = cx - block_w // 2
    top = cy - block_h // 2
    y = top
    sw = stroke_width if stroke_width > 0 else 0
    for line in lines:
        lw = _line_pixel_width(draw, font, line)
        x = left + (block_w - lw) // 2
        for tok, tf, ts in line:
            fill = tf if tf is not None else default_fill
            stroke_c = ts if ts is not None else default_stroke
            draw.text(
                (x, y),
                tok,
                font=font,
                fill=fill,
                stroke_width=sw,
                stroke_fill=stroke_c,
            )
            x += _token_width(draw, font, tok)
        y += lh


def _wrap_text_to_width(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
) -> str:
    text = text.strip()
    if not text or max_width < 40:
        return text
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    for w in words:
        test = " ".join(line + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line.append(w)
        else:
            if line:
                lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


_mp_face_detector: Any = None


def _get_mediapipe_face_detector() -> Any:
    """Lazy singleton: constructing FaceDetection is expensive; reuse across calls."""
    global _mp_face_detector
    if _mp_face_detector is not None:
        return _mp_face_detector
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "Face detection requires MediaPipe. Install: pip install 'reelcribe[reels-thumb]'"
        ) from exc
    if not hasattr(mp, "solutions"):
        raise ImportError(
            "This MediaPipe build has no `mediapipe.solutions`. "
            "Use: pip install 'mediapipe>=0.10.13,<0.10.30'"
        )
    _mp_face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )
    return _mp_face_detector


def detect_faces_mediapipe(rgb_array: Any) -> list[FaceBox]:
    """Run MediaPipe face detection on an H×W×3 uint8 RGB ndarray."""
    import numpy as np

    if not isinstance(rgb_array, np.ndarray) or rgb_array.ndim != 3:
        raise ValueError("Expected RGB numpy array H×W×3")
    h, w = rgb_array.shape[0], rgb_array.shape[1]
    detector = _get_mediapipe_face_detector()
    res = detector.process(rgb_array)
    faces: list[FaceBox] = []
    if not res.detections:
        return faces
    for det in res.detections:
        rel = det.location_data.relative_bounding_box
        left = int(rel.xmin * w)
        top = int(rel.ymin * h)
        width = max(1, int(rel.width * w))
        height = max(1, int(rel.height * h))
        faces.append(FaceBox(left=left, top=top, width=width, height=height))
    return faces


def analyze_thumbnail(
    image: Image.Image,
    *,
    faces: list[FaceBox] | None = None,
    face_margin_px: int | None = None,
) -> AnalysisResult:
    """Score text zones for a vertical thumbnail. Pass *faces* or detect via :func:`detect_faces_mediapipe`."""
    im = image.convert("RGB")
    iw, ih = im.size
    warnings: list[str] = []

    if not dimensions_match_ratio(iw, ih, 9, 16):
        warnings.append(
            f"Image is {iw}×{ih}; Meta Reels cover thumbnails are typically 1080×1920 (9:16)."
        )

    if face_margin_px is None:
        face_margin_px = max(24, int(0.02 * min(iw, ih)))

    face_list: list[FaceBox] = list(faces) if faces is not None else []
    if faces is None:
        try:
            import numpy as np

            arr = np.asarray(im)
            face_list = detect_faces_mediapipe(arr)
        except ImportError:
            warnings.append(
                "MediaPipe/NumPy not available; no face detection. "
                "Install: pip install 'reelcribe[reels-thumb]'"
            )
            face_list = []
        except Exception as exc:
            warnings.append(f"Face detection failed: {exc}")
            face_list = []

    if not face_list:
        warnings.append("No face boxes: recommendation uses zone order only (top_right first).")

    inflated: list[tuple[int, int, int, int]] = [
        _inflate_face(f, face_margin_px, iw, ih) for f in face_list
    ]

    zones_out: list[ZoneScore] = []
    for name, (nl, nt, nr, nb) in _ZONE_NORM:
        l = int(nl * iw)
        t = int(nt * ih)
        r = int(nr * iw)
        b = int(nb * ih)
        l, t, r, b = _clamp_box(l, t, r, b, iw, ih)
        zbox = (l, t, r, b)
        zarea = _box_area(zbox)
        overlap = 0
        for fbox in inflated:
            overlap += _intersection_area(zbox, fbox)
        overlap_ratio = (overlap / zarea) if zarea > 0 else 1.0
        valid = overlap_ratio <= _MAX_ZONE_FACE_OVERLAP
        mean_l = mean_luminance_region(im, zbox)
        zones_out.append(
            ZoneScore(
                name=name,
                bbox=zbox,
                overlap_ratio=overlap_ratio,
                valid=valid,
                mean_luminance=mean_l,
                text_style=text_style_for_luminance(mean_l),
            )
        )

    recommended: str | None = None
    for z in zones_out:
        if z.valid:
            recommended = z.name
            break
    if recommended is None and zones_out:
        # Fallback: smallest overlap
        best = min(zones_out, key=lambda z: z.overlap_ratio)
        recommended = best.name
        warnings.append("All zones overlap the face margin; picked least overlap.")

    face_dicts = [
        {"left": f.left, "top": f.top, "width": f.width, "height": f.height}
        for f in face_list
    ]

    return AnalysisResult(
        canvas_width=iw,
        canvas_height=ih,
        faces=face_dicts,
        zones=zones_out,
        recommended_zone=recommended,
        warnings=warnings,
    )


def analysis_to_json(result: AnalysisResult) -> str:
    payload: dict[str, Any] = {
        "canvas": {"width": result.canvas_width, "height": result.canvas_height},
        "faces": result.faces,
        "recommended_zone": result.recommended_zone,
        "warnings": result.warnings,
        "zones": [
            {
                "name": z.name,
                "bbox": {"left": z.bbox[0], "top": z.bbox[1], "right": z.bbox[2], "bottom": z.bbox[3]},
                "valid": z.valid,
                "overlap_ratio": round(z.overlap_ratio, 4),
                "mean_luminance": round(z.mean_luminance, 2),
                "text_style": z.text_style,
            }
            for z in result.zones
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _max_text_width_for_zone(
    layout_zone: ZoneScore,
    pad: int,
    iw: int,
    text_max_width_mult: float,
) -> int:
    """Wrap width: zone box × *text_max_width_mult*, capped to full canvas minus padding."""
    l, t, r, b = layout_zone.bbox
    base_w = max(40, (r - l) - 2 * pad)
    cap = max(40, iw - 2 * pad)
    mult = max(0.5, min(float(text_max_width_mult), 4.0))
    return max(40, min(int(base_w * mult), cap))


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TTF/TTC for drawing; fall back to bundled candidates if *font_path* fails."""
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return _try_load_font(size)


def render_preview(
    image: Image.Image,
    result: AnalysisResult,
    *,
    text: str = "",
    zone_name: str | None = None,
    draw_face: bool = True,
    draw_zones: bool = True,
    text_center_xy: tuple[int, int] | None = None,
    font_path: str | None = None,
    font_size: int = 42,
    fill_rgb: tuple[int, int, int] | None = None,
    stroke_rgb: tuple[int, int, int] | None = None,
    stroke_width: int = 3,
    auto_colors_from_zone: bool = True,
    highlight_zone_name: str | None = None,
    text_max_width_mult: float = 1.0,
) -> Image.Image:
    """Draw optional face margin, zone rectangles, and text on a copy of *image*.

    If *text_center_xy* is set, text is centered at that point (after wrapping to the
    zone that contains the point, or the nearest zone). Otherwise text is centered in
    *zone_name* as before.
    """
    im = image.convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw_o = ImageDraw.Draw(overlay)

    highlight = highlight_zone_name or zone_name or result.recommended_zone
    zone = zone_by_name(zone_name, result.zones)
    if zone is None and result.zones:
        zone = result.zones[0]

    iw, ih = im.size
    if draw_face and result.faces:
        margin = max(24, int(0.02 * min(iw, ih)))
        for fd in result.faces:
            fb = FaceBox(fd["left"], fd["top"], fd["width"], fd["height"])
            l, t, r, b = _inflate_face(fb, margin, iw, ih)
            draw_o.rectangle([l, t, r, b], outline=(255, 80, 80, 220), width=4)

    if draw_zones:
        for z in result.zones:
            l, t, r, b = z.bbox
            color = (80, 255, 120, 180) if z.valid else (255, 180, 80, 120)
            w = 5 if z.name == highlight else 2
            draw_o.rectangle([l, t, r, b], outline=color, width=w)

    im = Image.alpha_composite(im, overlay).convert("RGB")
    draw = ImageDraw.Draw(im)

    if not text or not text.strip():
        return im

    font = load_font(font_path, font_size)
    pad = max(8, font_size // 8)

    layout_zone: ZoneScore | None = zone
    if text_center_xy is not None:
        cx, cy = text_center_xy
        zhit = zone_for_point(cx, cy, result.zones)
        if zhit is not None:
            layout_zone = zhit
        elif result.zones:
            # Wrap using nearest zone bbox
            nx, ny = clamp_point_to_zones(cx, cy, result.zones)
            zhit = zone_for_point(nx, ny, result.zones)
            layout_zone = zhit or result.zones[0]

    if layout_zone is None:
        return im

    max_w = _max_text_width_for_zone(layout_zone, pad, iw, text_max_width_mult)

    if text_center_xy is None:
        cx = (l + r) // 2
        cy = (t + b) // 2
    else:
        cx, cy = text_center_xy

    if auto_colors_from_zone and fill_rgb is None and stroke_rgb is None:
        if layout_zone.text_style == "light_on_dark":
            fill_rgb = (250, 250, 250)
            stroke_rgb = (0, 0, 0)
        else:
            fill_rgb = (28, 28, 32)
            stroke_rgb = (255, 255, 255)
    else:
        fill_rgb = fill_rgb or (250, 250, 250)
        stroke_rgb = stroke_rgb or (0, 0, 0)

    sw = stroke_width if stroke_width > 0 else 0
    draw_multicolor_text_block(
        draw,
        text=text,
        font=font,
        cx=cx,
        cy=cy,
        max_width=max_w,
        default_fill=fill_rgb,
        default_stroke=stroke_rgb,
        stroke_width=sw,
    )

    return im


def render_text_layer_rgba(
    size: tuple[int, int],
    result: AnalysisResult,
    *,
    text: str,
    text_center_xy: tuple[int, int],
    zone_name: str | None,
    font_path: str | None,
    font_size: int,
    fill_rgb: tuple[int, int, int] | None,
    stroke_rgb: tuple[int, int, int] | None,
    stroke_width: int,
    auto_colors_from_zone: bool,
    text_max_width_mult: float = 1.0,
) -> Image.Image:
    """Transparent RGBA image of *size* with only the text (for UI overlay)."""
    w, h = size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if not text.strip():
        return out
    draw = ImageDraw.Draw(out)
    zone = zone_by_name(zone_name, result.zones)
    if zone is None and result.zones:
        zone = result.zones[0]
    if zone is None:
        return out
    pad = max(8, font_size // 8)
    font = load_font(font_path, font_size)
    layout_zone = zone
    cx, cy = text_center_xy
    zhit = zone_for_point(cx, cy, result.zones)
    if zhit is not None:
        layout_zone = zhit
    elif result.zones:
        nx, ny = clamp_point_to_zones(cx, cy, result.zones)
        zhit = zone_for_point(nx, ny, result.zones)
        layout_zone = zhit or result.zones[0]
    max_w = _max_text_width_for_zone(layout_zone, pad, w, text_max_width_mult)
    df: tuple[int, int, int]
    ds: tuple[int, int, int]
    if auto_colors_from_zone and fill_rgb is None and stroke_rgb is None:
        if layout_zone.text_style == "light_on_dark":
            df, ds = (250, 250, 250), (0, 0, 0)
        else:
            df, ds = (28, 28, 32), (255, 255, 255)
    else:
        df = fill_rgb or (250, 250, 250)
        ds = stroke_rgb or (0, 0, 0)
    sw = stroke_width if stroke_width > 0 else 0
    draw_multicolor_text_block(
        draw,
        text=text,
        font=font,
        cx=cx,
        cy=cy,
        max_width=max_w,
        default_fill=df,
        default_stroke=ds,
        stroke_width=sw,
    )
    return out


def paste_logo_centered(
    image: Image.Image,
    logo: Image.Image,
    *,
    center_xy: tuple[int, int],
    width_px: int,
) -> Image.Image:
    """Composite *logo* onto *image*, centered at *center_xy*, scaled to width *width_px*."""
    im = image.convert("RGBA")
    lg = logo.convert("RGBA")
    ow, oh = lg.size
    if ow <= 0 or oh <= 0:
        return im.convert("RGB")
    iw, ih = im.size
    w = max(1, min(int(width_px), iw))
    h = max(1, int(oh * (w / ow)))
    h = min(h, ih)
    logo_s = lg.resize((w, h), Image.Resampling.LANCZOS)
    cx, cy = center_xy
    x = int(cx - w // 2)
    y = int(cy - h // 2)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer.paste(logo_s, (x, y), logo_s)
    return Image.alpha_composite(im, layer).convert("RGB")


def ensure_1080x1920(image: Image.Image) -> tuple[Image.Image, bool]:
    """If image is not 1080×1920, return letterboxed :func:`reelcribe.social_images.reframe_contain` result."""
    from reelcribe.social_images import reframe_contain

    w, h = image.size
    if (w, h) == VERTICAL_9x16:
        return image, False
    out = reframe_contain(image, *VERTICAL_9x16)
    return out, True
