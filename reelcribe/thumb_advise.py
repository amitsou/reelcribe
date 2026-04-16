"""Local Ollama vision critique for Reels-style thumbnails: JSON advice + guide overlay.

**Models:** Plain ``llama3`` in Ollama is **text-only** and cannot see your PNG. Use a
**vision** model for image-based feedback, for example::

    ollama pull llama3.2-vision
    # or: llava, moondream, etc.

Then pass ``--ollama-model llama3.2-vision`` (or your tag from ``ollama list``).

Large images are downscaled (longest side) before base64 to keep requests smaller; see
``--max-image-side`` on the CLI.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from reelcribe.llm import verify_ollama_reachable

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_VISION_MODEL = "llama3.2-vision"

ADVICE_JSON_SCHEMA_HINT = """
Return a single JSON object only (no markdown fences), with these keys:
- title_anchor_norm: object with "x" and "y" floats in [0,1] for ideal title block center (9:16 frame).
- logo_anchor_norm: object with "x" and "y" floats in [0,1] for ideal logo center.
- title_colors_hex: object with optional "fill" and "outline" as "#RRGGBB" strings.
- font_suggestions: array of 1-3 short font family names (strings).
- logo_placement_note: one short string (e.g. "bottom-right, clear of face").
- notes_el: array of 2-6 short bullet strings in Greek or English with concrete fixes.
- confidence: float in [0,1] for how sure you are.
"""


def build_advise_system_prompt(language: str) -> str:
    lang = language.strip() or "English"
    return (
        "You are a senior social-media creative director and photographer specializing in "
        "vertical 9:16 Meta Reels / Shorts thumbnails: hierarchy, contrast, safe margins, "
        "readability at small sizes, and not obscuring faces. "
        f"Write notes_el in {lang} when possible. "
        "Respond with ONLY valid JSON matching the schema described by the user. "
        "No markdown, no code fences, no text before or after the JSON object."
    )


def build_advise_user_prompt() -> str:
    return (
        "Analyze this thumbnail image. Suggest where the main title block and brand logo "
        "should sit for maximum clarity and marketing impact on mobile feeds. "
        "Consider face/subject placement, negative space, and thumb-stopping contrast.\n\n"
        + ADVICE_JSON_SCHEMA_HINT
    )


def image_to_base64_png(im: Image.Image, max_side: int) -> str:
    """Resize so longest side <= *max_side*, encode as PNG base64 (no data URI prefix)."""
    w, h = im.size
    if max(w, h) > max_side and max_side > 0:
        scale = max_side / float(max(w, h))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ollama_chat_with_image(
    *,
    model: str,
    chat_url: str,
    image_b64: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 180.0,
) -> str:
    """POST to Ollama ``/api/chat`` with one user message and ``images`` array."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_b64],
            },
        ],
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama chat API at {chat_url!r}. Is Ollama running?"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON: {raw[:400]!r}") from exc
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Unexpected Ollama chat response: {raw[:400]!r}")
    return content.strip()


def parse_advice_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output; tolerate ```json fences and leading prose."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    blob = t[start : end + 1]
    return json.loads(blob)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))


def _anchor_from_advice(data: dict[str, Any], key: str) -> tuple[float, float] | None:
    raw = data.get(key)
    if not isinstance(raw, dict):
        return None
    return _clamp01(raw.get("x")), _clamp01(raw.get("y"))


def render_advice_overlay(
    image: Image.Image,
    advice: dict[str, Any],
    *,
    title_label: str = "Title",
    logo_label: str = "Logo",
) -> Image.Image:
    """Draw semi-transparent markers for title/logo anchors on a copy of *image*."""
    im = image.convert("RGBA")
    w, h = im.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(14, min(w, h) // 42)
    font = ImageFont.load_default()
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        try:
            font = ImageFont.truetype(path, size=size)
            break
        except OSError:
            continue

    title_pt = _anchor_from_advice(advice, "title_anchor_norm")
    logo_pt = _anchor_from_advice(advice, "logo_anchor_norm")

    def px(norm: tuple[float, float]) -> tuple[int, int]:
        return int(norm[0] * (w - 1)), int(norm[1] * (h - 1))

    r = max(8, min(w, h) // 60)
    if title_pt:
        tx, ty = px(title_pt)
        draw.ellipse(
            (tx - r, ty - r, tx + r, ty + r),
            outline=(0, 200, 255, 255),
            width=max(2, r // 4),
        )
        draw.rectangle((tx + r + 2, ty - r, tx + r + 220, ty + r), fill=(0, 0, 0, 180))
        draw.text((tx + r + 6, ty - r + 2), title_label, fill=(255, 255, 255, 255), font=font)
    if logo_pt:
        lx, ly = px(logo_pt)
        draw.rectangle(
            (lx - r * 2, ly - r * 2, lx + r * 2, ly + r * 2),
            outline=(255, 120, 0, 255),
            width=max(2, r // 4),
        )
        draw.rectangle((lx + r * 2 + 2, ly - r, lx + r * 2 + 160, ly + r), fill=(0, 0, 0, 180))
        draw.text((lx + r * 2 + 6, ly - r + 2), logo_label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(im, overlay)
    return out.convert("RGB")


def run_advise_pipeline(
    image_path: Path,
    *,
    model: str,
    chat_url: str,
    language: str,
    max_image_side: int,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    """Load image, call Ollama, parse JSON, return (advice_dict, raw_model_text)."""
    verify_ollama_reachable(chat_url)
    with Image.open(image_path) as im0:
        im = im0.copy()
    b64 = image_to_base64_png(im, max_image_side)
    system_p = build_advise_system_prompt(language)
    user_p = build_advise_user_prompt()
    raw_text = ollama_chat_with_image(
        model=model,
        chat_url=chat_url,
        image_b64=b64,
        system_prompt=system_p,
        user_prompt=user_p,
        timeout=timeout,
    )
    advice = parse_advice_json(raw_text)
    return advice, raw_text


def advice_to_markdown(advice: dict[str, Any]) -> str:
    """Human-readable summary from parsed advice JSON."""
    lines: list[str] = ["# Thumbnail layout advice", ""]
    ta = advice.get("title_anchor_norm")
    la = advice.get("logo_anchor_norm")
    if isinstance(ta, dict):
        lines.append(f"- **Title anchor (norm):** x={ta.get('x')}, y={ta.get('y')}")
    if isinstance(la, dict):
        lines.append(f"- **Logo anchor (norm):** x={la.get('x')}, y={la.get('y')}")
    tc = advice.get("title_colors_hex")
    if isinstance(tc, dict):
        lines.append(f"- **Title colors:** {tc}")
    fs = advice.get("font_suggestions")
    if isinstance(fs, list):
        lines.append(f"- **Font suggestions:** {', '.join(str(x) for x in fs)}")
    note = advice.get("logo_placement_note")
    if isinstance(note, str) and note.strip():
        lines.append(f"- **Logo note:** {note.strip()}")
    conf = advice.get("confidence")
    if conf is not None:
        lines.append(f"- **Confidence:** {conf}")
    lines.append("")
    lines.append("## Notes")
    notes = advice.get("notes_el")
    if isinstance(notes, list):
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append("- _(none)_")
    lines.append("")
    return "\n".join(lines)


def chat_base_url(chat_url: str) -> str:
    """``http://host:port`` from a full chat URL (for docs / logging)."""
    p = urlparse(chat_url)
    return f"{p.scheme}://{p.netloc}"
