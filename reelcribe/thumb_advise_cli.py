"""CLI: Ollama vision critique for a thumbnail PNG → JSON + guide overlay + markdown."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from reelcribe.social_images import SUPPORTED_IMAGE_EXTENSIONS
from reelcribe.thumb_advise import (
    DEFAULT_OLLAMA_CHAT_URL,
    DEFAULT_VISION_MODEL,
    advice_to_markdown,
    chat_base_url,
    render_advice_overlay,
    run_advise_pipeline,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reelcribe-thumb-advise",
        description=(
            "Send a thumbnail PNG to a local Ollama **vision** model and write layout advice "
            "(JSON + markdown) plus an optional guide image with anchor markers. "
            "Text-only models such as plain `llama3` cannot see the image — use e.g. "
            "`llama3.2-vision` or `llava`."
        ),
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Input PNG/JPEG/WebP thumbnail.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for outputs (default: same directory as input).",
    )
    p.add_argument(
        "--guide-image",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write guide PNG with title/logo markers (default: <stem>_advise_guide.png in output dir).",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_CHAT_URL,
        help=f"Ollama chat API URL. Default: {DEFAULT_OLLAMA_CHAT_URL}",
    )
    p.add_argument(
        "--ollama-model",
        default=DEFAULT_VISION_MODEL,
        help=f"Vision-capable Ollama model tag. Default: {DEFAULT_VISION_MODEL}",
    )
    p.add_argument(
        "--lang",
        default="Greek",
        help="Language hint for notes_el in the prompt (default: Greek).",
    )
    p.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        metavar="PX",
        help="Downscale longest image side to this many pixels before sending (default: 1024).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout seconds for Ollama (default: 180).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return p


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    path = args.input.expanduser().resolve()
    if not path.is_file():
        logger.error("Not a file: %s", path)
        return 1
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.error("Unsupported image type: %s", path.suffix)
        return 1

    out_dir = (args.output_dir or path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    json_path = out_dir / f"{stem}_advice.json"
    md_path = out_dir / f"{stem}_advice.md"
    guide_path = args.guide_image
    if guide_path is None:
        guide_path = out_dir / f"{stem}_advise_guide.png"
    else:
        guide_path = guide_path.expanduser().resolve()

    logger.info(
        "Ollama host: %s model=%s",
        chat_base_url(args.ollama_url),
        args.ollama_model,
    )
    try:
        advice, raw_text = run_advise_pipeline(
            path,
            model=args.ollama_model,
            chat_url=args.ollama_url,
            language=args.lang,
            max_image_side=args.max_image_side,
            timeout=args.timeout,
        )
    except (ConnectionError, RuntimeError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    json_path.write_text(
        json.dumps(advice, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_body = advice_to_markdown(advice)
    md_path.write_text(md_body + "\n", encoding="utf-8")
    raw_path = out_dir / f"{stem}_advice_raw.txt"
    raw_path.write_text(raw_text + "\n", encoding="utf-8")

    try:
        from PIL import Image

        with Image.open(path) as im:
            guide = render_advice_overlay(im, advice)
        guide_path.parent.mkdir(parents=True, exist_ok=True)
        guide.save(guide_path, "PNG", optimize=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write guide image: %s", exc)

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", raw_path)
    logger.info("Wrote %s", guide_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
