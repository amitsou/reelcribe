"""CLI: reframe stills (e.g. YouTube thumbnails) into common social aspect ratios."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image as PILImage

from reelcribe.social_images import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VERTICAL_9x16,
    find_image_files,
    instagram_feed_size,
    load_image,
    reframe,
    save_image,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelcribe-images",
        description=(
            "Repurpose a thumbnail or a folder of stills to standard social sizes. "
            "Default --fit contain keeps the whole frame (letterboxing); use --fit cover only "
            "if you want center-crop (bad for wide art with subjects on the edges). "
            "Outputs: 9x16/, 4x5/ or 3x4/."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        dest="input_path",
        help="Single image file or a directory of images (directory is non-recursive).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Parent folder for 9x16/, 4x5/ (or 3x4/). "
            "If omitted, uses the same directory as the input file, or the input directory."
        ),
    )
    parser.add_argument(
        "--feed-aspect",
        choices=["4:5", "3:4"],
        default="4:5",
        help="Portrait feed crop (default 4:5 → 1080x1350). 3:4 → 1080x1440.",
    )
    parser.add_argument(
        "--fit",
        choices=["contain", "cover"],
        default="contain",
        help=(
            "contain: fit entire image in the frame (letterbox, no clipping; default). "
            "cover: center-crop (fills frame; cuts off sides or top/bottom)."
        ),
    )
    parser.add_argument(
        "--vertical-align",
        choices=["top", "center", "bottom"],
        default="center",
        help="With --fit contain, where to place the image vertically in the canvas.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After each write, re-open the file and check pixel dimensions match "
            "1080×1920 (9x16) and the feed size (4:5 or 3:4). Exits with error if not."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG logging on stderr.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_path = args.input_path.expanduser().resolve()

    if not input_path.exists():
        logger.error("Input does not exist: %s", input_path)
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.error("Not a supported image file: %s", input_path)
            return 1
        images = [input_path]
        default_out = input_path.parent
    elif input_path.is_dir():
        images = find_image_files(input_path)
        default_out = input_path
    else:
        logger.error("Input must be a file or directory: %s", input_path)
        return 1

    output_base = args.output_dir.expanduser().resolve() if args.output_dir else default_out

    feed_size = instagram_feed_size(args.feed_aspect)
    feed_dir_name = "4x5" if args.feed_aspect == "4:5" else "3x4"
    out_vertical = output_base / "9x16"
    out_feed = output_base / feed_dir_name

    if not images:
        logger.warning("No supported images in %s", input_path)
        return 0

    total = len(images)
    logger.info(
        "Found %d image(s). fit=%s. Writing 9:16 (%dx%d) and %s (%dx%d).",
        total,
        args.fit,
        VERTICAL_9x16[0],
        VERTICAL_9x16[1],
        args.feed_aspect,
        feed_size[0],
        feed_size[1],
    )
    if args.verify:
        logger.info("--verify: will check written files match exact canvas sizes.")

    for idx, src in enumerate(images, start=1):
        logger.info("[%d/%d] %s", idx, total, src.name)
        try:
            im = load_image(src)
        except OSError as exc:
            logger.error("  Could not open %s: %s", src.name, exc)
            continue

        stem = src.stem
        suffix = src.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            suffix = ".png"

        try:
            vertical = reframe(
                im,
                *VERTICAL_9x16,
                fit=args.fit,
                vertical_align=args.vertical_align,
            )
            save_image(vertical, out_vertical / f"{stem}{suffix}")

            feed_img = reframe(
                im,
                *feed_size,
                fit=args.fit,
                vertical_align=args.vertical_align,
            )
            save_image(feed_img, out_feed / f"{stem}{suffix}")

            if args.verify:
                v_path = out_vertical / f"{stem}{suffix}"
                f_path = out_feed / f"{stem}{suffix}"
                with PILImage.open(v_path) as v_chk:
                    if v_chk.size != VERTICAL_9x16:
                        logger.error(
                            "  VERIFY failed %s: expected %dx%d, got %s",
                            v_path.name,
                            VERTICAL_9x16[0],
                            VERTICAL_9x16[1],
                            v_chk.size,
                        )
                        return 1
                with PILImage.open(f_path) as f_chk:
                    if f_chk.size != feed_size:
                        logger.error(
                            "  VERIFY failed %s: expected %dx%d, got %s",
                            f_path.name,
                            feed_size[0],
                            feed_size[1],
                            f_chk.size,
                        )
                        return 1
        except Exception as exc:
            logger.error("  Failed for %s: %s", src.name, exc)
            continue

        logger.debug(
            "  Wrote crops under %s/9x16 and %s/%s",
            output_base,
            output_base,
            feed_dir_name,
        )

    logger.info("Done. Outputs under: %s", output_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
