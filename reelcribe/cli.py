"""CLI entry point for reelcribe."""

import argparse
import logging
import sys
from pathlib import Path

from reelcribe.audio import check_ffmpeg, extract_audio, find_video_files
from reelcribe.llm import generate_title, save_title
from reelcribe.transcription import save_transcript, transcribe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelcribe",
        description=(
            "Extract audio, transcribe, and optionally generate AI titles "
            "from short video files – fully local."
        ),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing the input video files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory where outputs (WAV / TXT files) will be saved.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["audio", "transcribe", "full"],
        default="transcribe",
        help=(
            "'audio' – extract audio only; "
            "'transcribe' – audio + transcript (default); "
            "'full' – audio + transcript + AI title."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        metavar="SIZE",
        help="Whisper model size to use (tiny/base/small/medium/large). Default: base.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3",
        metavar="MODEL",
        help="Ollama model name for title generation. Default: llama3.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        metavar="URL",
        help="Ollama API endpoint. Default: http://localhost:11434/api/generate.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
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


def _should_skip(path: Path, skip_existing: bool) -> bool:
    """Return True if the output file already exists and skipping is enabled."""
    return skip_existing and path.exists()


def process_file(
    video_path: Path,
    output_dir: Path,
    mode: str,
    whisper_model: str,
    ollama_model: str,
    ollama_url: str,
    skip_existing: bool,
) -> None:
    """Process a single video file according to *mode*."""
    stem = video_path.stem
    wav_path = output_dir / f"{stem}.wav"
    txt_path = output_dir / f"{stem}.txt"
    title_path = output_dir / f"{stem}_title.txt"

    logger = logging.getLogger(__name__)
    logger.info("Processing: %s", video_path.name)

    # --- Audio extraction ---
    if _should_skip(wav_path, skip_existing):
        logger.info("  Skipping audio (already exists): %s", wav_path.name)
    else:
        try:
            extract_audio(video_path, wav_path)
        except Exception as exc:
            logger.error("  Audio extraction failed for %s: %s", video_path.name, exc)
            return

    if mode == "audio":
        return

    # --- Transcription ---
    if _should_skip(txt_path, skip_existing):
        logger.info("  Skipping transcript (already exists): %s", txt_path.name)
        transcript = txt_path.read_text(encoding="utf-8")
    else:
        try:
            transcript = transcribe(wav_path, model_name=whisper_model)
            save_transcript(transcript, txt_path)
        except Exception as exc:
            logger.error("  Transcription failed for %s: %s", video_path.name, exc)
            return

    if mode == "transcribe":
        return

    # --- Title generation ---
    if _should_skip(title_path, skip_existing):
        logger.info("  Skipping title (already exists): %s", title_path.name)
        return

    try:
        title = generate_title(transcript, model=ollama_model, ollama_url=ollama_url)
        save_title(title, title_path)
    except Exception as exc:
        logger.error("  Title generation failed for %s: %s", video_path.name, exc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input directory
    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return 1

    # Check ffmpeg availability early (only needed for audio/transcribe/full)
    try:
        check_ffmpeg()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover video files
    video_files = find_video_files(args.input_dir)
    if not video_files:
        logger.warning("No supported video files found in %s", args.input_dir)
        return 0

    total = len(video_files)
    logger.info("Found %d video file(s). Mode: %s", total, args.mode)

    for idx, video_path in enumerate(video_files, start=1):
        logger.info("[%d/%d] %s", idx, total, video_path.name)
        process_file(
            video_path=video_path,
            output_dir=args.output_dir,
            mode=args.mode,
            whisper_model=args.whisper_model,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            skip_existing=args.skip_existing,
        )

    logger.info("Done. Outputs saved to: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
