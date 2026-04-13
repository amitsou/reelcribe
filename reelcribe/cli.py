"""Command-line interface for the reelcribe package."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from reelcribe.audio import check_ffmpeg, extract_audio, find_video_files
from reelcribe.llm import (
    append_title_line,
    filenames_in_titles_file,
    generate_title,
    save_title,
    verify_ollama_reachable,
)
from reelcribe.transcription import save_transcript, transcribe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelcribe",
        description=(
            "Extract audio, transcribe video with Whisper, and optionally "
            "generate short titles via Ollama. Runs locally."
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
        help="Directory where output files are written (depends on --mode).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["audio", "transcribe", "titles", "full"],
        default="transcribe",
        help=(
            "audio: WAV only; "
            "transcribe: transcript .txt only (no WAV saved); "
            "titles: append lines to titles.txt (filename: title; no WAV or transcript saved); "
            "full: WAV, transcript, and title."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        metavar="SIZE",
        help="Whisper model size (tiny/base/small/medium/large). Default: base.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3",
        metavar="MODEL",
        help="Ollama model for title generation (titles and full modes). Default: llama3.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        metavar="URL",
        help="Ollama HTTP API URL. Default: http://localhost:11434/api/generate.",
    )
    parser.add_argument(
        "--lang",
        default="English",
        metavar="LANG",
        help=(
            "Language for Ollama-generated titles (titles and full modes). "
            "Examples: English, Greek, Spanish. Default: English."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip work when the expected output for that step already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging on stderr.",
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
    """Return True if *path* exists and *skip_existing* is enabled."""
    return skip_existing and path.exists()


def process_file(
    video_path: Path,
    output_dir: Path,
    mode: str,
    whisper_model: str,
    ollama_model: str,
    ollama_url: str,
    skip_existing: bool,
    *,
    title_lang: str = "English",
    titles_txt_path: Path | None = None,
    titles_done: set[str] | None = None,
) -> None:
    """Run the pipeline for one video according to *mode*.

    Parameters
    ----------
    video_path:
        Source video file.
    output_dir:
        Directory for outputs that this mode persists.
    mode:
        One of ``audio``, ``transcribe``, ``titles``, ``full`` (see CLI help).
    whisper_model:
        Whisper model size name passed to ``transcribe``.
    ollama_model:
        Ollama model name for ``generate_title``.
    ollama_url:
        Ollama ``/api/generate`` endpoint URL.
    skip_existing:
        If True, skip a step when its output file already exists.
    title_lang:
        Language hint for Ollama (``titles`` and ``full`` modes).
    titles_txt_path:
        Path to aggregate ``titles.txt`` when ``mode`` is ``titles``.
    titles_done:
        Mutable set of basenames already written to ``titles_txt_path``.
    """
    stem = video_path.stem
    wav_path = output_dir / f"{stem}.wav"
    txt_path = output_dir / f"{stem}.txt"
    title_path = output_dir / f"{stem}_title.txt"

    logger = logging.getLogger(__name__)
    logger.info("Processing: %s", video_path.name)

    if mode == "titles":
        if titles_txt_path is None or titles_done is None:
            raise ValueError("titles mode requires titles_txt_path and titles_done")
        if video_path.name in titles_done:
            logger.info(
                "  Skipping title (already in %s): %s",
                titles_txt_path.name,
                video_path.name,
            )
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_wav = Path(tmp) / f"{stem}.wav"
                extract_audio(video_path, tmp_wav)
                transcript = transcribe(tmp_wav, model_name=whisper_model)
            title = generate_title(
                transcript,
                model=ollama_model,
                ollama_url=ollama_url,
                language=title_lang,
            )
            append_title_line(titles_txt_path, video_path.name, title)
            titles_done.add(video_path.name)
        except Exception as exc:
            logger.error("  Title step failed for %s: %s", video_path.name, exc)
        return

    if mode == "audio":
        if _should_skip(wav_path, skip_existing):
            logger.info("  Skipping audio (already exists): %s", wav_path.name)
            return
        try:
            extract_audio(video_path, wav_path)
        except Exception as exc:
            logger.error("  Audio extraction failed for %s: %s", video_path.name, exc)
        return

    if mode == "transcribe":
        if _should_skip(txt_path, skip_existing):
            logger.info("  Skipping transcript (already exists): %s", txt_path.name)
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_wav = Path(tmp) / f"{stem}.wav"
                extract_audio(video_path, tmp_wav)
                transcript = transcribe(tmp_wav, model_name=whisper_model)
            save_transcript(transcript, txt_path)
        except Exception as exc:
            logger.error("  Transcription failed for %s: %s", video_path.name, exc)
        return

    # mode == "full"
    if _should_skip(wav_path, skip_existing):
        logger.info("  Skipping audio (already exists): %s", wav_path.name)
    else:
        try:
            extract_audio(video_path, wav_path)
        except Exception as exc:
            logger.error("  Audio extraction failed for %s: %s", video_path.name, exc)
            return

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

    if _should_skip(title_path, skip_existing):
        logger.info("  Skipping title (already exists): %s", title_path.name)
        return

    try:
        title = generate_title(
            transcript,
            model=ollama_model,
            ollama_url=ollama_url,
            language=title_lang,
        )
        save_title(title, title_path)
    except Exception as exc:
        logger.error("  Title generation failed for %s: %s", video_path.name, exc)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate paths, and process each discovered video.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` on invalid paths, missing ffmpeg, or
    unreachable Ollama when ``--mode`` is ``titles`` or ``full``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return 1

    try:
        check_ffmpeg()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    if args.mode in ("titles", "full"):
        try:
            verify_ollama_reachable(args.ollama_url)
        except ConnectionError as exc:
            logger.error("%s", exc)
            logger.error(
                "Title output is only written when Ollama accepts requests. "
                "Fix connectivity, then re-run."
            )
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_files = find_video_files(args.input_dir)
    if not video_files:
        logger.warning("No supported video files found in %s", args.input_dir)
        return 0

    total = len(video_files)
    logger.info("Found %d video file(s). Mode: %s", total, args.mode)

    titles_txt_path = args.output_dir / "titles.txt"
    titles_done: set[str] = set()
    if args.mode == "titles":
        if args.skip_existing:
            titles_done = filenames_in_titles_file(titles_txt_path)
        else:
            titles_txt_path.unlink(missing_ok=True)

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
            title_lang=args.lang,
            titles_txt_path=titles_txt_path if args.mode == "titles" else None,
            titles_done=titles_done if args.mode == "titles" else None,
        )

    logger.info("Done. Outputs saved to: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
