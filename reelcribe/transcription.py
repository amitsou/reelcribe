"""Transcription module – converts WAV audio to text using OpenAI Whisper."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe(audio_path: Path, model_name: str = "base") -> str:
    """Transcribe *audio_path* using the Whisper *model_name* and return the text.

    Parameters
    ----------
    audio_path:
        Path to the WAV (or any audio) file to transcribe.
    model_name:
        Whisper model size to use, e.g. ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, or ``"large"``.  Defaults to ``"base"``.

    Returns
    -------
    str
        The full transcript text.

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    ImportError
        If the ``openai-whisper`` package is not installed.
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'openai-whisper' package is required for transcription.\n"
            "Install it with:  pip install openai-whisper"
        ) from exc

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model '%s' …", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing %s …", audio_path)
    result = model.transcribe(str(audio_path))

    text: str = result["text"].strip()
    logger.info("Transcription complete (%d chars)", len(text))
    return text


def save_transcript(text: str, output_path: Path) -> Path:
    """Write *text* to *output_path* (UTF-8).

    Parameters
    ----------
    text:
        Transcript content.
    output_path:
        Destination ``.txt`` file path.

    Returns
    -------
    Path
        The path to the written transcript file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Transcript saved: %s", output_path)
    return output_path
