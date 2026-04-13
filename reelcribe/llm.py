"""Title generation by posting transcripts to a local Ollama HTTP API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def verify_ollama_reachable(ollama_url: str, timeout: float = 5.0) -> str:
    """Check that an Ollama server responds before running a long batch.

    Uses ``GET {origin}/api/tags`` (same host as *ollama_url*).

    Parameters
    ----------
    ollama_url:
        Full URL to ``/api/generate`` or any path on the Ollama host.
    timeout:
        Seconds to wait for the HTTP response.

    Returns
    -------
    str
        Base URL ``scheme://netloc`` that was probed.

    Raises
    ------
    ConnectionError
        If the URL is invalid or the server is not reachable.
    """
    parsed = urlparse(ollama_url)
    if not parsed.scheme or not parsed.netloc:
        raise ConnectionError(
            f"Invalid Ollama URL {ollama_url!r}. "
            "Use a full URL, e.g. http://localhost:11434/api/generate"
        )
    base = f"{parsed.scheme}://{parsed.netloc}"
    tags_url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            resp.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at {base}. "
            "Start the Ollama service, then retry (e.g. brew services start ollama)."
        ) from exc
    logger.debug("Ollama responded at %s", base)
    return base


TITLE_LINE_SEPARATOR = ": "


def build_title_prompt(transcript: str, language: str) -> str:
    """Build the Ollama prompt for a single short-form video title."""
    lang = language.strip()
    lang_block = (
        f"Write the title in {lang}.\n\n"
        if lang
        else ""
    )
    return (
        "You are a senior marketing and journalism professional with over twenty years of "
        "experience writing headlines and titles for digital media. Your titles are catchy, "
        "clear, and accurate to the source material.\n\n"
        "Based only on the transcript below, propose one short title for this short-form video. "
        "It should grab attention without being misleading.\n"
        f"{lang_block}"
        "Reply with the title text alone: no role labels, no quotes, no trailing punctuation.\n\n"
        f"Transcript:\n{transcript}\n\nTitle:"
    )


def generate_title(
    transcript: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    *,
    language: str = "English",
) -> str:
    """Request a single-line title for *transcript* from Ollama.

    Parameters
    ----------
    transcript:
        Full transcript used as context.
    model:
        Ollama model tag (for example ``llama3``).
    ollama_url:
        Full URL to the ``/api/generate`` endpoint.
    language:
        Natural language for the title (for example ``English``, ``Greek``, ``es``).
        Passed into the prompt; multilingual models such as Llama 3 follow this.

    Returns
    -------
    str
        Non-empty title string (leading and trailing whitespace removed).

    Raises
    ------
    ConnectionError
        If the HTTP request fails (for example Ollama is not running).
    RuntimeError
        If the response body is not valid JSON or lacks a ``response`` field.
    """
    prompt = build_title_prompt(transcript, language)
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode()

    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Requesting title from Ollama model '%s' ...", model)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at '{ollama_url}'. "
            "Ensure the Ollama service is running."
        ) from exc

    try:
        data = json.loads(body)
        title: str = data["response"].strip()
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"Unexpected response from Ollama API: {body[:200]}"
        ) from exc

    logger.info("Generated title: %s", title)
    return title


def save_title(title: str, output_path: Path) -> Path:
    """Write *title* to *output_path* as UTF-8 text.

    Parameters
    ----------
    title:
        Single-line or short title string.
    output_path:
        Destination path (parents are created if missing).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(title, encoding="utf-8")
    logger.info("Title saved: %s", output_path)
    return output_path


def append_title_line(
    output_path: Path,
    source_filename: str,
    title: str,
) -> None:
    """Append ``<source_filename>: <title>`` to *output_path* (UTF-8, one line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    one_line = title.replace("\n", " ").replace("\r", "").strip()
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"{source_filename}{TITLE_LINE_SEPARATOR}{one_line}\n")
    logger.info("Title line appended: %s", output_path)


def filenames_in_titles_file(path: Path) -> set[str]:
    """Return source filenames already present in a ``titles.txt``-style file."""
    if not path.is_file():
        return set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        idx = line.find(TITLE_LINE_SEPARATOR)
        if idx >= 0:
            names.add(line[:idx])
    return names
