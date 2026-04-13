"""LLM module – generates a short title from a transcript via the Ollama API."""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

_PROMPT_TEMPLATE = (
    "You are a helpful assistant that writes short, catchy titles for short videos.\n"
    "Given the following transcript, reply with ONLY the title – no explanation, "
    "no quotes, no punctuation at the end.\n\n"
    "Transcript:\n{transcript}\n\nTitle:"
)


def generate_title(
    transcript: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Call the Ollama API to generate a short title for *transcript*.

    Parameters
    ----------
    transcript:
        The full transcript text.
    model:
        Ollama model name (default: ``"llama3"``).
    ollama_url:
        Base URL of the Ollama ``/api/generate`` endpoint.

    Returns
    -------
    str
        A short title string.

    Raises
    ------
    ConnectionError
        If the Ollama server cannot be reached.
    RuntimeError
        If the API returns an unexpected response.
    """
    prompt = _PROMPT_TEMPLATE.format(transcript=transcript)
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode()

    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Requesting title from Ollama model '%s' …", model)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at '{ollama_url}'. "
            "Make sure Ollama is running (`ollama serve`)."
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
    """Write *title* to *output_path* (UTF-8).

    Parameters
    ----------
    title:
        The generated title string.
    output_path:
        Destination ``.txt`` file path.

    Returns
    -------
    Path
        The path to the written title file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(title, encoding="utf-8")
    logger.info("Title saved: %s", output_path)
    return output_path
