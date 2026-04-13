# reelcribe

![reelcribe logo](./reelcribe.png)

Handover notes for maintainers and users.

## Purpose

`reelcribe` is a small Python CLI that processes short video files in a single input directory. It can:

1. Extract mono 16 kHz WAV audio with **ffmpeg**
2. Transcribe speech with **OpenAI Whisper** (local, via the `openai-whisper` package)
3. Generate a short title per video using **Ollama** over HTTP (local inference; no bundled cloud APIs)

Outputs are written under a user-chosen output directory. Behavior depends on `--mode` (see below).

## Setup

Follow these in order. You need **Python + ffmpeg** for any mode that reads video. You need **Ollama with at least one model pulled** for `--mode titles` or `--mode full`.

### 1. Python environment and reelcribe

Requires **Python 3.10+**.

```bash
cd /path/to/reelcribe
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs `reelcribe` into the venv (including `openai-whisper`, which pulls in PyTorch and other dependencies). The first time you transcribe, Whisper may download the selected model weights.

Confirm the CLI:

```bash
reelcribe -h
```

### 2. ffmpeg

`reelcribe` calls the `ffmpeg` binary on your `PATH`. Install it if `ffmpeg -version` fails.

| Platform | Typical install |
|----------|-----------------|
| macOS (Homebrew) | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt update && sudo apt install ffmpeg` |
| Windows | Install from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to `PATH` |

### 3. Ollama (title modes only)

Skip this block if you only use `--mode audio` or `--mode transcribe`.

1. **Install Ollama**  
   - macOS: [ollama.com/download](https://ollama.com/download) or `brew install ollama`

2. **Run the service**  
   - macOS (Homebrew): `brew services start ollama`  
   - Or run `ollama serve` in a terminal and leave it open.

3. **Check the API responds**

   ```bash
   curl -s http://127.0.0.1:11434/api/tags
   ```

   You should get JSON. If the connection fails, fix Ollama before running reelcribe title modes.

4. **Pull a model** (required)

   An empty list (`"models":[]`) means the server is up but **no models are installed**. Title generation will fail until you pull at least one model:

   ```bash
   ollama pull llama3
   ```

   Use the **same name** you pass to reelcribe (default is `llama3`):

   ```bash
   ollama list
   reelcribe ... --ollama-model llama3
   ```

   For a smaller download, you might use e.g. `ollama pull llama3.2:3b` and then `--ollama-model llama3.2:3b`.

`reelcribe` uses `--ollama-url` (default `http://localhost:11434/api/generate`). Before processing videos in `titles` or `full` mode, it checks `GET http://<host>:<port>/api/tags` so you do not transcribe everything only to find Ollama unreachable.

### 4. Quick verification

| Check | Command / expectation |
|-------|------------------------|
| Python package | `python -c "import reelcribe; print(reelcribe.__version__)"` |
| ffmpeg | `ffmpeg -version` |
| Ollama up | `curl -s http://127.0.0.1:11434/api/tags` returns JSON |
| Model available | `ollama list` shows the model you use with `--ollama-model` |

## Repository layout

| Path | Role |
|------|------|
| `reelcribe/cli.py` | Argument parsing, orchestration, per-video pipeline |
| `reelcribe/audio.py` | ffmpeg discovery, WAV extraction, video file discovery |
| `reelcribe/transcription.py` | Whisper load/transcribe, transcript file I/O |
| `reelcribe/llm.py` | Ollama `/api/generate` client, title file I/O |
| `tests/` | Pytest unit tests (mocked ffmpeg, Whisper, HTTP) |

Entry point: `reelcribe = reelcribe.cli:main` in `pyproject.toml`.

## Usage

```text
reelcribe --input-dir DIR --output-dir DIR [options]
```

Short flags: `-i` / `-o` / `-m` / `-v`.

### Modes (`--mode` / `-m`)

| Mode | Writes | Notes |
|------|--------|--------|
| `audio` | `{stem}.wav` | Extract audio only. |
| `transcribe` | `{stem}.txt` | Transcript only. Audio is decoded to a **temporary** WAV (not kept in the output dir). |
| `titles` | `titles.txt` | One line per video: ``<filename>: <title>`` (UTF-8). Appends as each file completes. Temp WAV for decode + Whisper; no per-video WAV or transcript in the output dir. Requires Ollama. |
| `full` | `{stem}.wav`, `{stem}.txt`, `{stem}_title.txt` | Full pipeline. Requires Ollama for the title step. |

Default mode: `transcribe`.

### Common options

- `--whisper-model` (default `base`): Whisper size (`tiny` / `base` / `small` / `medium` / `large`)
- `--ollama-model` (default `llama3`): Ollama model tag for `titles` and `full`
- `--ollama-url`: Ollama generate endpoint (default `http://localhost:11434/api/generate`)
- `--lang` (default `English`): Language for Ollama-generated titles in `titles` and `full` modes (for example `Greek`, `Spanish`, `de`). The prompt asks the model to write in that language; quality depends on the model.
- `--skip-existing`: Skip work if outputs already exist (for `titles`, skips videos whose filename already appears as the first column in `titles.txt`)
- `--verbose` / `-v`: DEBUG logging on stderr

### Video inputs

Only **direct files** in `--input-dir` are considered (no subfolders). Supported extensions include `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.flv`, `.m4v`.

### Examples

```bash
reelcribe -i ~/Videos/in -o ~/Videos/out -m full --whisper-model small
```

Greek titles in a single `titles.txt` (aggregate file in the output directory):

```bash
reelcribe -i ~/Videos/in -o ~/Videos/out -m titles --lang Greek
```

## Testing

```bash
pytest
```

Tests mock external tools (ffmpeg subprocess, Whisper import, HTTP). They do not require GPU or Ollama at test time.

## Troubleshooting

- **`Could not reach Ollama` / no titles:** Start Ollama, confirm `curl` to `/api/tags` works, and run `ollama pull <model>` so `ollama list` matches `--ollama-model`. In `titles` mode, lines go to `titles.txt`; in `full` mode, each title is still `{stem}_title.txt`.
- **Port 11434 already in use:** Another Ollama instance is running; do not start a second `ollama serve`, or stop the service and use one launch method only.
- **Whisper slow or large download:** First run downloads weights; use a smaller `--whisper-model` (e.g. `tiny` or `base`) for speed.

For large batches, disk and RAM depend on Whisper model size; this CLI processes videos **sequentially**.

## License

See `LICENSE` in the repository root.
