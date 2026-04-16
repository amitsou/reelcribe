# reelcribe

![reelcribe logo](./reelcribe.png)

Handover notes for maintainers and users.

## Purpose

**reelcribe** is a small **local-first** toolkit (MIT) built around short-form video and social stills:

| Tool | Role |
|------|------|
| **`reelcribe`** | Batch **video** folder: ffmpeg → **Whisper** transcript → optional **Ollama** title (`/api/generate`, text-only). |
| **`reelcribe-images`** | **Still images** → standard crops (`9x16/`, `4x5/` or `3x4/`) with Pillow. |
| **`reelcribe-reels-thumb`** | **9:16 thumbnail** (1080×1920): face-aware **text zone** hints, JSON, preview PNG, optional Tk UI. Requires extra install **`[reels-thumb]`** (MediaPipe + NumPy). |
| **`reelcribe-thumb-advise`** | **One image** → Ollama **vision** (`/api/chat` + image) → JSON layout advice + markdown + **guide overlay** (Pillow). |

There are **no paid cloud API keys** in the project: Whisper and Ollama run on your machine.

---

## Setup

### 1. Python environment

Requires **Python 3.10+**.

```bash
cd /path/to/reelcribe
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

- **Base install** (`pip install -e .`): `reelcribe`, `reelcribe-images`, `reelcribe-thumb-advise`, Whisper, Pillow.
- **Dev / tests:** `pip install -e ".[dev]"` adds pytest.
- **Reels thumbnail tool:** `pip install -e ".[dev,reels-thumb]"` also installs **MediaPipe** (pinned `<0.10.30` for legacy face detection) and **NumPy** so `reelcribe-reels-thumb` is available.

First Whisper run may download model weights.

```bash
reelcribe -h
reelcribe-images -h
reelcribe-thumb-advise -h
# after [reels-thumb]:
reelcribe-reels-thumb -h
```

### 2. ffmpeg

Needed for **`reelcribe`** (video → audio). Install if `ffmpeg -version` fails.

| Platform | Typical install |
|----------|-----------------|
| macOS (Homebrew) | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt update && sudo apt install ffmpeg` |
| Windows | [ffmpeg.org](https://ffmpeg.org/download.html) on `PATH` |

### 3. Ollama

| Use case | Endpoint | Model type |
|----------|----------|------------|
| **`reelcribe`** modes `titles` / `full` | `POST …/api/generate` (default `http://localhost:11434/api/generate`) | **Text** (e.g. `llama3`) |
| **`reelcribe-thumb-advise`** | `POST …/api/chat` (default `http://localhost:11434/api/chat`) | **Vision** (e.g. `llama3.2-vision`, `llava`) — the model must accept `images` in chat |

**Install and run Ollama** (macOS: [ollama.com/download](https://ollama.com/download) or `brew install ollama`; then `brew services start ollama` or `ollama serve`).

Check the API:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

**Pull models:**

```bash
# titles on videos (text-only is fine)
ollama pull llama3

# thumbnail layout advisor (must see pixels)
ollama pull llama3.2-vision
# alternatives: llava, moondream — pick a tag from `ollama list` that supports images
```

Before video **titles/full**, `reelcribe` probes `GET {origin}/api/tags` so you do not transcribe a whole folder only to find Ollama down.

### 4. Quick verification

| Check | Command / expectation |
|-------|------------------------|
| Package | `python -c "import reelcribe; print(reelcribe.__version__)"` |
| ffmpeg | `ffmpeg -version` |
| Ollama | `curl -s http://127.0.0.1:11434/api/tags` returns JSON |
| Models | `ollama list` includes tags you pass to `--ollama-model` |

---

## 1. Video pipeline (`reelcribe`)

Processes **only direct video files** in one input directory (no subfolders). Supported extensions include `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.flv`, `.m4v`.

```text
reelcribe --input-dir DIR --output-dir DIR [options]
```

Short flags: `-i` / `-o` / `-m` / `-v`.

### Modes (`--mode` / `-m`)

| Mode | Writes | Notes |
|------|--------|--------|
| `audio` | `{stem}.wav` | Mono 16 kHz WAV via ffmpeg. |
| `transcribe` | `{stem}.txt` | Whisper transcript only; temp WAV for decode, **not** kept in output dir. |
| `titles` | `titles.txt` | One UTF-8 line per video: ``<filename>: <title>``. Temp WAV + Whisper; **no** per-video WAV/transcript in output. **Requires Ollama.** |
| `full` | `{stem}.wav`, `{stem}.txt`, `{stem}_title.txt` | Full pipeline. **Requires Ollama** for title. |

Default mode: `transcribe`.

### Common options

- `--whisper-model` (default `base`): `tiny` / `base` / `small` / `medium` / `large`
- `--ollama-model` (default `llama3`): model tag for `titles` and `full`
- `--ollama-url` (default `http://localhost:11434/api/generate`)
- `--lang` (default `English`): title language hint (e.g. `Greek`, `Spanish`)
- `--skip-existing`: skip when expected outputs already exist (`titles`: skip if filename already in `titles.txt`)
- `-v` / `--verbose`: DEBUG on stderr

### Examples

```bash
reelcribe -i ~/Videos/in -o ~/Videos/out -m full --whisper-model small
reelcribe -i ~/Videos/in -o ~/Videos/out -m titles --lang Greek
```

---

## 2. Social still reframing (`reelcribe-images`)

**What it does:** For every supported still in the input set, writes **two** derivative images: one **9:16** (1080×1920) and one **feed portrait** (1080×1350 for 4:5, or 1080×1440 for 3:4). Same **stem + extension** as the source file appears under each subfolder.

**Supported extensions:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` (case-insensitive).

### 2.1 Input shape: file vs folder

| You pass `-i` as… | What is processed | Default output parent (`-o` omitted) |
|-------------------|-------------------|-------------------------------------|
| **Single file** | That file only | The **same directory** as the file |
| **Directory** | Every supported image **directly** in that folder (not subfolders) | The **input directory** itself |

If `-o DIR` is set, both `9x16/` and `4x5/` (or `3x4/`) are created **under** `DIR`.

**On disk layout** (example: `reelcribe-images -i ./art -o ./out`):

```text
out/
  9x16/     hero.png, …   (each 1080×1920)
  4x5/      hero.png, …   (each 1080×1350)   ← or 3x4/ when --feed-aspect 3:4
```

### 2.2 `--fit`: contain vs cover

| `--fit` | Use when… | Behaviour |
|---------|-----------|-----------|
| **`contain`** (default) | You must **not** cut off logos, faces, or edges (typical YouTube-wide art). | Entire source image is visible; empty bands are **letterboxed** to fill the target aspect ratio. |
| **`cover`** | You want the frame **full bleed** and accept **cropping** (center crop). | Fills 1080×… canvas; left/right or top/bottom of the source may be **clipped**. Risky if the subject is off-center. |

### 2.3 `--vertical-align` (only meaningful with `--fit contain`)

Places the **fitted** image block inside the letterboxed canvas:

| Value | Typical use |
|-------|-------------|
| `center` (default) | Neutral; balanced bars top/bottom (or sides). |
| `top` | Ground / product at bottom of frame; more headroom above. |
| `bottom` | Sky / headline space at top; weight toward bottom. |

Ignored in practice for symmetric results when the fitted image already fills one axis; still applies whenever letterbox bars exist on the **vertical** axis.

### 2.4 `--feed-aspect`

| Value | Output folder name | Pixel size |
|-------|-------------------|------------|
| `4:5` (default) | `4x5/` | 1080×1350 |
| `3:4` | `3x4/` | 1080×1440 |

The **9:16** branch is unchanged (always `9x16/` at 1080×1920).

### 2.5 `--verify`

After each write, re-opens the files and checks pixel dimensions. **Exit code 1** if any file is wrong — use in CI or when you distrust the pipeline.

### 2.6 Batch behaviour

Images are processed **in order**. If one file fails to open or throws during reframe, the CLI **logs an error and continues** with the next file (unless `--verify` fails a check, which aborts the whole run).

### 2.7 Example commands (copy-paste cases)

```bash
# One landscape thumbnail → crops next to the file
reelcribe-images -i "./YouTube Thumb.png"

# Folder → explicit output root; Instagram-style 3:4 feed + letterboxed contain
reelcribe-images -i ~/Art/in -o ~/Art/out --feed-aspect 3:4 --fit contain

# Full-bleed Reels cover crops (accept center crop)
reelcribe-images -i ./keyart.jpg -o ./exports --fit cover

# CI / sanity check written dimensions
reelcribe-images -i ./in -o ./out --verify
```

**Design reference:** Full-screen Reels use **9:16**. Tighter grid previews ≈ **3:4** export. **Safe zone** for manual design (not auto-drawn): `SAFE_ZONE_BBOX_9x16` in `reelcribe/social_images.py`.

---

## 3. Reels thumbnail zones (`reelcribe-reels-thumb`)

**Requires:** `pip install -e ".[reels-thumb]"` (MediaPipe + NumPy).

**What it does:** Loads **one** image, optionally normalizes it to **1080×1920** vertical 9:16, runs **face-aware zone analysis**, then either prints **JSON**, writes a **preview PNG**, opens an interactive **UI**, or only prints **logs** — depending on flags.

**Supported extensions:** same as `reelcribe-images` (see §2.1).

### 3.1 Canvas modes (mutually exclusive “geometry strategy”)

Exactly **one** of these paths applies; do not combine `--strict-1080p` with `--no-letterbox`.

| Situation | Flags | Result |
|-----------|-------|--------|
| **Normal / design files** | *(default)* neither flag | If not 1080×1920, image is scaled with **contain + letterbox** to 1080×1920, then analyzed. |
| **Pipeline: asset is already export-sized** | `--strict-1080p` | **Fails fast** if width×height ≠ 1080×1920. No resize. |
| **Debugging only** | `--no-letterbox` | Analyzes at **native** resolution; zones are tuned for 9:16 — **not** recommended for real Meta exports. |

### 3.2 Output modes (can be combined)

You may pass **any subset** of `--json`, `--preview`, `--ui`. All requested outputs run **after** analysis on the same normalized image.

| Flags present | What you get |
|---------------|--------------|
| **None** of `--json` / `--preview` / `--ui` | **Console only:** recommended zone name + **warnings** on stderr (good for a quick check). |
| `--json` | One JSON blob printed to **stdout** (pipe to `jq`, redirect to file). |
| `--preview OUT.png` | PNG written to `OUT.png` (zones, face margin, sample title text). |
| `--ui` | Tk window: cycle zones, tweak preview (needs working **Tkinter**). |
| `--json` **and** `--preview` | Both file **and** stdout JSON in the same run. |

### 3.3 Preview-only options (`--text`, `--zone`)

These affect **only** `--preview` (and the `--ui` flow, which uses the same default sample text until you change it in the window).

| Flag | Role |
|------|------|
| `--text "…"` | Sample headline drawn on the preview (default `Your title`). |
| `--zone NAME` | Prefer this **zone id** for placing sample text. If omitted, the **recommended** zone is used. |

Built-in zone ids (same names appear in `--json` on each `ZoneScore`): `top_right`, `top_left`, `mid_right`, `mid_left`, `bottom_right`, `bottom_left`. If `--zone` does **not** match any zone returned for this image, preview text falls back to the **first** zone in the scored list (you still get a PNG; check `--json` for valid names on that asset).

### 3.4 `--ui` vs `--preview`

| Case | What to run |
|------|-------------|
| Headless server / no display | `--preview file.png` only. |
| pyenv Python **without** Tcl/Tk | CLI exits with error on `--ui`; use `--preview`. |
| Interactive desktop | `--ui` opens a window to cycle zones and **export a PNG** via the built-in save dialog. For a **fixed output path** from a script, use `--preview path.png` (with or without `--ui`). |

### 3.5 Example commands

```bash
# Quick: only logs recommended zone + warnings
reelcribe-reels-thumb -i ./cover.png

# Machine-readable zones for scripts
reelcribe-reels-thumb -i ./cover.png --json > analysis.json

# Visual mock with Greek sample copy in a specific zone
reelcribe-reels-thumb -i ./cover.png --preview ./preview.png --text "Νέο επεισόδιο" --zone top_left

# CI asset: must already be 1080×1920
reelcribe-reels-thumb -i ./export_1080x1920.png --strict-1080p --json

# JSON + frozen preview in one invocation
reelcribe-reels-thumb -i ./cover.png --json --preview ./zones.png
```

---

## 4. Thumbnail layout advisor (`reelcribe-thumb-advise`)

**What it does:** Sends **one** image file to Ollama **`/api/chat`** as base64 (vision), expects **JSON** in the reply, parses it, writes **sidecar** reports, and draws a **guide PNG** with Pillow (markers from JSON — not from the LLM as “painted” pixels).

**Input:** **Single file only** (not a directory). Extensions: same still set as §2.1.

**Prerequisites:** Ollama running; a **vision-capable** model (e.g. `llama3.2-vision`). Plain text models like `llama3` **cannot** see the image.

### 4.1 Output files (always, unless the run fails before write)

Let `{stem}` be the input filename without extension. Outputs go to **`-o` directory** if set, else the **input file’s directory**:

| File | Content |
|------|---------|
| `{stem}_advice.json` | Parsed JSON object (UTF-8, indented). |
| `{stem}_advice.md` | Short markdown summary for humans. |
| `{stem}_advice_raw.txt` | Exact model **text** before parsing (debug / prompt tuning). |
| `{stem}_advise_guide.png` | **Default** path for the guide image (see `--guide-image`). |

The **original image is never modified.**

### 4.2 Flags and when to use them

| Flag | Purpose |
|------|---------|
| `-i` / `--input` | **Required.** Path to the thumbnail. |
| `-o` / `--output-dir` | Put **all** outputs here (directory is created if needed). |
| `--guide-image PATH` | Write the guide PNG to a **custom** path instead of `{stem}_advise_guide.png` under the output dir. |
| `--ollama-url` | Default `http://localhost:11434/api/chat`. Change host/port if Ollama runs elsewhere. |
| `--ollama-model` | Vision model tag (default `llama3.2-vision`). Must match `ollama list`. |
| `--lang` | Hint for **language of bullet notes** in the JSON (`notes_el`); default **Greek** in the CLI. |
| `--max-image-side` | Longest side in pixels before base64 (default **1024**): faster uploads, slightly less detail for the model. |
| `--timeout` | HTTP timeout seconds (default **180**). |
| `-v` / `--verbose` | DEBUG logging. |

### 4.3 Typical flows

| Goal | Command sketch |
|------|----------------|
| Default local run | `reelcribe-thumb-advise -i ./draft.png` |
| English bullet notes | `reelcribe-thumb-advise -i ./draft.png --lang English` |
| Collect reviews in one folder | `reelcribe-thumb-advise -i ./draft.png -o ~/Reviews/exports/` |
| Fixed guide filename for upload | `reelcribe-thumb-advise -i ./draft.png --guide-image ./out/guide_draft.png` |
| Remote Ollama | `reelcribe-thumb-advise -i ./d.png --ollama-url http://192.168.1.10:11434/api/chat` |
| Large 4K source, faster request | `reelcribe-thumb-advise -i ./big.png --max-image-side 768` |

### 4.4 Failure behaviour (what to expect)

| Problem | Symptom / outcome |
|---------|-------------------|
| Ollama down or wrong URL | **ConnectionError**; non-zero exit; no new files. |
| Model returns non-JSON | **RuntimeError** or **ValueError** from parser; no JSON file or partial writes depending on failure point (pipeline fails before writes). |
| Vision model missing / wrong tag | HTTP or model error from Ollama; check `ollama list` and pull a vision tag. |
| Guide PNG cannot be written (rare) | JSON/MD/raw may still be written; CLI **warns** and continues (exit 0 if advice step succeeded). |

---

## End-to-end creative chain (optional)

Example order (all local):

1. **`reelcribe-images`** — from landscape key art to `9x16/1080×1920` still.  
2. **`reelcribe-reels-thumb`** — zone JSON + preview for text placement (needs `[reels-thumb]`).  
3. Design in your editor, **export draft** PNG.  
4. **`reelcribe-thumb-advise`** — marketing/layout critique + guide overlay (needs vision Ollama).

---

## Repository layout

| Path | Role |
|------|------|
| `reelcribe/cli.py` | Video CLI |
| `reelcribe/audio.py` | ffmpeg, WAV extract, video discovery |
| `reelcribe/transcription.py` | Whisper, transcript I/O |
| `reelcribe/llm.py` | Ollama `/api/generate`, titles file helpers |
| `reelcribe/social_images.py` | Reframe, safe zone constants, image I/O |
| `reelcribe/images_cli.py` | `reelcribe-images` |
| `reelcribe/reels_text_zones.py` | Face/text zone analysis (MediaPipe) |
| `reelcribe/reels_thumb_cli.py` | `reelcribe-reels-thumb` |
| `reelcribe/thumb_advise.py` | Ollama `/api/chat` + JSON + overlay |
| `reelcribe/thumb_advise_cli.py` | `reelcribe-thumb-advise` |
| `tests/` | Pytest (mocked subprocess / HTTP / optional heavy deps) |

Console scripts are declared in `pyproject.toml` under `[project.scripts]`.

---

## Testing

```bash
pytest
```

Most tests mock ffmpeg, Whisper, and HTTP and **do not** require Ollama or GPU. Some suites may import optional native stacks (e.g. MediaPipe); install `.[reels-thumb]` if a test is skipped or fails on import.

---

## Troubleshooting

- **`Could not reach Ollama` (video titles):** Start Ollama, check `/api/tags`, `ollama pull` the same tag as `--ollama-model`.
- **`reelcribe-thumb-advise` errors / empty vision:** Use a **vision** model on **`/api/chat`**; plain `llama3` does not see images.
- **`reelcribe-reels-thumb` import errors:** `pip install -e ".[reels-thumb]"`.
- **Tkinter missing (`--ui`):** Use `--preview OUT.png` or install Python with Tcl/Tk (see CLI error hints on macOS/pyenv).
- **Port 11434 in use:** One Ollama instance only.
- **Whisper slow / large download:** Smaller `--whisper-model` (e.g. `tiny`, `base`).
- **Full `pytest` crash (segfault):** Often a native extension (MediaPipe / torch); run a subset, e.g. `pytest tests/test_llm.py tests/test_thumb_advise.py`, or update OS / isolate the failing module.

Video processing is **sequential**; batch time and RAM depend on Whisper model size.

---

## License

See `LICENSE` in the repository root.
