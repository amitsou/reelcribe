"""CLI: face-aware text zone suggestions for 1080×1920 Reels / Meta Suite thumbnails."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from reelcribe.reels_text_zones import (
    analyze_thumbnail,
    analysis_to_json,
    clamp_point_to_zones,
    ensure_1080x1920,
    paste_logo_centered,
    plain_text_from_markup,
    render_preview,
    render_text_layer_rgba,
    zone_for_point,
)
from reelcribe.social_images import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VERTICAL_9x16,
    dimensions_match_ratio,
    load_image,
)

logger = logging.getLogger(__name__)


def _tkinter_available() -> bool:
    """True if this interpreter can load Tk (many pyenv builds omit ``_tkinter``)."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _log_tkinter_unavailable() -> None:
    logger.error(
        "Tkinter is not available: this Python was built without Tcl/Tk (`_tkinter`). "
        "Common with pyenv unless you link Homebrew tcl-tk when installing Python."
    )
    logger.error(
        "Workaround: use --preview out.png (open the PNG in Preview) instead of --ui."
    )
    logger.error(
        "Fix (macOS + Homebrew): brew install tcl-tk, then reinstall Python with pyenv "
        "using PYTHON_CONFIGURE_OPTS that point to tcl-tk — see pyenv wiki \"Installing "
        "Python with tkinter support\"."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reelcribe-reels-thumb",
        description=(
            "Suggest where to place short on-image text on a Meta Reels cover. The working "
            f"canvas is always {VERTICAL_9x16[0]}×{VERTICAL_9x16[1]} (9:16). If the file is not "
            "already that size, it is normalized with contain + letterboxing unless you pass "
            "--no-letterbox."
        ),
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Input image (PNG/JPEG/WebP…).",
    )
    p.add_argument(
        "--strict-1080p",
        action="store_true",
        help=(
            f"Fail if the file is not already exactly {VERTICAL_9x16[0]}×{VERTICAL_9x16[1]} "
            "(no conversion). Use in pipelines that guarantee pre-sized assets."
        ),
    )
    p.add_argument(
        "--no-letterbox",
        action="store_true",
        help=(
            "Skip conversion to 1080×1920 (not recommended). Zone logic and export assume a "
            "9:16 canvas; use only for debugging."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print analysis JSON to stdout.",
    )
    p.add_argument(
        "--preview",
        type=Path,
        metavar="OUT.png",
        help="Write a preview image with zones, face margin, and sample text.",
    )
    p.add_argument(
        "--text",
        default="Your title",
        help="Sample text drawn on the preview (default: %(default)s).",
    )
    p.add_argument(
        "--zone",
        metavar="NAME",
        default=None,
        help="Zone for preview text (e.g. top_right). Default: recommended zone.",
    )
    p.add_argument(
        "--ui",
        action="store_true",
        help=(
            "Open a small Tk window to cycle zones and export a preview. "
            "Requires Tkinter (not present in some pyenv builds; use --preview otherwise)."
        ),
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

    if args.ui and not _tkinter_available():
        _log_tkinter_unavailable()
        return 1

    try:
        im = load_image(path)
    except OSError as exc:
        logger.error("Could not open image: %s", exc)
        return 1

    w0, h0 = im.size
    if args.strict_1080p:
        if im.size != VERTICAL_9x16:
            logger.error(
                "Strict check failed: need exactly %dx%d (vertical 9:16), got %dx%d. "
                "Resize your asset or omit --strict-1080p to auto-convert.",
                VERTICAL_9x16[0],
                VERTICAL_9x16[1],
                w0,
                h0,
            )
            return 1
        if not dimensions_match_ratio(w0, h0, 9, 16):
            logger.error("Internal check: dimensions are not 9:16.")
            return 1
        logger.info(
            "Input validated: %dx%d (Reels cover / Meta Suite thumbnail size).",
            w0,
            h0,
        )
    elif args.no_letterbox:
        logger.warning(
            "Skipping 1080×1920 normalization — analyzing at %dx%d. "
            "Exports and zones are designed for %dx%d.",
            w0,
            h0,
            VERTICAL_9x16[0],
            VERTICAL_9x16[1],
        )
    else:
        if im.size == VERTICAL_9x16:
            logger.info(
                "Input is already %dx%d (9:16). No conversion needed.",
                VERTICAL_9x16[0],
                VERTICAL_9x16[1],
            )
        else:
            logger.warning(
                "Input is %dx%d, not %dx%d. Normalizing to vertical 9:16 (contain + letterbox).",
                w0,
                h0,
                VERTICAL_9x16[0],
                VERTICAL_9x16[1],
            )
        im, padded = ensure_1080x1920(im)
        if im.size != VERTICAL_9x16:
            logger.error("Normalization failed: expected %dx%d.", VERTICAL_9x16[0], VERTICAL_9x16[1])
            return 1
        if not dimensions_match_ratio(im.size[0], im.size[1], 9, 16):
            logger.error("Normalized image is not 9:16.")
            return 1
        if padded:
            logger.info(
                "Converted to %dx%d. All further steps use this canvas.",
                VERTICAL_9x16[0],
                VERTICAL_9x16[1],
            )
        else:
            logger.debug("Canvas size confirmed: %dx%d.", im.size[0], im.size[1])

    result = analyze_thumbnail(im)

    if args.json:
        print(analysis_to_json(result))

    if args.preview:
        preview = render_preview(
            im,
            result,
            text=args.text,
            zone_name=args.zone,
        )
        out = args.preview.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        preview.save(out, "PNG", optimize=True)
        logger.info("Wrote preview: %s", out)

    if args.ui:
        _run_ui(im, result, path, sample_text=args.text)

    if not args.json and not args.preview and not args.ui:
        z = result.recommended_zone or "?"
        logger.info("Recommended zone: %s", z)
        for w in result.warnings:
            logger.warning("%s", w)

    return 0


def _unique_label(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base} ({n})" in taken:
        n += 1
    return f"{base} ({n})"


def _prioritized_font_candidates() -> list[tuple[str, str]]:
    """Hand-picked fonts shown first (poster / UI / Greek-friendly when installed)."""
    home = Path.home()
    win = Path(os.environ.get("SystemRoot", "C:\\")) / "Windows" / "Fonts"
    return [
        ("Bebas Neue", str(home / "Library/Fonts/BebasNeue-Regular.ttf")),
        ("Bebas Neue", "/Library/Fonts/BebasNeue-Regular.ttf"),
        ("Anton", str(home / "Library/Fonts/Anton-Regular.ttf")),
        ("Anton", "/Library/Fonts/Anton-Regular.ttf"),
        ("Oswald Bold", str(home / "Library/Fonts/Oswald-Bold.ttf")),
        ("Archivo Narrow Bold", str(home / "Library/Fonts/ArchivoNarrow-Bold.ttf")),
        ("Barlow Condensed Bold", str(home / "Library/Fonts/BarlowCondensed-Bold.ttf")),
        ("Montserrat Bold", str(home / "Library/Fonts/Montserrat-Bold.ttf")),
        ("Poppins Bold", str(home / "Library/Fonts/Poppins-Bold.ttf")),
        ("Roboto Bold", str(home / "Library/Fonts/Roboto-Bold.ttf")),
        ("Open Sans Bold", str(home / "Library/Fonts/OpenSans-Bold.ttf")),
        ("Lato Bold", str(home / "Library/Fonts/Lato-Bold.ttf")),
        ("Inter", "/System/Library/Fonts/Supplemental/Inter.ttf"),
        ("Inter Bold", "/System/Library/Fonts/Supplemental/Inter-Bold.ttf"),
        ("SF Pro Display Bold", "/System/Library/Fonts/Supplemental/SF-Pro-Display-Bold.otf"),
        ("SF Pro Text Bold", "/System/Library/Fonts/Supplemental/SF-Pro-Text-Bold.otf"),
        ("Helvetica Neue Bold", "/System/Library/Fonts/Supplemental/HelveticaNeue-Bold.ttf"),
        ("Helvetica Neue", "/System/Library/Fonts/HelveticaNeue.ttc"),
        ("Arial Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"),
        ("Arial Italic", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ("Arial Narrow Bold", "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"),
        ("Georgia Bold", "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
        ("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"),
        ("Verdana Bold", "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
        ("Verdana", "/System/Library/Fonts/Supplemental/Verdana.ttf"),
        ("Helvetica", "/System/Library/Fonts/Supplemental/Helvetica.ttf"),
        ("Helvetica Bold", "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"),
        ("Impact", "/System/Library/Fonts/Supplemental/Impact.ttf"),
        ("Trebuchet Bold", "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf"),
        ("Times Bold", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        ("Times New Roman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        ("Futura Bold", "/System/Library/Fonts/Supplemental/Futura Bold.ttf"),
        ("Gill Sans Bold", "/System/Library/Fonts/Supplemental/GillSans-Bold.ttf"),
        ("Noto Sans Bold", "/System/Library/Fonts/Supplemental/NotoSans-Bold.ttf"),
        ("Noto Serif Bold", "/System/Library/Fonts/Supplemental/NotoSerif-Bold.ttf"),
        ("Palatino Bold", "/System/Library/Fonts/Supplemental/Palatino Bold.ttf"),
        ("Copperplate Bold", "/System/Library/Fonts/Supplemental/Copperplate Bold.ttf"),
        ("Chalkboard Bold", "/System/Library/Fonts/Supplemental/Chalkboard Bold.ttf"),
        ("American Typewriter Bold", "/System/Library/Fonts/Supplemental/AmericanTypewriter-Bold.ttf"),
        ("Courier New Bold", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"),
        ("Menlo Bold", "/System/Library/Fonts/Supplemental/Menlo-Bold.ttf"),
        ("DejaVu Sans Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("DejaVu Sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("Liberation Sans Bold", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ("Noto Sans", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        ("Segoe UI Bold", str(win / "segoeuib.ttf")),
        ("Segoe UI", str(win / "segoeui.ttf")),
        ("Calibri Bold", str(win / "calibrib.ttf")),
        ("Calibri", str(win / "calibri.ttf")),
        ("Consolas Bold", str(win / "consolab.ttf")),
        ("Arial Bold", str(win / "arialbd.ttf")),
        ("Arial", str(win / "arial.ttf")),
        ("Tahoma Bold", str(win / "tahomabd.ttf")),
        ("Verdana", str(win / "verdana.ttf")),
    ]


def _glob_font_entries(max_files: int = 500) -> list[tuple[str, str]]:
    """Extra fonts from standard folders (non-recursive per dir; cap total)."""
    dirs: list[Path] = []
    if os.name == "nt":
        w = Path(os.environ.get("SystemRoot", "C:\\")) / "Windows" / "Fonts"
        if w.is_dir():
            dirs.append(w)
    dirs.extend(
        [
            Path.home() / "Library/Fonts",
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
        ]
    )
    if sys.platform.startswith("linux"):
        for p in (
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/liberation"),
            Path("/usr/share/fonts/truetype/noto"),
        ):
            dirs.append(p)

    entries: list[tuple[str, str]] = []
    exts = (".ttf", ".otf", ".ttc")
    for d in dirs:
        if not d.is_dir() or len(entries) >= max_files:
            break
        try:
            for path in sorted(d.iterdir()):
                if len(entries) >= max_files:
                    break
                if not path.is_file() or path.suffix.lower() not in exts:
                    continue
                label = path.stem.replace("-", " ").replace("_", " ")
                entries.append((label, str(path)))
        except OSError:
            continue
    return entries


def _discover_font_presets() -> list[tuple[str, str]]:
    """(label, path) pairs: prioritized picks first, then fonts discovered on disk."""
    out: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    seen_labels: set[str] = set()

    for label, path in _prioritized_font_candidates():
        if not os.path.isfile(path) or path in seen_paths:
            continue
        label = _unique_label(label, seen_labels)
        seen_paths.add(path)
        seen_labels.add(label)
        out.append((label, path))

    for label, path in _glob_font_entries():
        if path in seen_paths:
            continue
        if not os.path.isfile(path):
            continue
        label = _unique_label(label, seen_labels)
        seen_paths.add(path)
        seen_labels.add(label)
        out.append((label, path))

    return out


# Readability presets for short titles on mixed backgrounds (outline keeps contrast).
_OPTIMA_COLOR_PRESETS: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = [
    ("Λευκό + μαύρο outline", (250, 250, 250), (0, 0, 0)),
    ("Charcoal + λευκό outline", (28, 28, 32), (255, 255, 255)),
    ("Deep blue + λευκό outline", (28, 52, 92), (255, 255, 255)),
    ("Cream + σκούρο outline", (255, 248, 240), (20, 20, 24)),
    ("Χρυσαφί + μαύρο outline", (255, 214, 120), (0, 0, 0)),
]


def _reels_thumb_ui_font() -> str:
    import tkinter.font as tkfont

    fams = set(tkfont.families())
    for name in ("SF Pro Text", "Segoe UI", "Helvetica Neue", "Helvetica"):
        if name in fams:
            return name
    return "Helvetica"


def _apply_reels_thumb_theme(root: tk.Tk) -> tuple[ttk.Style, dict[str, str], str]:
    """Dark, editor-style chrome for the thumbnail editor window."""
    import tkinter as tk
    from tkinter import ttk

    ui_font = _reels_thumb_ui_font()
    C: dict[str, str] = {
        "bg": "#121218",
        "panel": "#1a1a24",
        "panel_elev": "#22222e",
        "fg": "#ececf1",
        "muted": "#8e8e9e",
        "accent": "#3b7cff",
        "accent_hover": "#2f68e6",
        "canvas_bg": "#08080c",
        "text_bg": "#1c1c26",
        "text_fg": "#f4f4f8",
        "border": "#34343f",
        "menu_bg": "#262630",
        "menu_fg": "#ececf1",
    }
    root.configure(bg=C["bg"])
    # Popdown list (dropdown) for ttk.Combobox uses an internal Listbox on some platforms.
    root.option_add("*TCombobox*Listbox.background", C["text_bg"])
    root.option_add("*TCombobox*Listbox.foreground", C["text_fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure("Main.TFrame", background=C["bg"])
    style.configure("Card.TFrame", background=C["panel"], relief="flat")
    style.configure("Card.TLabel", background=C["panel"], foreground=C["fg"], font=(ui_font, 11))
    style.configure("TLabel", background=C["bg"], foreground=C["fg"], font=(ui_font, 11))
    style.configure("Head.TLabel", background=C["bg"], foreground=C["fg"], font=(ui_font, 18, "bold"))
    style.configure("Sub.TLabel", background=C["bg"], foreground=C["muted"], font=(ui_font, 10))
    style.configure("Hint.TLabel", background=C["bg"], foreground=C["muted"], font=(ui_font, 9))
    style.configure("TButton", font=(ui_font, 10), padding=(10, 5))
    style.configure(
        "Accent.TButton",
        font=(ui_font, 10, "bold"),
        padding=(14, 8),
    )
    style.map(
        "Accent.TButton",
        background=[("active", C["accent_hover"]), ("pressed", "#2558d4")],
    )
    try:
        style.configure(
            "Accent.TButton",
            background=C["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focuscolor="none",
        )
    except tk.TclError:
        pass
    style.configure("TCheckbutton", background=C["bg"], foreground=C["fg"], font=(ui_font, 10))
    style.configure("TSpinbox", fieldbackground=C["text_bg"], foreground=C["fg"], font=(ui_font, 10))
    style.map("TSpinbox", fieldbackground=[("readonly", C["text_bg"])])
    style.configure(
        "TCombobox",
        fieldbackground=C["text_bg"],
        foreground=C["text_fg"],
        font=(ui_font, 10),
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("readonly", C["text_bg"]),
            ("disabled", C["panel"]),
        ],
        foreground=[
            ("readonly", C["text_fg"]),
            ("disabled", C["muted"]),
        ],
    )
    style.configure("TLabelframe", background=C["bg"], foreground=C["fg"], font=(ui_font, 11, "bold"))
    style.configure("TLabelframe.Label", background=C["bg"], foreground=C["fg"])
    style.configure(
        "Card.TLabelframe",
        background=C["panel"],
        foreground=C["fg"],
        font=(ui_font, 11, "bold"),
    )
    style.configure("Card.TLabelframe.Label", background=C["panel"], foreground=C["fg"])
    style.configure("HintCard.TLabel", background=C["panel"], foreground=C["muted"], font=(ui_font, 9))
    return style, C, ui_font


def _run_ui(
    im,
    result,
    source_path: Path,
    *,
    sample_text: str,
) -> None:
    import math
    import tkinter as tk
    from tkinter import colorchooser, filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    iw, ih = im.size
    disp_max_h = 720
    scale = min(1.0, disp_max_h / ih)
    dw, dh = int(iw * scale), int(ih * scale)

    fonts = _discover_font_presets()
    if not fonts:
        fonts = [("Default (Pillow)", "")]
    font_labels = [f[0] for f in fonts]
    font_path_by_label = dict(fonts)

    z0 = next(
        (z for z in result.zones if z.name == result.recommended_zone),
        result.zones[0] if result.zones else None,
    )
    if z0 is None:
        return
    l0, t0, r0, b0 = z0.bbox
    img_cx = (l0 + r0) // 2
    img_cy = (t0 + b0) // 2

    root = tk.Tk()
    root.title("reelcribe — Reels thumbnail")
    root.minsize(1020, 640)
    _style, C, ui_font = _apply_reels_thumb_theme(root)

    main = ttk.Frame(root, style="Main.TFrame", padding=20)
    main.pack(fill="both", expand=True)

    header = ttk.Frame(main, style="Main.TFrame")
    header.pack(fill="x", pady=(0, 12))
    ttk.Label(header, text="Reels thumbnail", style="Head.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text=(
            "Σύρε τον τίτλο στο preview. Για διαφορετικό χρώμα σε «Your» και «title»: "
            "επίλεξε τη λέξη με το ποντίκι → «Χρώμα στην επιλογή» ή δεξί κλικ στο κείμενο."
        ),
        style="Sub.TLabel",
        wraplength=880,
    ).pack(anchor="w", pady=(8, 0))

    work = ttk.Frame(main, style="Main.TFrame")
    work.pack(fill="both", expand=True)

    # Left: all controls in a scrollable column (avoids clipped buttons at the bottom).
    SIDEBAR_W = 400
    sidebar_outer = ttk.Frame(work, style="Main.TFrame")
    sidebar_outer.pack(side="left", fill="both", expand=False, padx=(0, 12))

    sb_row = ttk.Frame(sidebar_outer, style="Main.TFrame")
    sb_row.pack(fill="both", expand=True)

    sb_canvas = tk.Canvas(
        sb_row,
        width=SIDEBAR_W,
        highlightthickness=0,
        bg=C["bg"],
    )
    vsb = ttk.Scrollbar(sb_row, orient="vertical", command=sb_canvas.yview)
    ctrl = ttk.Frame(sb_canvas, style="Main.TFrame")
    ctrl_win = sb_canvas.create_window((0, 0), window=ctrl, anchor="nw")

    def _scroll_region(_: tk.Event | None = None) -> None:
        br = sb_canvas.bbox("all")
        if br:
            sb_canvas.configure(scrollregion=br)

    def _canvas_width(ev: tk.Event) -> None:
        w = max(1, int(ev.width))
        sb_canvas.itemconfigure(ctrl_win, width=w)
        _scroll_region()

    ctrl.bind("<Configure>", _scroll_region)
    sb_canvas.bind("<Configure>", _canvas_width)

    sb_canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    sb_canvas.configure(yscrollcommand=vsb.set)

    def _sb_wheel(ev: tk.Event) -> None:
        d = getattr(ev, "delta", 0) or 0
        if d:
            sb_canvas.yview_scroll(int(-1 * (d / 120)), "units")
            return
        num = getattr(ev, "num", None)
        if num == 4:
            sb_canvas.yview_scroll(-1, "units")
        elif num == 5:
            sb_canvas.yview_scroll(1, "units")

    def _bind_sidebar_wheel_recursive(w: tk.Misc) -> None:
        """Scroll the left panel with the wheel; skip Text (editor) and Spinbox (value)."""
        for child in w.winfo_children():
            _bind_sidebar_wheel_recursive(child)
        if isinstance(w, (tk.Text, tk.Spinbox)):
            return
        w.bind("<MouseWheel>", _sb_wheel)
        w.bind("<Button-4>", _sb_wheel)
        w.bind("<Button-5>", _sb_wheel)

    preview_col = ttk.Frame(work, style="Main.TFrame")
    preview_col.pack(side="left", fill="both", expand=True)

    preview_card = ttk.Frame(preview_col, style="Card.TFrame", padding=12)
    preview_card.pack(fill="both", expand=True)

    font_choice = tk.StringVar(value=font_labels[0])
    size_var = tk.IntVar(value=42)
    stroke_w_var = tk.IntVar(value=3)
    text_spread_var = tk.DoubleVar(value=1.0)
    auto_color_var = tk.BooleanVar(value=True)
    fill_rgb: list[int] = [250, 250, 250]
    stroke_rgb: list[int] = [0, 0, 0]

    photo_holder: list[ImageTk.PhotoImage | None] = [None]
    caption_photo: list[ImageTk.PhotoImage | None] = [None]
    caption_item: list[int | None] = [None]
    canvas = tk.Canvas(
        preview_card,
        width=dw,
        height=dh,
        highlightthickness=1,
        highlightbackground=C["border"],
        bg=C["canvas_bg"],
    )
    canvas.pack()

    text_lf = ttk.LabelFrame(
        ctrl,
        text="Κείμενο & χρώματα ανά επιλογή",
        padding=(14, 12),
        style="Card.TLabelframe",
    )
    text_lf.pack(fill="x", pady=(0, 8))
    ttk.Label(
        text_lf,
        text="Επίλεξε λέξη/φράση με το ποντίκι, μετά χρώμα (κουμπί παρακάτω ή δεξί κλικ).",
        style="HintCard.TLabel",
    ).pack(anchor="w", pady=(0, 10))

    text_widget = tk.Text(
        text_lf,
        height=5,
        width=42,
        wrap="word",
        font=(ui_font, 13),
        bg=C["text_bg"],
        fg=C["text_fg"],
        insertbackground=C["text_fg"],
        selectbackground=C["accent"],
        selectforeground="#ffffff",
        relief="flat",
        highlightthickness=1,
        highlightbackground=C["border"],
        highlightcolor=C["accent"],
        padx=12,
        pady=10,
    )
    text_widget.insert("1.0", sample_text)
    text_widget.pack(fill="both", expand=True)

    logo_src: list[Image.Image | None] = [None]
    logo_photo: list[ImageTk.PhotoImage | None] = [None]
    logo_cx_img: list[int] = [int(iw * 0.86)]
    logo_cy_img: list[int] = [int(ih * 0.10)]
    logo_w_var = tk.IntVar(value=max(20, min(160, dw // 5)))
    drag_logo: dict[str, bool | float] = {
        "active": False,
        "resize": False,
        "w0": 0.0,
        "lcx": 0.0,
        "lcy": 0.0,
        "d0": 0.0,
    }

    highlight_name: list[str | None] = [result.recommended_zone]

    def current_font_path() -> str:
        return font_path_by_label.get(font_choice.get(), "")

    def canvas_xy_from_image(ix: int, iy: int) -> tuple[float, float]:
        return ix * scale, iy * scale

    def image_xy_from_canvas(cx: float, cy: float) -> tuple[int, int]:
        return int(cx / scale), int(cy / scale)

    def redraw_base() -> None:
        hl = highlight_name[0] or result.recommended_zone
        base = render_preview(
            im,
            result,
            text="",
            highlight_zone_name=hl,
            zone_name=hl,
        )
        disp = base.resize((dw, dh), Image.Resampling.LANCZOS)
        photo_holder[0] = ImageTk.PhotoImage(disp)
        canvas.delete("bg")
        canvas.create_image(0, 0, anchor="nw", image=photo_holder[0], tags="bg")

    def refresh_text_item() -> None:
        canvas.delete("caption")
        caption_item[0] = None
        txt = text_widget.get("1.0", "end-1c")
        if not txt.strip():
            return
        lyr = render_text_layer_rgba(
            (iw, ih),
            result,
            text=txt,
            text_center_xy=(img_cx, img_cy),
            zone_name=highlight_name[0],
            font_path=current_font_path() or None,
            font_size=int(size_var.get()),
            fill_rgb=tuple(fill_rgb) if not auto_color_var.get() else None,
            stroke_rgb=tuple(stroke_rgb) if not auto_color_var.get() else None,
            stroke_width=int(stroke_w_var.get()),
            auto_colors_from_zone=auto_color_var.get(),
            text_max_width_mult=float(text_spread_var.get()),
        )
        lyr_s = lyr.resize((dw, dh), Image.Resampling.LANCZOS)
        cap = ImageTk.PhotoImage(lyr_s)
        caption_photo[0] = cap
        # Full-size overlay aligned with the background (nw 0,0). Using anchor=center
        # with (img_cx*scale, img_cy*scale) was wrong: it placed the bitmap center, not
        # the text center, so dragging broke.
        caption_item[0] = canvas.create_image(
            0, 0, image=caption_photo[0], anchor="nw", tags="caption"
        )
        canvas.tag_raise("caption")

    def _logo_canvas_size() -> tuple[int, int]:
        if logo_src[0] is None:
            return 0, 0
        ow, oh = logo_src[0].size
        w = int(logo_w_var.get())
        w = max(1, min(w, dw))
        h = max(1, int(oh * (w / ow)))
        return w, h

    def logo_hit_test_canvas(ex: float, ey: float) -> bool:
        if logo_src[0] is None:
            return False
        w_disp, h_disp = _logo_canvas_size()
        lcx, lcy = canvas_xy_from_image(logo_cx_img[0], logo_cy_img[0])
        pad = 8.0
        return (
            abs(ex - lcx) <= w_disp / 2 + pad
            and abs(ey - lcy) <= h_disp / 2 + pad
        )

    def logo_br_corner_canvas() -> tuple[float, float, float, float]:
        """Bottom-right corner (canvas px), half-diagonal for resize."""
        w_disp, h_disp = _logo_canvas_size()
        lcx, lcy = canvas_xy_from_image(logo_cx_img[0], logo_cy_img[0])
        bx = lcx + w_disp / 2
        by = lcy + h_disp / 2
        d0 = math.hypot(w_disp / 2, h_disp / 2)
        return bx, by, d0, float(w_disp)

    def logo_corner_resize_hit(ex: float, ey: float) -> bool:
        if logo_src[0] is None:
            return False
        bx, by, _, _ = logo_br_corner_canvas()
        return math.hypot(ex - bx, ey - by) <= 16.0

    def refresh_logo_item() -> None:
        canvas.delete("logo")
        canvas.delete("logo_handle")
        logo_photo[0] = None
        if logo_src[0] is None:
            return
        w_disp, h_disp = _logo_canvas_size()
        resized = logo_src[0].resize((w_disp, h_disp), Image.Resampling.LANCZOS)
        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")
        logo_photo[0] = ImageTk.PhotoImage(resized)
        lcx, lcy = canvas_xy_from_image(logo_cx_img[0], logo_cy_img[0])
        canvas.create_image(lcx, lcy, image=logo_photo[0], anchor="center", tags="logo")
        bx, by, _, _ = logo_br_corner_canvas()
        hr = 7
        canvas.create_rectangle(
            bx - hr,
            by - hr,
            bx + hr,
            by + hr,
            outline=C["accent"],
            width=2,
            tags="logo_handle",
        )
        canvas.tag_raise("logo")
        canvas.tag_raise("logo_handle")

    def refresh_all_overlays() -> None:
        refresh_text_item()
        refresh_logo_item()

    def apply_all() -> None:
        nonlocal img_cx, img_cy
        img_cx, img_cy = clamp_point_to_zones(img_cx, img_cy, result.zones)
        zhit = zone_for_point(img_cx, img_cy, result.zones)
        highlight_name[0] = zhit.name if zhit else result.recommended_zone
        redraw_base()
        refresh_all_overlays()
        root.title(
            f"reelcribe-reels-thumb — {highlight_name[0]} @ ({img_cx}, {img_cy})"
        )

    drag: dict[str, float | bool] = {"active": False, "x": 0.0, "y": 0.0}
    grab_radius = max(100.0, min(dw, dh) * 0.22)

    def on_press(e: tk.Event) -> None:
        if logo_src[0] is not None:
            if logo_corner_resize_hit(float(e.x), float(e.y)):
                w_disp, h_disp = _logo_canvas_size()
                lcx, lcy = canvas_xy_from_image(logo_cx_img[0], logo_cy_img[0])
                d_press = math.hypot(float(e.x) - lcx, float(e.y) - lcy)
                d_ref = math.hypot(w_disp / 2, h_disp / 2)
                if d_press < 6.0:
                    d_press = d_ref
                drag_logo["active"] = True
                drag_logo["resize"] = True
                drag_logo["w0"] = float(w_disp)
                drag_logo["lcx"] = lcx
                drag_logo["lcy"] = lcy
                drag_logo["d0"] = d_press
                return
            if logo_hit_test_canvas(float(e.x), float(e.y)):
                drag_logo["active"] = True
                drag_logo["resize"] = False
                return
        if not text_widget.get("1.0", "end-1c").strip():
            return
        sx, sy = canvas_xy_from_image(img_cx, img_cy)
        dist = ((e.x - sx) ** 2 + (e.y - sy) ** 2) ** 0.5
        if dist > grab_radius:
            return
        drag["active"] = True
        drag["x"] = float(e.x)
        drag["y"] = float(e.y)

    def on_release(_: tk.Event) -> None:
        if drag_logo["active"]:
            drag_logo["active"] = False
            drag_logo["resize"] = False
            return
        if not drag["active"]:
            return
        drag["active"] = False
        apply_all()

    def on_motion(e: tk.Event) -> None:
        nonlocal img_cx, img_cy
        if drag_logo["active"] and drag_logo.get("resize"):
            lcx = float(drag_logo["lcx"])
            lcy = float(drag_logo["lcy"])
            d0 = float(drag_logo["d0"])
            w0 = float(drag_logo["w0"])
            d1 = math.hypot(float(e.x) - lcx, float(e.y) - lcy)
            s = d1 / max(1.0, d0)
            w_new = max(20, min(dw, int(w0 * s)))
            logo_w_var.set(w_new)
            refresh_logo_item()
            return
        if drag_logo["active"]:
            ix, iy = image_xy_from_canvas(float(e.x), float(e.y))
            logo_cx_img[0] = max(0, min(iw - 1, ix))
            logo_cy_img[0] = max(0, min(ih - 1, iy))
            refresh_logo_item()
            return
        if not drag["active"]:
            return
        ix, iy = image_xy_from_canvas(float(e.x), float(e.y))
        img_cx, img_cy = clamp_point_to_zones(ix, iy, result.zones)
        zhit = zone_for_point(img_cx, img_cy, result.zones)
        highlight_name[0] = zhit.name if zhit else highlight_name[0]
        refresh_text_item()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_release)

    def set_preset(
        fr: tuple[int, int, int], sr: tuple[int, int, int]
    ) -> None:
        auto_color_var.set(False)
        fill_rgb[:] = list(fr)
        stroke_rgb[:] = list(sr)
        refresh_text_item()

    def pick_fill() -> None:
        auto_color_var.set(False)
        c = colorchooser.askcolor(title="Χρώμα κειμένου")
        if c[0] is None:
            return
        r, g, b = (int(x) for x in c[0])
        fill_rgb[:] = [r, g, b]
        refresh_text_item()

    def pick_stroke() -> None:
        auto_color_var.set(False)
        c = colorchooser.askcolor(title="Χρώμα outline")
        if c[0] is None:
            return
        r, g, b = (int(x) for x in c[0])
        stroke_rgb[:] = [r, g, b]
        refresh_text_item()

    def browse_font() -> None:
        p = filedialog.askopenfilename(
            filetypes=[("Fonts", "*.ttf *.ttc *.otf"), ("All", "*.*")]
        )
        if not p:
            return
        label = os.path.basename(p)
        font_path_by_label[label] = p
        if label not in font_labels:
            font_labels.append(label)
            combo_font["values"] = tuple(font_labels)
        font_choice.set(label)
        refresh_text_item()

    def insert_markup_example() -> None:
        text_widget.insert(
            "insert",
            '[#fafafa]Λέξη ανοιχτή[/] [#2a6b9c]και μπλε[/] κείμενο χωρίς tag',
        )
        refresh_text_item()

    def wrap_selection_with_color() -> None:
        """Wrap selected text in [#RRGGBB]…[/], or insert empty span at cursor."""
        c = colorchooser.askcolor(title="Χρώμα (επιλογή ή κενό tag)")
        if not c[0]:
            return
        r, g, b = (int(x) for x in c[0])
        hx = f"{r:02x}{g:02x}{b:02x}"
        open_tag = f"[#{hx}]"
        close_tag = "[/]"
        sel = text_widget.tag_ranges(tk.SEL)
        if len(sel) == 2:
            a, bidx = sel[0], sel[1]
            chunk = text_widget.get(a, bidx)
            text_widget.delete(a, bidx)
            text_widget.insert(a, f"{open_tag}{chunk}{close_tag}")
        else:
            ins = text_widget.index(tk.INSERT)
            text_widget.insert(ins, f"{open_tag}{close_tag}")
            text_widget.mark_set(tk.INSERT, f"{ins}+{len(open_tag)}c")
        refresh_text_item()

    def word_color_wizard() -> None:
        plain = plain_text_from_markup(text_widget.get("1.0", "end-1c"))
        words = plain.split()
        if not words:
            messagebox.showinfo("Κενό", "Βάλε κείμενο πρώτα (ή χωρίς markup).")
            return
        colors: list[tuple[int, int, int]] = [(250, 250, 250)] * len(words)
        top = tk.Toplevel(root)
        top.title("Χρώμα ανά λέξη")
        for i, w in enumerate(words):
            tk.Label(top, text=w[:48] + ("…" if len(w) > 48 else "")).grid(
                row=i, column=0, sticky="w", padx=4, pady=2
            )

            def pick(i: int = i) -> None:
                c = colorchooser.askcolor(title="Χρώμα λέξης")
                if c[0]:
                    colors[i] = tuple(int(x) for x in c[0])

            tk.Button(top, text="Χρώμα…", command=pick).grid(row=i, column=1, padx=4)

        def ok() -> None:
            parts: list[str] = []
            for i, w in enumerate(words):
                r, g, b = colors[i]
                hx = f"{r:02x}{g:02x}{b:02x}"
                parts.append(f"[#{hx}]{w}[/]")
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", " ".join(parts))
            top.destroy()
            refresh_text_item()

        tk.Button(top, text="OK", command=ok).grid(row=len(words), column=0, columnspan=2, pady=8)

    def char_color_wizard() -> None:
        plain = plain_text_from_markup(text_widget.get("1.0", "end-1c"))
        if not plain:
            messagebox.showinfo("Κενό", "Βάλε κείμενο πρώτα.")
            return
        chars = list(plain)
        if len(chars) > 500:
            messagebox.showwarning(
                "Πολλοί χαρακτήρες", "Μέχρι 500 χαρακτήρες σε αυτόν τον οδηγό."
            )
            return
        colors: list[tuple[int, int, int]] = [(250, 250, 250)] * len(chars)
        top = tk.Toplevel(root)
        top.title("Χρώμα ανά γράμμα")
        top.minsize(360, 200)

        body: tk.Frame | tk.Canvas
        if len(chars) > 40:
            outer = tk.Frame(top)
            outer.pack(fill="both", expand=True, padx=8, pady=8)
            cv = tk.Canvas(outer, height=min(480, 24 * len(chars) + 40), highlightthickness=0)
            sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
            inner = tk.Frame(cv)
            inner_id = cv.create_window((0, 0), window=inner, anchor="nw")

            def _scroll(_: tk.Event | None = None) -> None:
                cv.configure(scrollregion=cv.bbox("all"))
                cv.itemconfigure(inner_id, width=cv.winfo_width())

            inner.bind("<Configure>", _scroll)
            cv.bind("<Configure>", _scroll)

            def _wheel(ev: tk.Event) -> None:
                d = getattr(ev, "delta", 0) or 0
                if d:
                    cv.yview_scroll(-1 if d > 0 else 1, "units")

            cv.bind("<MouseWheel>", _wheel)
            cv.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            cv.configure(yscrollcommand=sb.set)
            body = inner
        else:
            body = tk.Frame(top)
            body.pack(fill="both", expand=True, padx=8, pady=8)

        for i, ch in enumerate(chars):
            disp = "⏎" if ch == "\n" else ("␣" if ch == " " else ch)
            tk.Label(body, text=disp, width=4).grid(row=i, column=0, sticky="w")

            def pick(i: int = i) -> None:
                c = colorchooser.askcolor(title="Χρώμα χαρακτήρα")
                if c[0]:
                    colors[i] = tuple(int(x) for x in c[0])

            tk.Button(body, text="Χρώμα…", command=pick).grid(row=i, column=1, padx=4)

        def ok() -> None:
            parts: list[str] = []
            for i, ch in enumerate(chars):
                r, g, b = colors[i]
                hx = f"{r:02x}{g:02x}{b:02x}"
                parts.append(f"[#{hx}]{ch}[/]")
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", "".join(parts))
            top.destroy()
            refresh_text_item()

        tk.Button(top, text="OK", command=ok).pack(pady=8)

    def export_png() -> None:
        fp = current_font_path()
        auto = auto_color_var.get()
        fr = tuple(fill_rgb) if not auto else None
        sr = tuple(stroke_rgb) if not auto else None
        pv = render_preview(
            im,
            result,
            text=text_widget.get("1.0", "end-1c"),
            text_center_xy=(img_cx, img_cy),
            font_path=fp or None,
            font_size=int(size_var.get()),
            fill_rgb=fr,
            stroke_rgb=sr,
            stroke_width=int(stroke_w_var.get()),
            auto_colors_from_zone=auto,
            highlight_zone_name=highlight_name[0],
            text_max_width_mult=float(text_spread_var.get()),
            draw_zones=False,
            draw_face=False,
        )
        zname = highlight_name[0] or "thumb"
        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{source_path.stem}_preview_{zname}.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if not out:
            return
        if logo_src[0] is not None:
            w_full = max(1, int(round(logo_w_var.get() / scale)))
            w_full = min(w_full, iw)
            pv = paste_logo_centered(
                pv,
                logo_src[0],
                center_xy=(logo_cx_img[0], logo_cy_img[0]),
                width_px=w_full,
            )
        pv.save(out, "PNG", optimize=True)
        messagebox.showinfo("Saved", out)

    def load_logo() -> None:
        p = filedialog.askopenfilename(
            title="Λογότυπο",
            filetypes=[
                ("PNG / WebP / JPEG", "*.png *.webp *.jpg *.jpeg"),
                ("All", "*.*"),
            ],
        )
        if not p:
            return
        path = Path(p).expanduser().resolve()
        try:
            lg = load_image(path).convert("RGBA")
        except OSError as exc:
            messagebox.showerror("Λογότυπο", str(exc))
            return
        logo_src[0] = lg
        refresh_logo_item()

    def clear_logo() -> None:
        logo_src[0] = None
        refresh_logo_item()

    def _text_context_menu(ev: tk.Event) -> None:
        menu = tk.Menu(
            root,
            tearoff=0,
            bg=C["menu_bg"],
            fg=C["menu_fg"],
            activebackground=C["accent"],
            activeforeground="#ffffff",
        )
        menu.add_command(label="Χρώμα στην επιλογή…", command=wrap_selection_with_color)
        try:
            menu.tk_popup(ev.x_root, ev.y_root)
        finally:
            menu.grab_release()

    text_widget.bind("<KeyRelease>", lambda _e: refresh_text_item())
    text_widget.bind("<Button-3>", _text_context_menu)
    text_widget.bind("<Button-2>", _text_context_menu)
    text_widget.bind("<Control-Button-1>", _text_context_menu)

    actions = ttk.Frame(ctrl, style="Main.TFrame")
    actions.pack(fill="x", pady=(12, 0))
    ttk.Button(actions, text="Εφαρμογή ζωνών", command=apply_all).pack(fill="x", pady=3)
    ttk.Button(
        actions,
        text="Χρώμα στην επιλογή…",
        style="Accent.TButton",
        command=wrap_selection_with_color,
    ).pack(fill="x", pady=3)
    ttk.Button(actions, text="Παράδειγμα markup", command=insert_markup_example).pack(
        fill="x", pady=3
    )
    ttk.Button(actions, text="Χρώμα ανά λέξη…", command=word_color_wizard).pack(
        fill="x", pady=3
    )
    ttk.Button(actions, text="Χρώμα ανά γράμμα…", command=char_color_wizard).pack(
        fill="x", pady=3
    )

    font_row = ttk.Frame(ctrl, style="Main.TFrame")
    font_row.pack(fill="x", pady=(14, 0))
    ttk.Label(font_row, text="Γραμματοσειρά:").pack(anchor="w")
    combo_font = ttk.Combobox(
        font_row,
        textvariable=font_choice,
        values=tuple(font_labels),
        width=34,
        state="readonly",
    )
    combo_font.pack(fill="x", pady=(4, 6))
    combo_font.bind("<<ComboboxSelected>>", lambda _e: refresh_text_item())
    ttk.Button(font_row, text="Άλλο αρχείο…", command=browse_font).pack(fill="x", pady=2)

    size_row = ttk.Frame(ctrl, style="Main.TFrame")
    size_row.pack(fill="x", pady=(6, 0))
    ttk.Label(size_row, text="Μέγεθος γραμματοσειράς (px):").pack(anchor="w")

    def sync_size_widgets() -> None:
        v = max(12, min(220, int(size_var.get())))
        size_var.set(v)
        sz_scale.set(v)
        refresh_text_item()

    def bump_size(delta: int) -> None:
        v = max(12, min(220, int(size_var.get()) + delta))
        size_var.set(v)
        sz_scale.set(v)
        refresh_text_item()

    def on_size_scale(val: str) -> None:
        v = int(round(float(val)))
        size_var.set(v)
        refresh_text_item()

    sz_scale = tk.Scale(
        size_row,
        from_=12,
        to=220,
        orient=tk.HORIZONTAL,
        command=on_size_scale,
        length=320,
        resolution=1,
        bg=C["bg"],
        fg=C["fg"],
        highlightthickness=0,
        troughcolor=C["panel"],
    )
    sz_scale.pack(fill="x", pady=(4, 6))
    sz_scale.set(int(size_var.get()))

    size_btns = ttk.Frame(size_row, style="Main.TFrame")
    size_btns.pack(fill="x")
    ttk.Button(size_btns, text="−", width=3, command=lambda: bump_size(-4)).pack(
        side="left"
    )
    sz_spin = tk.Spinbox(
        size_btns,
        from_=12,
        to=220,
        textvariable=size_var,
        width=5,
        bg=C["text_bg"],
        fg=C["text_fg"],
        insertbackground=C["text_fg"],
        highlightthickness=0,
    )
    sz_spin.pack(side="left", padx=6)
    ttk.Button(size_btns, text="+", width=3, command=lambda: bump_size(4)).pack(
        side="left"
    )
    ttk.Label(size_btns, text="ή σύρε τη γραμμή", style="Hint.TLabel").pack(
        side="left", padx=(10, 0)
    )
    sz_spin.bind("<ButtonRelease-1>", lambda _e: sync_size_widgets())
    sz_spin.bind("<KeyRelease>", lambda _e: sync_size_widgets())

    outline_row = ttk.Frame(ctrl, style="Main.TFrame")
    outline_row.pack(fill="x", pady=(10, 0))
    ttk.Label(outline_row, text="Outline px:").pack(side="left")
    sw_spin = tk.Spinbox(
        outline_row,
        from_=0,
        to=8,
        textvariable=stroke_w_var,
        width=5,
        bg=C["text_bg"],
        fg=C["text_fg"],
        insertbackground=C["text_fg"],
        highlightthickness=0,
    )
    sw_spin.pack(side="left", padx=(8, 0))
    sw_spin.bind("<ButtonRelease-1>", lambda _e: refresh_text_item())
    sw_spin.bind("<KeyRelease>", lambda _e: refresh_text_item())

    ttk.Checkbutton(
        ctrl,
        text="Αυτόματα χρώματα (αντίθεση ζώνης)",
        variable=auto_color_var,
        command=refresh_text_item,
    ).pack(anchor="w", pady=(10, 0))

    spread_row = ttk.Frame(ctrl, style="Main.TFrame")
    spread_row.pack(fill="x", pady=(14, 0))
    ttk.Label(spread_row, text="Απλωμα κειμένου (×πλάτος ζώνης):").pack(anchor="w")
    ttk.Label(
        spread_row,
        text="1 = wrap μόνο μέσα στη ζώνη · μεγαλύτερο = πιο πλατές μέχρι το πλάτος της εικόνας.",
        style="Sub.TLabel",
        wraplength=360,
    ).pack(anchor="w", pady=(2, 6))

    def on_text_spread_scale(val: str) -> None:
        text_spread_var.set(float(val))
        refresh_text_item()

    sp_scale = tk.Scale(
        spread_row,
        from_=1.0,
        to=3.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        command=on_text_spread_scale,
        length=320,
        bg=C["bg"],
        fg=C["fg"],
        highlightthickness=0,
        troughcolor=C["panel"],
    )
    sp_scale.pack(fill="x")
    sp_scale.set(1.0)

    help_lbl = ttk.Label(
        ctrl,
        text=(
            "Markup: [#RRGGBB]λέξη[/] ή [#γέμισμα:#outline]λέξη[/]. "
            "Χρώμα στην επιλογή: επίλεξε κείμενο και «Χρώμα στην επιλογή». "
            "Σύρσιμο τίτλου: κάνε κλικ κοντά στο κείμενο (όχι στα μαύρα περιθώρια)."
        ),
        style="Sub.TLabel",
        wraplength=360,
    )
    help_lbl.pack(anchor="w", pady=(12, 0))

    presets_card = ttk.LabelFrame(
        ctrl,
        text="Προτεινόμενα χρώματα",
        padding=(12, 10),
        style="Card.TLabelframe",
    )
    presets_card.pack(fill="x", pady=(12, 0))
    for label, fr, sr in _OPTIMA_COLOR_PRESETS:
        ttk.Button(
            presets_card,
            text=label,
            command=lambda fr=fr, sr=sr: set_preset(fr, sr),
        ).pack(fill="x", pady=3)

    colors = ttk.Frame(ctrl, style="Main.TFrame")
    colors.pack(fill="x", pady=(10, 0))
    ttk.Button(colors, text="Χρώμα κειμένου…", command=pick_fill).pack(fill="x", pady=3)
    ttk.Button(colors, text="Χρώμα outline…", command=pick_stroke).pack(fill="x", pady=3)

    btns = ttk.Frame(ctrl, style="Main.TFrame")
    btns.pack(fill="x", pady=(16, 8))
    ttk.Button(
        btns,
        text="Εξαγωγή PNG…",
        style="Accent.TButton",
        command=export_png,
    ).pack(fill="x", pady=3)
    ttk.Button(btns, text="Κλείσιμο", command=root.destroy).pack(fill="x", pady=3)

    logo_lf = ttk.LabelFrame(
        ctrl,
        text="Λογότυπο (PNG με διαφάνεια)",
        padding=(14, 12),
        style="Card.TLabelframe",
    )
    ttk.Label(
        logo_lf,
        text=(
            "Σύρε για μετακίνηση · σύρε τη μπλε γωνία (κάτω-δεξιά) για αλλαγή μεγέθους. "
            "Πλάτος spinbox = μέγεθος στο preview (εξαγωγή στο 1080×1920)."
        ),
        style="HintCard.TLabel",
        wraplength=340,
    ).pack(anchor="w", pady=(0, 8))
    lr = ttk.Frame(logo_lf, style="Card.TFrame")
    lr.pack(fill="x")
    ttk.Button(lr, text="Φόρτωση λογότυπου…", command=load_logo).pack(fill="x", pady=2)
    ttk.Button(lr, text="Αφαίρεση", command=clear_logo).pack(fill="x", pady=2)
    sz_logo = ttk.Frame(logo_lf, style="Card.TFrame")
    sz_logo.pack(fill="x", pady=(8, 0))
    ttk.Label(sz_logo, text="Πλάτος στο preview (px):").pack(anchor="w")
    logo_spin = tk.Spinbox(
        sz_logo,
        from_=20,
        to=max(20, dw),
        textvariable=logo_w_var,
        width=6,
        bg=C["text_bg"],
        fg=C["text_fg"],
        insertbackground=C["text_fg"],
        highlightthickness=0,
    )
    logo_spin.pack(anchor="w", pady=4)
    logo_spin.bind("<ButtonRelease-1>", lambda _e: refresh_logo_item())
    logo_spin.bind("<KeyRelease>", lambda _e: refresh_logo_item())
    logo_lf.pack(fill="x", pady=(12, 0), after=text_lf)

    _bind_sidebar_wheel_recursive(sidebar_outer)

    apply_all()
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
