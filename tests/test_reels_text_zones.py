"""Tests for reels_text_zones (no MediaPipe required for scoring logic)."""

from PIL import Image

from reelcribe.reels_text_zones import (
    FaceBox,
    ZoneScore,
    _intersection_area,
    _max_text_width_for_zone,
    analyze_thumbnail,
    clamp_point_to_zones,
    ensure_1080x1920,
    mean_luminance_region,
    parse_colored_markup,
    paste_logo_centered,
    plain_text_from_markup,
    render_preview,
    text_style_for_luminance,
)


def test_intersection_area_disjoint():
    assert _intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_intersection_area_overlap():
    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    assert _intersection_area(a, b) == 50 * 50


def test_text_style_midpoint():
    assert text_style_for_luminance(50) == "light_on_dark"
    assert text_style_for_luminance(200) == "dark_on_light"


def test_analyze_without_faces_prefers_top_right_order():
    im = Image.new("RGB", (1080, 1920), color=(40, 40, 40))
    r = analyze_thumbnail(im, faces=[])
    assert r.recommended_zone == "top_right"


def test_face_blocking_top_right_moves_recommendation():
    """Large face in top-right should invalidate top_right first."""
    im = Image.new("RGB", (1080, 1920), (60, 60, 60))
    # Cover top-right quadrant heavily inside inflated margin
    face = FaceBox(left=700, top=80, width=320, height=400)
    r = analyze_thumbnail(im, faces=[face])
    assert r.recommended_zone != "top_right"


def test_ensure_1080x1920_noop_for_exact():
    im = Image.new("RGB", (1080, 1920), (1, 1, 1))
    out, padded = ensure_1080x1920(im)
    assert not padded
    assert out.size == (1080, 1920)


def test_ensure_1080x1920_converts_landscape():
    im = Image.new("RGB", (1920, 1080), (2, 2, 2))
    out, padded = ensure_1080x1920(im)
    assert padded
    assert out.size == (1080, 1920)


def test_mean_luminance_black_white():
    black = Image.new("RGB", (100, 100), (0, 0, 0))
    white = Image.new("RGB", (100, 100), (255, 255, 255))
    assert mean_luminance_region(black, (0, 0, 100, 100)) < 10
    assert mean_luminance_region(white, (0, 0, 100, 100)) > 240


def test_render_preview_round_trip():
    im = Image.new("RGB", (1080, 1920), (100, 100, 100))
    r = analyze_thumbnail(im, faces=[])
    out = render_preview(im, r, text="X", zone_name="top_right")
    assert out.size == (1080, 1920)


def test_parse_colored_markup_basic():
    segs = parse_colored_markup("α [#ff0000]β[/] γ")
    assert len(segs) == 3
    assert segs[0][0] == "α "
    assert segs[0][1] is None
    assert segs[1][0] == "β"
    assert segs[1][1] == (255, 0, 0)
    assert segs[2][0] == " γ"


def test_plain_text_from_markup():
    assert plain_text_from_markup("[#ff0000]hi[/]") == "hi"


def test_clamp_point_to_zones():
    zones = [
        ZoneScore(
            name="top_right",
            bbox=(500, 0, 1000, 400),
            overlap_ratio=0.0,
            valid=True,
            mean_luminance=128.0,
            text_style="dark_on_light",
        )
    ]
    x, y = clamp_point_to_zones(2000, 2000, zones)
    assert x < 1000 and y < 400


def test_max_text_width_mult_caps_to_canvas():
    z = ZoneScore(
        name="bottom_left",
        bbox=(0, 1400, 500, 1900),
        overlap_ratio=0.0,
        valid=True,
        mean_luminance=50.0,
        text_style="light_on_dark",
    )
    pad = 16
    iw = 1080
    base = _max_text_width_for_zone(z, pad, iw, 1.0)
    wide = _max_text_width_for_zone(z, pad, iw, 3.0)
    assert wide >= base
    assert wide <= iw - 2 * pad


def test_paste_logo_centered():
    base = Image.new("RGB", (100, 100), (0, 128, 0))
    logo = Image.new("RGBA", (20, 20), (255, 0, 0, 200))
    out = paste_logo_centered(base, logo, center_xy=(50, 50), width_px=40)
    assert out.size == (100, 100)
    assert out.mode == "RGB"
