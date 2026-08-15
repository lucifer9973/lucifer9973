from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Iterable

import cv2
import cairosvg
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "assets" / "photo.png"
LOGOS = ROOT / "assets" / "logos"
OUT_DARK = ROOT / "dark.svg"
OUT_LIGHT = ROOT / "light.svg"

W, H = 1180, 610
BG_DARK = "#0A101F"
BG_LIGHT = "#FFFFFF"
PANEL_DARK = "#0D1628"
PANEL_LIGHT = "#F8FAFC"
BORDER_DARK = "#243247"
BORDER_LIGHT = "#CBD5E1"
UI_DARK = "#22D3EE"
UI_LIGHT = "#0891B2"
PORTRAIT_DARK = "#A78BFA"
PORTRAIT_LIGHT = "#7C3AED"
ACCENT = "#10B981"
TEXT_DARK = "#F8FAFC"
MUTED_DARK = "#94A3B8"
TEXT_LIGHT = "#0F172A"
MUTED_LIGHT = "#475569"
RED = "#EF4444"

# Terminal geometry: left ~38%, right ~62%.
TERM_X, TERM_Y = 8, 8
TERM_W, TERM_H = 1164, 594
BODY_Y = 68
LEFT_X, LEFT_Y, LEFT_W, LEFT_H = 28, 96, 418, 474
RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H = 474, 96, 674, 474
PORTRAIT_X, PORTRAIT_Y = 47, 116
PORTRAIT_W, PORTRAIT_H = 380, 430
GRID_W, GRID_H = 300, 340

RNG = np.random.default_rng(9973)


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % rgb


def parse_svg_paths(svg_text: str) -> str:
    # Font Awesome brand SVGs contain one or more <path ... d="..."> elements.
    paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg_text)
    if not paths:
        raise ValueError("No path data found in logo SVG")
    return " ".join(paths)


def rasterize_logo(svg_path: Path, target_w: int, target_h: int) -> np.ndarray:
    svg = svg_path.read_text(encoding="utf-8")
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=target_w, output_height=target_h)
    arr = np.array(Image.open(__import__("io").BytesIO(png)).convert("RGBA"))
    alpha = arr[..., 3]
    # Padding-safe binary mask.
    return alpha > 30


def crop_head_shoulders(im: Image.Image) -> Image.Image:
    # The supplied portrait is already centered head-and-shoulders.
    # Use a deterministic crop that preserves hair, shoulders and shirt.
    w, h = im.size
    target_ratio = 300 / 340
    crop_h = int(h * 0.96)
    crop_w = int(crop_h * target_ratio)
    crop_w = min(crop_w, w)
    x0 = (w - crop_w) // 2
    y0 = max(0, int(h * 0.03))
    y1 = min(h, y0 + crop_h)
    if y1 - y0 < crop_h:
        y0 = max(0, h - crop_h)
    return im.crop((x0, y0, x0 + crop_w, y0 + crop_h))


def background_mask(arr_rgb: np.ndarray) -> np.ndarray:
    # Matches the master prompt's color-distance + morphology approach.
    h, w, _ = arr_rgb.shape
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    corner = np.concatenate(
        [
            lab[: max(25, h // 12), : max(25, w // 12)].reshape(-1, 3),
            lab[: max(25, h // 12), -max(25, w // 12) :].reshape(-1, 3),
            lab[-max(25, h // 12) :, : max(25, w // 12)].reshape(-1, 3),
            lab[-max(25, h // 12) :, -max(25, w // 12) :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corner, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    # Conservative threshold; the face and glasses separate strongly from the dark background.
    subject = dist > 9.5
    kernel = np.ones((9, 9), np.uint8)
    subject = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    # Fill holes and keep the largest meaningful connected component.
    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(subject)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        subject = mask
    return subject > 0


def prepare_binary_density(im: Image.Image, dark: bool) -> np.ndarray:
    crop = crop_head_shoulders(im)
    # Keep contrast exactly in the neighborhood specified by the prompt.
    crop = ImageOps.autocontrast(crop.convert("RGB"), cutoff=1)
    crop = ImageEnhance.Contrast(crop).enhance(1.3)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=3))
    crop = crop.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    rgb = np.array(crop)

    if dark:
        mask = background_mask(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Within subject, darker pixels -> more visible dots.
        subject_tone = gray
        subject_tone = cv2.normalize(subject_tone, None, 0, 255, cv2.NORM_MINMAX)
        subject_tone = np.where(mask, subject_tone, 0).astype(np.uint8)
        # Thresholded Floyd-Steinberg image.
        pil = Image.fromarray(subject_tone, mode="L")
        binary = np.array(pil.convert("1", dither=Image.Dither.FLOYDSTEINBERG), dtype=np.uint8) * 255
        binary[~mask] = 0
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        tone = 255 - gray
        pil = Image.fromarray(tone.astype(np.uint8), mode="L")
        binary = np.array(pil.convert("1", dither=Image.Dither.FLOYDSTEINBERG), dtype=np.uint8) * 255

    return binary > 0


def mask_points(mask: np.ndarray, max_points: int | None = None) -> np.ndarray:
    ys, xs = np.where(mask)
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    if max_points is not None and len(pts) > max_points:
        idx = RNG.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def mask_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return horizontal runs as (y, x0, x1_exclusive)."""
    runs: list[tuple[int, int, int]] = []
    h, w = mask.shape
    for y in range(h):
        row = mask[y]
        xs = np.flatnonzero(row)
        if len(xs) == 0:
            continue
        start = int(xs[0])
        prev = start
        for x in xs[1:]:
            x = int(x)
            if x != prev + 1:
                runs.append((y, start, prev + 1))
                start = x
            prev = x
        runs.append((y, start, prev + 1))
    return runs


def run_path(run: tuple[int, int, int], sx: float, sy: float, ox: float, oy: float, r: float = 0.72) -> str:
    y, x0, x1 = run
    px = ox + x0 * sx - r
    py = oy + y * sy - r
    width = max(1.0, (x1 - x0) * sx + 2 * r)
    height = max(1.0, 2 * r)
    return f"M{px:.2f},{py:.2f}h{width:.2f}v{height:.2f}h-{width:.2f}z"


def choose_dense_portrait_points(mask: np.ndarray, count: int = 900) -> np.ndarray:
    pts = mask_points(mask)
    if len(pts) < count:
        # Repeat deterministically only when necessary.
        idx = RNG.choice(len(pts), size=count, replace=True)
        return pts[idx]
    # Stratify by x/y buckets to avoid over-clustering in the face.
    bins = []
    gx, gy = 30, 30
    for by in range(gy):
        for bx in range(gx):
            x0, x1 = bx * GRID_W / gx, (bx + 1) * GRID_W / gx
            y0, y1 = by * GRID_H / gy, (by + 1) * GRID_H / gy
            sub = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)]
            if len(sub):
                bins.append(sub[RNG.integers(0, len(sub))])
    bins = np.array(bins)
    if len(bins) >= count:
        idx = RNG.choice(len(bins), size=count, replace=False)
        return bins[idx]
    idx = RNG.choice(len(pts), size=count - len(bins), replace=False)
    return np.vstack([bins, pts[idx]])


def fit_logo_mask(mask: np.ndarray, box_w: int = 245, box_h: int = 190) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty logo mask")
    minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = mask[miny : maxy + 1, minx : maxx + 1]
    ch, cw = cropped.shape
    scale = min(box_w / cw, box_h / ch)
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS)
    result = np.zeros((GRID_H, GRID_W), dtype=bool)
    x0 = (GRID_W - nw) // 2
    y0 = (GRID_H - nh) // 2
    result[y0 : y0 + nh, x0 : x0 + nw] = np.array(resized) > 80
    return result


def sample_mask_points(mask: np.ndarray, count: int = 900) -> np.ndarray:
    pts = mask_points(mask)
    if len(pts) == 0:
        raise ValueError("Empty logo mask after fitting")
    if len(pts) >= count:
        # Use deterministic quantile-ish sampling after sorting by angle around centroid.
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        pts = pts[order]
        idx = np.linspace(0, len(pts) - 1, count).astype(int)
        return pts[idx]
    idx = RNG.choice(len(pts), size=count, replace=True)
    return pts[idx]


def assignment_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    # Optimal transport-style one-to-one pairing using Hungarian assignment.
    d = src[:, None, :] - dst[None, :, :]
    cost = np.sum(d * d, axis=2)
    r, c = linear_sum_assignment(cost)
    out = np.zeros_like(src)
    out[r] = dst[c]
    return out


def dots_as_paths(points: Iterable[tuple[float, float]], scale_x: float, scale_y: float, ox: float, oy: float, r: float = 0.72) -> str:
    # Use path runs, not font glyphs. Horizontal unit rectangles are compact and crisp.
    parts = []
    for x, y in points:
        px = ox + x * scale_x - r
        py = oy + y * scale_y - r
        w = 2 * r
        parts.append(f"M{px:.2f},{py:.2f}h{w:.2f}v{w:.2f}h-{w:.2f}z")
    return "".join(parts)


def make_dense_groups(points: np.ndarray, group_count: int = 94) -> list[np.ndarray]:
    # Position + noise before grouping avoids mathematically clean bands/grid artifacts.
    noisy = points.copy()
    noisy[:, 0] += RNG.normal(0, 4, len(noisy))
    noisy[:, 1] += RNG.normal(0, 4, len(noisy))
    # Quantize by y with jitter, then split any oversized band by x when needed.
    idx = np.argsort(noisy[:, 1])
    groups: list[list[int]] = [[] for _ in range(group_count)]
    for n, original_idx in enumerate(idx):
        g = min(group_count - 1, int(n * group_count / len(idx)))
        groups[g].append(int(original_idx))
    return [np.array(g, dtype=int) for g in groups]


def text_row(label: str, value: str, y: int, theme: str, x_label: int = 500, x_lead1: int = 638, x_lead2: int = 812, x_val: int = 830) -> str:
    if theme == "dark":
        label_color, value_color, leader = MUTED_DARK, TEXT_DARK, BORDER_DARK
    else:
        label_color, value_color, leader = MUTED_LIGHT, TEXT_LIGHT, BORDER_LIGHT
    # Compute leader length from label/value columns rather than hard-code text in SVG.
    return (
        f'<text x="{x_label}" y="{y}" font-size="13" fill="{label_color}" letter-spacing="1.1" textLength="128" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>'
        f'<line x1="{x_lead1}" y1="{y-4}" x2="{x_lead2}" y2="{y-4}" stroke="{leader}" stroke-width="1" stroke-dasharray="1 6" stroke-linecap="round"/>'
        f'<text x="{x_val}" y="{y}" font-size="14" fill="{value_color}" textLength="315" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>'
    )


def svg_header(theme: str) -> tuple[str, str, str, str, str, str]:
    dark = theme == "dark"
    bg = BG_DARK if dark else BG_LIGHT
    panel = PANEL_DARK if dark else PANEL_LIGHT
    border = BORDER_DARK if dark else BORDER_LIGHT
    ui = UI_DARK if dark else UI_LIGHT
    portrait = PORTRAIT_DARK if dark else PORTRAIT_LIGHT
    text = TEXT_DARK if dark else TEXT_LIGHT
    return bg, panel, border, ui, portrait, text


def build_svg(theme: str, dense_mask: np.ndarray, logo_points: list[np.ndarray], portrait_points: np.ndarray) -> str:
    bg, panel, border, ui, portrait_color, text = svg_header(theme)
    muted = MUTED_DARK if theme == "dark" else MUTED_LIGHT
    raw_dense = mask_points(dense_mask)
    sx = PORTRAIT_W / GRID_W
    sy = PORTRAIT_H / GRID_H
    dense_screen = np.column_stack([PORTRAIT_X + raw_dense[:, 0] * sx, PORTRAIT_Y + raw_dense[:, 1] * sy])
    runs = mask_runs(dense_mask)
    # Keep the dark portrait close to the target density (~15–17k dots).
    # The light mode intentionally preserves the supplied photo background per the master prompt.
    # Make logo/portrait morph points use the exact same display transform.
    p0 = np.column_stack([PORTRAIT_X + portrait_points[:, 0] * sx, PORTRAIT_Y + portrait_points[:, 1] * sy])
    logos_screen = [np.column_stack([PORTRAIT_X + p[:, 0] * sx, PORTRAIT_Y + p[:, 1] * sy]) for p in logo_points]
    # Match consecutive states using one-to-one assignment.
    l1 = assignment_map(p0, logos_screen[0])
    l2 = assignment_map(l1, logos_screen[1])
    l3 = assignment_map(l2, logos_screen[2])
    p_back = assignment_map(l3, p0)

    # 94 drift bands. Runs are grouped by noisy y-position to retain compact SVG path runs.
    noisy_runs = []
    for r in runs:
        y, x0, x1 = r
        noisy_runs.append((y + float(RNG.normal(0, 4)), x0, x1, r))
    noisy_runs.sort(key=lambda v: v[0])
    groups: list[list[tuple[int,int,int]]] = [[] for _ in range(94)]
    for i, item in enumerate(noisy_runs):
        groups[min(93, int(i * 94 / max(1, len(noisy_runs))))].append(item[3])
    target_centroid = logos_screen[0].mean(axis=0)
    # Precompute group translations with the required ~42% drift.
    drift = target_centroid - np.array([PORTRAIT_X + PORTRAIT_W / 2, PORTRAIT_Y + PORTRAIT_H / 2])
    drift *= 0.42
    duration = 14.2
    times = [0, 3/14.2, 4.3/14.2, 6.3/14.2, 7.6/14.2, 9.6/14.2, 10.9/14.2, 12.9/14.2, 1]
    kt = ";".join(f"{x:.6f}" for x in times)

    # 60 scattered intro groups across the whole portrait. Runs are shuffled as units
    # so the SVG remains compact while the appearance is spatially interleaved.
    intro_groups = [[] for _ in range(60)]
    run_order = RNG.permutation(len(runs))
    for i, ridx in enumerate(run_order):
        intro_groups[i % 60].append(runs[int(ridx)])

    elements = []
    elements.append(f'<rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="16" fill="{bg}"/>')
    elements.append(f'<rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="16" fill="none" stroke="{border}" stroke-width="1.4"/>')
    elements.append(f'<rect x="{TERM_X+1}" y="{TERM_Y+1}" width="{TERM_W-2}" height="58" rx="15" fill="{panel}"/>')
    elements.append(f'<line x1="{TERM_X+1}" y1="{BODY_Y}" x2="{TERM_X+TERM_W-1}" y2="{BODY_Y}" stroke="{border}" stroke-width="1"/>')

    # Top bar.
    elements.append(f'<circle cx="31" cy="37" r="5" fill="#FF5F56"/><circle cx="49" cy="37" r="5" fill="#FFBD2E"/><circle cx="67" cy="37" r="5" fill="#27C93F"/>')
    elements.append(f'<text x="92" y="42" font-size="16" fill="{muted}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace">profile.sh --live</text>')
    elements.append(f'<circle cx="1070" cy="36" r="5" fill="{RED}"><animate attributeName="opacity" values="1;.35;1" dur="1.25s" repeatCount="indefinite"/></circle>')
    elements.append(f'<text x="1085" y="41" font-size="12" fill="{RED}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-weight="700">LIVE</text>')

    # Panels.
    elements.append(f'<rect x="{LEFT_X}" y="{LEFT_Y}" width="{LEFT_W}" height="{LEFT_H}" rx="12" fill="{panel}" stroke="{border}"/>')
    elements.append(f'<rect x="{RIGHT_X}" y="{RIGHT_Y}" width="{RIGHT_W}" height="{RIGHT_H}" rx="12" fill="{panel}" stroke="{border}"/>')
    elements.append(f'<text x="52" y="128" font-size="13" fill="{muted}" letter-spacing="2" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace">VISUAL.MAP</text>')

    # Portrait clip area.
    elements.append(f'<rect x="{PORTRAIT_X}" y="{PORTRAIT_Y}" width="{PORTRAIT_W}" height="{PORTRAIT_H}" rx="9" fill="{bg}" opacity=".25"/>')
    elements.append(f'<rect x="{PORTRAIT_X}" y="{PORTRAIT_Y}" width="{PORTRAIT_W}" height="{PORTRAIT_H}" rx="9" fill="none" stroke="{border}" stroke-width="1"/>')

    # Dense portrait: duplicate layer as required. Group by intro shards for the one-time shimmer.
    intro_layer_parts = []
    for i, idxs in enumerate(intro_groups):
        path = "".join(run_path(r, sx, sy, PORTRAIT_X, PORTRAIT_Y) for r in idxs)
        delay = 0.12 + (i / 60) * 1.65 + float(RNG.uniform(-0.03, 0.03))
        intro_layer_parts.append(f'<path d="{path}" fill="{portrait_color}" opacity="0"><animate attributeName="opacity" begin="{delay:.3f}s" dur="2s" values="0;1" fill="freeze"/></path>')
    elements.append('<g aria-label="portrait intro">' + ''.join(intro_layer_parts) + '</g>')
    # Duplicate dense portrait layer for loop / drift.
    elements.append('<g aria-label="portrait loop duplicate">')
    for gi, idxs in enumerate(groups):
        path = "".join(run_path(r, sx, sy, PORTRAIT_X, PORTRAIT_Y) for r in idxs)
        frac = gi / max(1, len(groups)-1)
        tx = drift[0] * (0.88 + 0.24 * math.sin(frac * math.pi))
        ty = drift[1] * (0.88 + 0.24 * math.cos(frac * math.pi))
        elements.append(
            f'<g transform="translate(0 0)"><path d="{path}" fill="{portrait_color}">'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{tx:.2f} {ty:.2f};{tx:.2f} {ty:.2f};0 0" keyTimes="0;0.211268;0.302817;0.908451" dur="{duration}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="1;1;0;0;1" keyTimes="0;0.211268;0.302817;0.908451;1" dur="{duration}s" repeatCount="indefinite"/></path></g>'
        )
    elements.append('</g>')

    # Traveller layer: 900 points, hidden in portrait phase, morphs via optimal transport mappings.
    trav = []
    for i in range(900):
        a = p0[i]; b = l1[i]; c = l2[i]; d = l3[i]; e = p_back[i]
        vals_x = ';'.join(f'{v[0]:.2f}' for v in [a,b,c,d,e])
        vals_y = ';'.join(f'{v[1]:.2f}' for v in [a,b,c,d,e])
        opacity_vals = '0;0;1;1;1;1;1;1;0'
        trav.append(
            f'<circle cx="{a[0]:.2f}" cy="{a[1]:.2f}" r="0.82" fill="{ui}">'
            f'<animate attributeName="cx" dur="{duration}s" repeatCount="indefinite" values="{vals_x}" keyTimes="{kt}"/>'
            f'<animate attributeName="cy" dur="{duration}s" repeatCount="indefinite" values="{vals_y}" keyTimes="{kt}"/>'
            f'<animate attributeName="opacity" dur="{duration}s" repeatCount="indefinite" values="{opacity_vals}" keyTimes="{kt}"/>'
            f'</circle>'
        )
    elements.append(f'<g aria-label="logo travellers">{"".join(trav)}</g>')

    # Right panel heading + handle pill.
    elements.append(f'<text x="500" y="127" font-size="13" fill="{muted}" letter-spacing="2" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace">SYSTEM.INFO</text>')
    elements.append(f'<rect x="1010" y="111" width="118" height="28" rx="14" fill="none" stroke="{ui}"/>')
    elements.append(f'<text x="1069" y="130" text-anchor="middle" font-size="14" fill="{ui}" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace">lucifer9973</text>')

    rows = [
        ("SUBJECT", "SHOBHIT RAJ"),
        ("ROLE", "PRODUCT ENGINEER INTERN"),
        ("ORIGIN", "GURUGRAM, INDIA"),
        ("EDUCATION", "B.TECH CSE @ KIIT UNIVERSITY"),
        ("STATUS", "SINGLE · SPECTENT"),
        ("TOOLCHAIN", "VS CODE · GIT · GITHUB · POSTMAN · DOCKER"),
        ("CORE.LANG", "PYTHON · JS/TS · SQL"),
        ("CORE.FRONTEND", "REACT.JS · TYPESCRIPT · TAILWIND CSS · HTML5 · CSS3"),
        ("CORE.BACKEND", "FASTAPI · FLASK · NODE · EXPRESS · REST APIS"),
        ("CORE.DATABASE", "POSTGRES · MONGO · REDIS · SQLALCHEMY"),
        ("CORE.INFRA", "DOCKER · AWS · GIT · GITHUB · LINUX"),
        ("GRID.MAIL", "RAJSHOBHIT48@GMAIL.COM"),
        ("GRID.PORTFOLIO", "SPACE-PORTFOLIO-UMBER.VERCEL.APP"),
        ("GRID.LINKEDIN", "SHOBHITRAJ-AI"),
        ("GRID.X", "@SHOBHITRAJ729"),
        ("GRID.INSTAGRAM", "@SHOBHITRAJ729"),
    ]
    y = 165
    for label, value in rows:
        elements.append(text_row(label, value, y, theme))
        y += 23

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<title>Shobhit Raj profile.sh live developer terminal</title>'
        + ''.join(elements)
        + '</svg>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the profile.sh --live SVG banner.")
    parser.add_argument("--photo", default=str(PHOTO))
    args = parser.parse_args()
    photo = Image.open(args.photo).convert("RGB")

    dark_mask = prepare_binary_density(photo, dark=True)
    light_mask = prepare_binary_density(photo, dark=False)
    # Portrait morph points derive from the dark portrait so the animation starts as the subject silhouette.
    portrait_points = choose_dense_portrait_points(dark_mask, 900)

    logo_paths = [LOGOS / "python.svg", LOGOS / "react.svg", LOGOS / "docker.svg"]
    logo_points = []
    for p in logo_paths:
        base = rasterize_logo(p, 320, 260)
        fitted = fit_logo_mask(base, 245, 190)
        logo_points.append(sample_mask_points(fitted, 900))

    data_dir = ROOT / "generator" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "dark_mask.npy", dark_mask)
    np.save(data_dir / "light_mask.npy", light_mask)
    np.save(data_dir / "portrait_points.npy", portrait_points)
    np.save(data_dir / "logo_python_points.npy", logo_points[0])
    np.save(data_dir / "logo_react_points.npy", logo_points[1])
    np.save(data_dir / "logo_docker_points.npy", logo_points[2])

    OUT_DARK.write_text(build_svg("dark", dark_mask, logo_points, portrait_points), encoding="utf-8")
    OUT_LIGHT.write_text(build_svg("light", light_mask, logo_points, portrait_points), encoding="utf-8")
    print(f"Generated {OUT_DARK} ({OUT_DARK.stat().st_size/1024:.1f} KB)")
    print(f"Generated {OUT_LIGHT} ({OUT_LIGHT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
