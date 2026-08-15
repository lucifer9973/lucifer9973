from __future__ import annotations

from pathlib import Path
import math
import random
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.optimize import linear_sum_assignment
import cairosvg

ROOT = Path('/mnt/data/profile_build')
ROOT.mkdir(exist_ok=True)
PHOTO = Path('/mnt/data/photo.png')
LOGO_DIR = ROOT / 'logos'
LOGO_DIR.mkdir(exist_ok=True)

WIDTH, HEIGHT = 1180, 610
PORTRAIT_W, PORTRAIT_H = 400, 492
PANEL_X, PANEL_Y = 36, 84
RNG = np.random.default_rng(729)

DARK = {
    'bg': '#070B16',
    'panel': '#0A101F',
    'top': '#0B1222',
    'stroke': '#243247',
    'muted': '#94A3B8',
    'muted2': '#475569',
    'text': '#F8FAFC',
    'ui': '#22D3EE',
    'ui2': '#0891B2',
    'portrait': '#A78BFA',
    'accent': '#10B981',
    'purple': '#7C3AED',
    'live': '#F87171',
    'pill': '#4C1D95',
    'pill_text': '#E9D5FF',
}
LIGHT = {
    'bg': '#F8FAFC',
    'panel': '#FFFFFF',
    'top': '#F1F5F9',
    'stroke': '#CBD5E1',
    'muted': '#475569',
    'muted2': '#94A3B8',
    'text': '#0F172A',
    'ui': '#0891B2',
    'ui2': '#0E7490',
    'portrait': '#7C3AED',
    'accent': '#059669',
    'purple': '#6D28D9',
    'live': '#DC2626',
    'pill': '#EDE9FE',
    'pill_text': '#5B21B6',
}


def load_and_segment_photo(path: Path) -> Image.Image:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]

    # Lightweight GrabCut on a reduced image, followed by brightness gating.
    scale = min(0.42, 800 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    smask = np.zeros(small.shape[:2], np.uint8)
    rect = (max(4, int(small.shape[1] * 0.03)), max(4, int(small.shape[0] * 0.02)),
            int(small.shape[1] * 0.94), int(small.shape[0] * 0.95))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, smask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    fg = np.where((smask == cv2.GC_FGD) | (smask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The source has a very dark studio backdrop. Keep the subject while suppressing the halo.
    luminance_keep = (gray >= 13).astype(np.uint8)
    fg = (fg & luminance_keep).astype(np.uint8)

    # Keep the largest connected foreground component and lightly close small gaps.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == largest).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # The supplied portrait has a nearly-black studio backdrop and dark clothing.
    # Use a tight, deterministic silhouette guide so dark mode does not turn the backdrop
    # into a cloud of dots. GrabCut remains the interior refinement step.
    silhouette = np.zeros((h, w), np.uint8)
    head_poly = np.array([
        [330, 360], [360, 285], [420, 220], [500, 185], [590, 175], [680, 210],
        [745, 275], [775, 370], [770, 500], [750, 610], [700, 700], [650, 770],
        [560, 805], [460, 790], [385, 735], [335, 650], [315, 540], [315, 445]
    ], np.int32)
    shoulder_poly = np.array([
        [290, 735], [390, 760], [470, 805], [535, 825], [610, 805], [690, 760],
        [785, 805], [900, 900], [1030, 1040], [1080, 1240], [1085, 1450],
        [30, 1450], [35, 1280], [90, 1130], [160, 1030], [220, 900]
    ], np.int32)
    cv2.fillPoly(silhouette, [head_poly, shoulder_poly], 1)
    # Soften the hand mask edges without reopening the backdrop.
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    fg = (fg & silhouette).astype(np.uint8)

    # Crop head + shoulders with a stable aspect ratio close to the prompt's 300x340 grid.
    x0, x1 = int(w * 0.08), int(w * 0.92)
    y0, y1 = int(h * 0.06), int(h * 0.88)
    crop = img[y0:y1, x0:x1]
    cmask = fg[y0:y1, x0:x1]

    target_ratio = 300 / 340
    ch, cw = crop.shape[:2]
    desired_w = min(cw, int(ch * target_ratio))
    desired_h = min(ch, int(cw / target_ratio))
    cx = cw // 2
    cy = int(ch * 0.48)
    left = max(0, cx - desired_w // 2)
    top = max(0, cy - desired_h // 2)
    left = min(left, cw - desired_w)
    top = min(top, ch - desired_h)
    crop = crop[top:top + desired_h, left:left + desired_w]
    cmask = cmask[top:top + desired_h, left:left + desired_w]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, (cmask * 255).astype(np.uint8)])
    return Image.fromarray(rgba, 'RGBA').resize((300, 340), Image.Resampling.LANCZOS)


def floyd_steinberg(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = gray.astype(np.float32).copy()
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.uint8)
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    for y in range(h):
        x_range = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        for x in x_range:
            if not mask[y, x]:
                arr[y, x] = 0
                continue
            old = arr[y, x]
            new = 255 if old >= 128 else 0
            out[y, x] = 1 if new == 255 else 0
            err = old - new
            # Hard-clear diffusion bleed at the mask edge.
            if y + 1 < h and mask[y + 1, x]:
                arr[y + 1, x] += err * 7 / 16
            if y + 1 < h and x - 1 >= 0 and mask[y + 1, x - 1]:
                arr[y + 1, x - 1] += err * 3 / 16
            if y + 1 < h and x + 1 < w and mask[y + 1, x + 1]:
                arr[y + 1, x + 1] += err * 1 / 16
            nx = x + 1 if y % 2 == 0 else x - 1
            if 0 <= nx < w and mask[y, nx]:
                arr[y, nx] += err * 7 / 16
    return out


def portrait_points():
    im = load_and_segment_photo(PHOTO)
    rgb = np.array(im.convert('RGB'))
    alpha = np.array(im.getchannel('A')) / 255.0
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    # Build a subject-only mask for dark mode. The photo has a black studio backdrop,
    # so flat black regions are excluded while facial detail, hair edges and the shirt
    # silhouette remain available to the dither.
    gray8 = lum.astype(np.uint8)
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    local = cv2.GaussianBlur(gray8.astype(np.float32), (0, 0), 2)
    local2 = cv2.GaussianBlur((gray8.astype(np.float32)) ** 2, (0, 0), 2)
    local_std = np.sqrt(np.maximum(local2 - local * local, 0))
    base_mask = ((alpha > 0.10) & ((gray8 >= 18) | (edge >= 18) | (local_std >= 4))).astype(np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    p = np.clip((lum / 255.0 - 0.46) * 1.25 + 0.46, 0, 1) * 255
    p[base_mask == 0] = 0
    dark_bin = floyd_steinberg(p, base_mask)

    # Light mode follows the prompt by retaining the image/background and emphasizing dark structure.
    light_source = np.clip((255 - lum) * 1.12, 0, 255)
    light_bin = floyd_steinberg(light_source, base_mask)

    def to_panel_points(binary: np.ndarray):
        ys, xs = np.where(binary > 0)
        sx, sy = 1.20, 1.30
        ox, oy = 57, 110
        pts = [(ox + float(x) * sx, oy + float(y) * sy) for x, y in zip(xs, ys)]
        return pts

    return to_panel_points(dark_bin), to_panel_points(light_bin)


def path_runs(points_set: set[tuple[int, int]], y_scale: float = 1.0, x_scale: float = 1.0):
    by_y: dict[int, list[int]] = {}
    for x, y in points_set:
        by_y.setdefault(y, []).append(x)
    chunks: list[tuple[int, int, int]] = []
    for y, xs in by_y.items():
        xs.sort()
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                chunks.append((start, prev, y))
                start = prev = x
        chunks.append((start, prev, y))
    d = []
    for x0, x1, y in chunks:
        X = 57 + x0 * x_scale
        Y = 110 + y * y_scale
        w = (x1 - x0 + 1) * x_scale
        d.append(f'M{X:.2f},{Y:.2f}h{w:.2f}v1.45h-{w:.2f}z')
    return ''.join(d)


def render_logo_points(svg_path: Path | None, custom_svg: str | None = None, count: int = 800, crop_box=None):
    tmp = LOGO_DIR / ((svg_path.stem if svg_path else 'react') + '.png')
    if custom_svg is not None:
        custom_path = LOGO_DIR / 'react-source.svg'
        custom_path.write_text(custom_svg)
        svg_path = custom_path
    cairosvg.svg2png(url=str(svg_path), write_to=str(tmp), output_width=256, output_height=256, background_color='transparent')
    im = Image.open(tmp).convert('RGBA')
    if crop_box is not None:
        im = im.crop(crop_box)
    im = im.resize((256, 256), Image.Resampling.LANCZOS)
    a = np.array(im.getchannel('A'))
    ys, xs = np.where(a > 90)
    coords = np.stack([xs, ys], axis=1)
    if len(coords) == 0:
        raise RuntimeError(f'Logo mask empty: {svg_path}')
    idx = RNG.choice(len(coords), size=count, replace=len(coords) < count)
    sel = coords[idx].astype(float)
    sel[:, 0] = 236 + (sel[:, 0] - 128) * 2.05
    sel[:, 1] = 332 + (sel[:, 1] - 128) * 2.05
    return sel


def make_github_svg() -> str:
    return """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 98 96\">
      <path fill=\"#ffffff\" fill-rule=\"evenodd\" d=\"M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z"/>
    </svg>"""

def make_react_svg() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
      <g fill="none" stroke="#ffffff" stroke-width="18">
        <ellipse cx="128" cy="128" rx="96" ry="36" transform="rotate(0 128 128)"/>
        <ellipse cx="128" cy="128" rx="96" ry="36" transform="rotate(60 128 128)"/>
        <ellipse cx="128" cy="128" rx="96" ry="36" transform="rotate(120 128 128)"/>
      </g>
      <circle cx="128" cy="128" r="15" fill="#ffffff"/>
    </svg>'''


def text_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))


def row_svg(label: str, value: str, y: float, dark: dict[str, str], delay: float, font_size: float = 14.0) -> str:
    dots = '.' * 52
    return (
        f'<g opacity="0">'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="{delay:.2f}s" fill="freeze"/>'
        f'<animateTransform attributeName="transform" type="translate" values="-8 0;0 0" dur="0.4s" begin="{delay:.2f}s" fill="freeze"/>'
        f'<text x="470" y="{y:.1f}" font-size="{font_size}" textLength="655" lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
        f'<tspan fill="{dark["ui"]}">{text_escape(label)} </tspan>'
        f'<tspan fill="{dark["muted"]}" opacity="0.35">{dots}</tspan>'
        f'<tspan fill="{dark["text"]}" font-weight="600"> {text_escape(value)}</tspan>'
        f'</text></g>'
    )


def make_banner(dark: dict[str, str], dark_mode: bool, portrait: list[tuple[float, float]], logo_sets: list[np.ndarray]) -> str:
    font = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono','Courier New',monospace"
    bg = dark['bg']
    panel = dark['panel']
    top = dark['top']
    stroke = dark['stroke']
    # Build portrait groups from stable random chunks so the intro scatters across the whole frame.
    pts = portrait.copy()
    random.Random(42).shuffle(pts)
    chunks = np.array_split(np.array(pts, dtype=float), 60)
    portrait_intro = []
    for i, c in enumerate(chunks):
        if len(c) == 0:
            continue
        d = ''.join(f'M{x:.2f},{y:.2f}h2.25v1.45h-2.25z' for x, y in c)
        begin = 0.18 + i * 0.028
        portrait_intro.append(
            f'<path d="{d}" fill="{dark["portrait"]}" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" dur="2s" begin="{begin:.3f}s" fill="freeze"/>'
            f'</path>'
        )
    portrait_intro_svg = ''.join(portrait_intro)

    # Organic portrait drift bands: this follows the working pattern from the reference profile.
    # Each band is dense during the portrait phase, then drifts toward the first morph target and fades.
    drift_rng = np.random.default_rng(20260815)
    portrait_np = np.asarray(pts, dtype=float)
    order = np.argsort(portrait_np[:, 1])
    sorted_pts = portrait_np[order]
    bands = np.array_split(sorted_pts, 94)
    # First morph target centroid.
    first_logo_centroid = np.mean(logo_sets[0], axis=0)
    drift_bands = []
    for idx, band in enumerate(bands):
        if len(band) == 0:
            continue
        d = ''.join(f'M{x:.2f},{y:.2f}h2.25v1.45h-2.25z' for x, y in band)
        c = band.mean(axis=0)
        vec = first_logo_centroid - c
        norm = max(float(np.linalg.norm(vec)), 1.0)
        # Keep the drift bounded and add small noise so the bands don't form a rigid square lattice.
        scale = min(0.42, 92.0 / norm)
        dx, dy = vec * scale
        dx += float(drift_rng.normal(0, 4.0))
        dy += float(drift_rng.normal(0, 4.0))
        opacity = '1;1;0;0;0;0;0;0;1'
        key_times = '0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000'
        drift_bands.append(
            f'<g opacity="1">'
            f'<animate attributeName="opacity" values="{opacity}" keyTimes="{key_times}" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;0 0;{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};0 0" keyTimes="{key_times}" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>'
            f'<path d="{d}"/>'
            f'</g>'
        )
    drift_bands_svg = ''.join(drift_bands)

    # Traveler dots: the visible morph layer. It follows the exact timing pattern of the reference.
    start_idx = np.array(RNG.choice(len(portrait_np), size=len(logo_sets[0]), replace=False))
    starts = portrait_np[start_idx].copy()
    current = starts.copy()
    matched = [current]
    for target in logo_sets:
        _, c = linear_sum_assignment(((current[:, None, :] - target[None, :, :]) ** 2).sum(axis=2))
        nxt = target[c]
        matched.append(nxt)
        current = nxt
    matched.append(starts)

    travelers = []
    key_times = '0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000'
    loop = '13.9s'
    for i in range(len(starts)):
        positions = ['%.2f %.2f' % tuple(matched[j][i]) for j in range(5)]
        values = [positions[0], positions[0], positions[1], positions[1], positions[2], positions[2], positions[3], positions[3], positions[4]]
        travelers.append(
            f'<use href="#travelerDot" opacity="0">'
            f'<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" keyTimes="{key_times}" dur="{loop}" begin="3.2s" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="{";".join(values)}" keyTimes="{key_times}" dur="{loop}" begin="3.2s" repeatCount="indefinite"/>'
            f'<animate attributeName="fill" values="{dark["portrait"]};{dark["portrait"]};{dark["ui"]};{dark["ui"]};{dark["purple"]};{dark["purple"]};{dark["accent"]};{dark["accent"]};{dark["portrait"]}" keyTimes="{key_times}" dur="{loop}" begin="3.2s" repeatCount="indefinite"/>'
            f'</use>'
        )
    travelers_svg = ''.join(travelers)

    rows = [
        ('Subject', 'Shobhit Raj'),
        ('Role', 'Product Engineer Intern'),
        ('Origin', 'Gurugram, India'),
        ('Education', 'B.Tech CSE @ KIIT University'),
        ('Status', 'Single · Spectent'),
        ('ToolChain', 'VS Code, Git, GitHub, Postman, Docker'),
        ('Core.Lang', 'Python, JavaScript/TypeScript, SQL'),
        ('Core.Frontend', 'React.js, TypeScript, JavaScript, Tailwind CSS, HTML5, CSS3'),
        ('Core.Backend', 'FastAPI, Flask, Node.js, Express.js, REST APIs'),
        ('Core.Database', 'PostgreSQL, MongoDB, Redis, SQLAlchemy'),
        ('Core.Infra', 'Docker, AWS, Git/GitHub, Linux'),
        ('Grid.Mail', 'rajshobhit48@gmail.com'),
        ('Grid.Portfolio', 'space-portfolio-umber.vercel.app'),
        ('Grid.LinkedIn', 'shobhitraj-ai'),
        ('Grid.GitHub', '@lucifer9973'),
        ('Grid.X', '@Shobhitraj729'),
        ('Grid.Instagram', '@shobhitraj729'),
    ]
    row_y = 162
    rows_svg = []
    for idx, (lab, val) in enumerate(rows):
        y = row_y + idx * 22.0
        if idx == 11:
            rows_svg.append(
                f'<text x="470" y="{y-8:.1f}" font-size="13" fill="{dark["muted"]}">- Contact</text>'
                f'<line x1="542" y1="{y-12:.1f}" x2="1125" y2="{y-12:.1f}" stroke="{dark["stroke"]}"/>'
            )
            y += 2
        rows_svg.append(row_svg(lab, val, y, dark, 0.90 + idx * 0.11, 14.0))
    rows_svg = ''.join(rows_svg)

    # Dark/light-specific gradient definitions.
    light_panel = dark_mode is False
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{font}" role="img" aria-label="Shobhit Raj — profile.sh --live">
<defs>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{dark['purple']}"><animate attributeName="stop-color" values="{dark['purple']};{dark['ui']};{dark['accent']};{dark['purple']}" dur="10s" repeatCount="indefinite"/></stop>
    <stop offset="0.5" stop-color="{dark['ui']}"><animate attributeName="stop-color" values="{dark['ui']};{dark['accent']};{dark['purple']};{dark['ui']}" dur="10s" repeatCount="indefinite"/></stop>
    <stop offset="1" stop-color="{dark['accent']}"><animate attributeName="stop-color" values="{dark['accent']};{dark['purple']};{dark['ui']};{dark['accent']}" dur="10s" repeatCount="indefinite"/></stop>
  </linearGradient>
  <linearGradient id="portraitGrad" x1="0" y1="0" x2="0" y2="520" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="{dark['portrait']}"/>
    <stop offset="0.5" stop-color="{dark['portrait']}"/>
    <stop offset="1" stop-color="{dark['ui']}"/>
    <animateTransform attributeName="gradientTransform" type="translate" values="0 -120;0 120;0 -120" dur="9s" repeatCount="indefinite"/>
  </linearGradient>
  <filter id="glow8" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="8"/></filter>
  <filter id="glow3" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3"/></filter>
  <filter id="txtGlow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="0.9" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <clipPath id="winClip"><rect x="2" y="2" width="1176" height="606" rx="18"/></clipPath>
  <rect id="travelerDot" x="-1.8" y="-1.2" width="4.0" height="2.6" rx="0.8" fill="{dark['portrait']}" shape-rendering="crispEdges"/>
</defs>
<rect x="2" y="2" width="1176" height="606" rx="18" fill="{dark['bg']}"/>
<g clip-path="url(#winClip)">
  <rect x="2" y="2" width="1176" height="606" fill="{dark['panel']}"/>
  <rect x="2" y="2" width="1176" height="46" fill="{dark['top']}"/>
  <line x1="2" y1="48" x2="1178" y2="48" stroke="{dark['stroke']}"/>
  <circle cx="30" cy="25" r="5.5" fill="#ff5f56"/><circle cx="50" cy="25" r="5.5" fill="#ffbd2e"/><circle cx="70" cy="25" r="5.5" fill="#27c93f"/>
  <text x="590" y="29" text-anchor="middle" font-size="12" fill="{dark['muted']}">rajshobhit48@gmail.com · % ./profile.sh --live</text>

  <text x="38" y="74" font-size="10" letter-spacing="3" fill="{dark['muted2']}">VISUAL.MAP</text>
  <rect x="36" y="84" width="400" height="492" rx="10" fill="none" stroke="{dark['ui']}" stroke-width="2" opacity="0.45" filter="url(#glow3)"/>
  <rect x="36" y="84" width="400" height="492" rx="10" fill="{dark['panel']}" stroke="{dark['ui']}" stroke-opacity="0.35"/>
  <path d="M50 84H36V98M422 84H436V98M50 576H36V562M422 576H436V562" fill="none" stroke="{dark['ui']}" stroke-width="2" opacity="0.8"/>
  <g transform="translate(0 0)">
    <g fill="{dark['portrait']}" shape-rendering="crispEdges">
      <set attributeName="opacity" to="0" begin="3.2s"/>
      {portrait_intro_svg}
    </g>
    <g fill="{dark['portrait']}" shape-rendering="crispEdges" opacity="0">
      <set attributeName="opacity" to="1" begin="3.2s"/>
      {drift_bands_svg}
    </g>
    <g>{travelers_svg}</g>
  </g>

  <text x="470" y="106" font-size="13" letter-spacing="2" fill="{dark['ui']}" filter="url(#txtGlow)">SYSTEM.INFO</text>
  <line x1="566" y1="102" x2="1061" y2="102" stroke="{dark['stroke']}"/>
  <text x="1125" y="106" text-anchor="end" font-size="12" fill="{dark['live']}" font-weight="700"><tspan>●</tspan> LIVE<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/></text>
  <g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="0.6s" fill="freeze"/>
    <rect x="470" y="122" width="190" height="20" rx="4" fill="{dark['pill']}"/>
    <text x="479" y="136" font-size="13" font-weight="700" fill="{dark['pill_text']}">lucifer9973</text>
    <line x1="665" y1="132" x2="1125" y2="132" stroke="{dark['stroke']}"/>
  </g>
  {rows_svg}
  <g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="3.30s" fill="freeze"/>
    <text x="470" y="582" font-size="13" fill="{dark['muted']}">▸ More about me &amp; projects below in README ↓ <tspan fill="{dark['ui']}">█<animate attributeName="fill-opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/></tspan></text>
  </g>
</g>
<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" stroke="url(#accent)" stroke-width="3" opacity="0.55" filter="url(#glow8)"/>
<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" stroke="url(#accent)" stroke-width="1.6"/>
</svg>'''
    return svg


def write_outputs():
    dark_points, light_points = portrait_points()
    # Keep light mode in the same practical size range as dark mode while preserving the same visual character.
    if len(light_points) > 10000:
        light_points = light_points[::2]
    # Logo sources: GitHub mark path, React canonical three-orbit geometry, and Docker symbol artwork.
    github_svg = LOGO_DIR / 'github-source.svg'
    docker_svg = Path('/mnt/data/docker.svg')
    react_src = make_react_svg()
    github_svg.write_text(make_github_svg())
    logos = [
        render_logo_points(github_svg, count=900),
        render_logo_points(None, custom_svg=react_src, count=900),
        render_logo_points(docker_svg, count=900, crop_box=(0, 0, 256, 188)),
    ]
    dark_svg = make_banner(DARK, True, dark_points, logos)
    light_svg = make_banner(LIGHT, False, light_points, logos)
    (ROOT / 'dark.svg').write_text(dark_svg)
    (ROOT / 'light.svg').write_text(light_svg)
    # Store source data for reproducibility.
    np.save(ROOT / 'portrait-dark.npy', np.asarray(dark_points))
    np.save(ROOT / 'portrait-light.npy', np.asarray(light_points))
    for i, arr in enumerate(logos, 1):
        np.save(ROOT / f'logo-{i}.npy', arr)
    print('dark', len(dark_svg), 'light', len(light_svg))


if __name__ == '__main__':
    write_outputs()
