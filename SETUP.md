# J.A.R.V.I.S Developer Interface — Setup Guide

## Overview

This repository contains a JARVIS-themed GitHub profile built with pure SVG animations. The profile consists of:

- **Hero Dashboard** (1180×610) — Primary profile banner (`assets/hero-dark.svg` / `assets/hero-light.svg`)
- **Expanded Dashboard** (1536×1024) — Full system interface (`assets/jarvis-dark.svg` / `assets/jarvis-light.svg`)
- **README.md** — Comprehensive profile with stats, projects, and contact

## Tech Stack

- Pure SVG (no JavaScript)
- SMIL Animations (`animate`, `animateTransform`, `animateMotion`)
- SVG Filters (Gaussian Blur, Drop Shadow)
- CSS3 via `<style>` blocks
- Markdown with HTML embeds

## File Structure

```
lucifer9973/
├── README.md                 # Main profile README
├── SETUP.md                  # This file
├── LICENSE                   # MIT License
├── .gitignore                # Git ignore rules
├── dark.svg                  # Root dark wrapper
├── light.svg                 # Root light wrapper
├── assets/
│   ├── hero-dark.svg         # ★ Primary dark dashboard
│   ├── hero-light.svg        # Primary light dashboard
│   ├── jarvis-dark.svg       # Expanded dark dashboard
│   ├── jarvis-light.svg      # Expanded light dashboard
│   ├── dark.svg              # Asset dark wrapper
│   ├── light.svg             # Asset light wrapper
│   ├── code-typing-loop.gif  # Background animation
│   ├── matrixrain.gif        # Matrix rain effect
│   └── fonts/                # Reserved for web fonts
└── icons/                    # Reserved for custom icons
```

## How to Use

### 1. Fork or Clone

```bash
git clone https://github.com/lucifer9973/lucifer9973.git
cd lucifer9973
```

### 2. Customize for Your Profile

1. **Update personal details** in:
   - `assets/hero-dark.svg` — Name, role, company, education, mission
   - `assets/hero-light.svg` — Same (light mode variant)
   - `README.md` — All text sections
   - `assets/jarvis-dark.svg` — Extended details
   - `assets/jarvis-light.svg` — Extended details (light mode)

2. **Replace placeholder links** in `README.md`:
   - Portfolio URL
   - Project repository URLs
   - LinkedIn, email, etc.

3. **Configure GitHub Stats**:
   The README uses `github-readme-stats` API. Stats auto-populate when deployed.

### 3. Local Preview

Clone the repo and open any SVG in a browser:

```bash
# View the hero dashboard
start assets/hero-dark.svg

# Or open in browser directly
```

### 4. Deploy to GitHub Profile

1. Create repository `lucifer9973/lucifer9973` (must match your username)
2. Copy all files to root of the repository
3. Commit and push

```bash
git add .
git commit -m "Deploy JARVIS Developer Interface v4.2"
git push origin main
```

The dashboard will appear on your GitHub profile automatically.

## Color System

| Token | Hex | Usage |
|-------|-----|-------|
| Primary | `#00F5FF` | Cyan accents, headers, borders |
| Secondary | `#3D5AFE` | Blue accents, secondary elements |
| Accent | `#00FFC8` | Green accents, success states |
| Danger | `#FF1744` | Alerts, recording indicators |
| Warning | `#FFD600` | Warnings, attention markers |
| Background | `#05070A` | Main dark background |
| Panel | `#10161D` | Card/panel backgrounds |
| Glass | `rgba(255,255,255,0.05)` | Subtle overlays |

## Typography

| Font | Usage | Weight |
|------|-------|--------|
| Orbitron | Headers, titles, clock | 700, 900 |
| JetBrains Mono | Code, data, metrics | 400, 600, 700 |
| Rajdhani | Labels, descriptions | 400, 600, 700 |

## SVG Animation Reference

The dashboard uses pure SMIL animations:

- `<animate>` — Opacity, position, color transitions
- `<animateTransform>` — Rotation (rings, radar), translation (particles)
- `<animateMotion>` — Orbital paths, circuit traces
- `<animate attributeName="stroke-dashoffset">` — Flowing circuit lines
- `<animate attributeName="width">` — Progress bars, boot sequences

## Performance Notes

- All SVGs are optimized for GitHub rendering
- Avoids expensive filters where possible
- Gradients and filters are defined once in `<defs>` and reused
- No external dependencies — works offline

## Credits

Designed and engineered by **Shobhit Raj**
- [GitHub](https://github.com/lucifer9973)
- [LinkedIn](https://linkedin.com/in/shobhitraj-ai)
- Email: rajshobhit48@gmail.com

---

*J.A.R.V.I.S Developer Interface — Built with precision. Inspired by Stark Industries.*
