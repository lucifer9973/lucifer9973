# GitHub Profile — Setup

This repository implements the `profile.sh --live` banner, self-hosted stats layout, contribution snake, and social layer.

## 1. Profile repository

The repository must be named exactly:

`lucifer9973/lucifer9973`

Keep the repository public.

## 2. Banner files

`dark.svg` and `light.svg` are generated from `assets/photo.png` by:

`python generator/generate_banner.py`

The generator keeps the source masks and morph-point data under `generator/data/`.

The private source photo is intentionally ignored by Git via `.gitignore`. The generated SVGs are the public assets.

## 3. Self-host github-readme-stats

Create a GitHub classic token:

- GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate new token (classic)
- Expiration: No expiration
- Scope: `repo`
- Copy the token immediately and treat it like a password.

Fork `anuraghazra/github-readme-stats` and deploy the fork to Vercel using the Hobby plan.

Add the Vercel environment variable:

`PAT_1=<your-token>`

After deployment, replace every `YOUR-INSTANCE.vercel.app` occurrence in `README.md` with the deployed Vercel hostname.

## 4. Contribution snake

Repository Settings → Actions → General → Workflow permissions → choose **Read and write permissions** → Save.

This is the **repository's** Settings page, not account settings.

The workflow in `.github/workflows/snake.yml` runs every 12 hours, supports manual dispatch, and runs on pushes to `main`.

Wait for the first green Action run. It creates the `output` branch. The README snake URLs intentionally point to that branch.

## 5. Local banner regeneration

Install dependencies:

`python -m pip install -r generator/requirements.txt`

Put the source photo at:

`assets/photo.png`

Run:

`python generator/generate_banner.py`

The output is:

- `dark.svg`
- `light.svg`
- `generator/data/*.npy`

## 6. Final order

The README is intentionally assembled in this order:

1. theme-aware banner
2. GitHub stats
3. contribution snake
4. social badges

No second visual identity system is layered on top of the banner.
