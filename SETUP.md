# Profile Setup

1. Replace the root `dark.svg`, `light.svg`, and `README.md` in `lucifer9973/lucifer9973`.
2. Keep `.github/workflows/snake.yml` in the repository.
3. In the profile repository: Settings → Actions → General → Workflow permissions → Read and write permissions.
4. Run `Generate Snake Animation` once from Actions. It creates the `output` branch.
5. In the separate GitHub Readme Stats Vercel project, add `PAT_1` with your GitHub classic PAT and redeploy if necessary.
6. The README already points to the current Vercel instance:
   `https://github-readme-stats-lovat-six-21.vercel.app`

The generator source is in `generator/generate_profile_banner.py`; `.npy` files preserve the generated point data so the SVGs can be reproduced without rebuilding the point extraction from scratch.
