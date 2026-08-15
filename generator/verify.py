from __future__ import annotations

from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "generator" / "data"


def evenness(mask: np.ndarray, groups: int = 60) -> float:
    h, w = mask.shape
    counts = []
    for i in range(groups):
        y0 = int(i * h / groups)
        y1 = int((i + 1) * h / groups)
        x0 = int((i * 37) % w)
        x1 = min(w, x0 + w // 2)
        counts.append(mask[y0:y1, x0:x1].sum())
    counts = np.asarray(counts, dtype=float)
    return float(counts.std() / max(counts.mean(), 1.0))


def svg_stats(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    return {
        "size_kb": path.stat().st_size / 1024,
        "circle_count": text.count("<circle "),
        "path_count": text.count("<path "),
        "animate_count": text.count("<animate "),
        "animate_transform_count": text.count("<animateTransform "),
    }


def main() -> None:
    for name in ["dark_mask.npy", "light_mask.npy", "portrait_points.npy", "logo_python_points.npy", "logo_react_points.npy", "logo_docker_points.npy"]:
        p = DATA / name
        a = np.load(p)
        print(f"{name}: shape={a.shape}, dtype={a.dtype}")

    for name in ["dark.svg", "light.svg"]:
        print(name, svg_stats(ROOT / name))

    print("dark intro evenness proxy:", round(evenness(np.load(DATA / "dark_mask.npy")), 4))
    print("light intro evenness proxy:", round(evenness(np.load(DATA / "light_mask.npy")), 4))


if __name__ == "__main__":
    main()
