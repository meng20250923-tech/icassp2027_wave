"""Procedural vessel phantoms for independent PAT train/test examples."""

from __future__ import annotations

import numpy as np


def random_vessel_phantom(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create a normalized random vessel network inside the reconstruction FOV."""
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    image = np.zeros((n, n), dtype=np.float64)
    center = (n - 1) / 2

    def point() -> tuple[float, float]:
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.05 * n, 0.32 * n)
        return center + radius * np.cos(angle), center + radius * np.sin(angle)

    def tube(start: tuple[float, float], end: tuple[float, float], width: float, value: float) -> None:
        x0, y0 = start
        x1, y1 = end
        vx, vy = x1 - x0, y1 - y0
        fraction = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        distance = np.hypot(xx - (x0 + fraction * vx), yy - (y0 + fraction * vy))
        image[:] += value * np.exp(-(distance / width) ** 2)

    junctions = [point() for _ in range(int(rng.integers(2, 5)))]
    for _ in range(int(rng.integers(4, 8))):
        start = junctions[int(rng.integers(len(junctions)))]
        end = point()
        tube(start, end, rng.uniform(0.9, 2.0), rng.uniform(0.45, 1.0))
    for x0, y0 in junctions:
        radius = rng.uniform(1.8, 3.8)
        image += rng.uniform(0.25, 0.7) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * radius**2))
    return image / (image.max() + 1e-9)
