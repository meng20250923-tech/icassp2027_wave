"""In-distribution and structural-OOD initial-pressure phantom families."""
from __future__ import annotations

import numpy as np

from .phantoms import random_vessel_phantom


def dense_branching_vessel_phantom(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create an OOD vessel tree with finer, denser, hierarchical branches.

    Unlike the in-distribution random-junction generator, this family grows
    from one or two roots, uses more branches, and spans a larger radial FOV.
    It is used only for held-out structural generalization tests.
    """
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    image = np.zeros((n, n), dtype=np.float64)
    center = (n - 1) / 2

    def tube(start: tuple[float, float], end: tuple[float, float], width: float, value: float) -> None:
        x0, y0 = start
        x1, y1 = end
        vx, vy = x1 - x0, y1 - y0
        fraction = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        distance = np.hypot(xx - (x0 + fraction * vx), yy - (y0 + fraction * vy))
        image[:] += value * np.exp(-(distance / width) ** 2)

    root_angle = rng.uniform(0, 2 * np.pi)
    root_radius = rng.uniform(0.02 * n, 0.10 * n)
    root = (center + root_radius * np.cos(root_angle), center + root_radius * np.sin(root_angle))
    frontier = [(root, root_angle + rng.uniform(-np.pi, np.pi), 1.35, 0)]
    for _ in range(int(rng.integers(10, 16))):
        if not frontier:
            break
        start, direction, width, depth = frontier.pop(int(rng.integers(len(frontier))))
        direction += rng.normal(scale=0.35 + 0.08 * depth)
        length = rng.uniform(0.10 * n, 0.22 * n) * (0.9**depth)
        end = (start[0] + length * np.cos(direction), start[1] + length * np.sin(direction))
        if not (0.10 * n < end[0] < 0.90 * n and 0.10 * n < end[1] < 0.90 * n):
            continue
        tube(start, end, width, rng.uniform(0.35, 0.85))
        if depth < 3:
            for split in (-1, 1):
                if rng.random() < 0.82:
                    frontier.append((end, direction + split * rng.uniform(0.35, 0.85), width * rng.uniform(0.58, 0.78), depth + 1))
    image += 0.35 * np.exp(-((xx - root[0]) ** 2 + (yy - root[1]) ** 2) / (2 * rng.uniform(1.3, 2.4) ** 2))
    return image / (image.max() + 1e-9)


def sample_phantom(n: int, rng: np.random.Generator, family: str = "id") -> np.ndarray:
    """Sample an initial-pressure image from an explicitly named family."""
    if family == "id":
        return random_vessel_phantom(n, rng)
    if family == "ood":
        return dense_branching_vessel_phantom(n, rng)
    raise ValueError("phantom family must be 'id' or 'ood'")
