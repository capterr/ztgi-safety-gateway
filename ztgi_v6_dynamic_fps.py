#!/usr/bin/env python3
"""
ZTGI V6 dynamic FPS demo.

Implements:
1) dF/dt = alpha * (F_target - F) - beta * F
2) M_FPS(t) = integral( (F · gradG) * (1/gamma) dt )
3) Reality collapse: selects one potential scenario based on FPS focus.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize01(v: Sequence[float]) -> List[float]:
    n = l2_norm(v)
    if n <= 1e-12:
        return [0.0 for _ in v]
    return [clamp01(x / n) for x in v]


@dataclass
class ZTGIV6DynamicFPS:
    dim: int = 5
    alpha: float = 1.3
    beta: float = 0.35
    dt: float = 0.1
    gamma_min: float = 0.6
    gamma_max: float = 2.0
    rng_seed: int = 42
    F: List[float] = field(default_factory=list)
    M: float = 0.0

    def __post_init__(self) -> None:
        random.seed(self.rng_seed)
        if not self.F:
            self.F = [0.0] * self.dim

    def step_focus(self, F_target: Sequence[float]) -> None:
        # Euler integration for dF/dt
        for i in range(self.dim):
            dF = self.alpha * (F_target[i] - self.F[i]) - self.beta * self.F[i]
            self.F[i] += self.dt * dF
        self.F = normalize01(self.F)

    def step_meaning(self, gradG: Sequence[float], gamma: float) -> float:
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        dM = (dot(self.F, gradG) / gamma) * self.dt
        self.M += dM
        return dM

    def collapse_reality(self, potentials: Sequence[Sequence[float]]) -> Tuple[int, float]:
        # Axiom-3 style toy collapse: scenario most aligned with current focus gets higher chance.
        scores = [max(0.0, dot(self.F, p)) + 1e-9 for p in potentials]
        total = sum(scores)
        probs = [s / total for s in scores]
        r = random.random()
        c = 0.0
        for idx, p in enumerate(probs):
            c += p
            if r <= c:
                return idx, p
        return len(probs) - 1, probs[-1]


def simulate(steps: int, seed: int) -> None:
    engine = ZTGIV6DynamicFPS(rng_seed=seed)

    # Channels: [price, volume, sentiment, orderbook, whale]
    potentials = [
        [0.9, 0.2, 0.4, 0.1, 0.7],
        [0.2, 0.9, 0.1, 0.6, 0.2],
        [0.4, 0.3, 0.8, 0.2, 0.1],
    ]

    print("step\tM_total\tdM\tF_norm\tchosen_scenario\tp")
    for t in range(steps):
        # Target focus drifts by regime, proving FPS is dynamic (not static).
        regime = (t // 20) % 3
        if regime == 0:
            F_target = [0.9, 0.3, 0.2, 0.2, 0.5]
        elif regime == 1:
            F_target = [0.2, 0.9, 0.4, 0.7, 0.3]
        else:
            F_target = [0.3, 0.2, 0.9, 0.2, 0.2]

        # Reality gradient (anomaly/change vector)
        gradG = [
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        ]

        # Context noise/friction -> gamma
        gamma = 1.0 + random.uniform(-0.2, 0.5)

        engine.step_focus(F_target)
        dM = engine.step_meaning(gradG, gamma)
        scenario, p = engine.collapse_reality(potentials)

        print(
            f"{t}\t{engine.M:.4f}\t{dM:.4f}\t{l2_norm(engine.F):.3f}\t"
            f"{scenario}\t{p:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ZTGI V6 dynamic FPS demo")
    parser.add_argument("--steps", type=int, default=60, help="number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    simulate(args.steps, args.seed)


if __name__ == "__main__":
    main()
