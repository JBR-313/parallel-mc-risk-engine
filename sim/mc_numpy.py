"""Vectorised NumPy reference of the same model (independent RNG).

Used to cross-check the C simulators statistically and as the Python baseline
for the JAX version.

  python sim/mc_numpy.py --paths 1000000 --steps 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with h5py.File(path, "r") as h:
        return h["S0"][()], h["mu"][()], h["cov"][()]


def simulate_losses(S0, mu, cov, paths, steps, T, value, seed, mem_mb=64):
    """Memory use is bounded by mem_mb: paths are processed in chunks and the
    steps are accumulated one at a time, so only (chunk x n) arrays are live."""
    n = len(S0)
    L = np.linalg.cholesky(cov)
    dt = T / steps
    drift = (mu - 0.5 * np.diag(cov)) * dt
    w = value / n
    chunk = max(1_000, int(mem_mb * 2**20 / (3 * 8 * n)))  # z, z @ L.T, logS
    rng = np.random.default_rng(seed)
    out = np.empty(paths)
    for start in range(0, paths, chunk):
        m = min(chunk, paths - start)
        logS = np.zeros((m, n))
        for _ in range(steps):
            logS += drift + np.sqrt(dt) * (rng.standard_normal((m, n)) @ L.T)
        out[start:start + m] = value - w * np.exp(logS).sum(axis=1)
    return out


def var_cvar(loss, alpha):
    k = int(np.ceil(alpha * len(loss))) - 1
    v = np.partition(loss, k)[k]
    return v, loss[loss >= v].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "data/stats/params.h5"))
    ap.add_argument("--paths", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--T", type=float, default=10 / 252)
    ap.add_argument("--value", type=float, default=1e8)
    ap.add_argument("--seed", type=int, default=410)
    a = ap.parse_args()
    S0, mu, cov = load(Path(a.params))
    t0 = time.perf_counter()
    loss = simulate_losses(S0, mu, cov, a.paths, a.steps, a.T, a.value, a.seed)
    elapsed = time.perf_counter() - t0
    v95, c95 = var_cvar(loss, 0.95)
    v99, c99 = var_cvar(loss, 0.99)
    print(json.dumps({
        "impl": "numpy", "workers": 1, "mode": "var", "paths": a.paths, "steps": a.steps,
        "assets": len(S0), "elapsed": elapsed, "value": a.value, "mean_loss": loss.mean(),
        "VaR95": v95, "CVaR95": c95, "VaR99": v99, "CVaR99": c99,
    }))


if __name__ == "__main__":
    main()
