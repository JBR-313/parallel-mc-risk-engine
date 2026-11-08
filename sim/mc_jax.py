"""JAX version of the VaR simulator (same model as the C code, JAX RNG).

Design
  * One jit-compiled function simulates a chunk of `chunk` paths: lax.fori_loop
    over the time steps, each step one (chunk x n) @ (n x n) matrix product
    z @ L^T, which XLA runs as a multithreaded GEMM. The C code instead does a
    per-path triangular mat-vec, so this is the "restructure into dense linear
    algebra" alternative to explicit threading.
  * Chunks have a fixed shape (the last one is padded and truncated) so the
    function is compiled exactly once; compile time is reported separately.
  * Memory is bounded by --mem-mb (only a few chunk x n arrays are live).
  * RNG: chunk c uses key fold_in(PRNGKey(seed), c), step s uses fold_in(., s),
    so results do not depend on the number of threads. With the default threefry
    PRNG, normal generation is ~80 % of a step and scales only ~1.8x on 16 cores
    (Amdahl); --prng rbg is 1.6x slower on one core but scales ~7x.
  * Parallelism: XLA's CPU thread pool. Restrict the cores with taskset
    (bench/scaling.py --impl jax does this).

  python sim/mc_jax.py --paths 100000 --steps 10 [--dtype float32]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "data/stats/params.h5"))
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--T", type=float, default=10 / 252)
    ap.add_argument("--value", type=float, default=1e8)
    ap.add_argument("--seed", type=int, default=410)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--mem-mb", type=float, default=64)
    ap.add_argument("--prng", choices=["threefry2x32", "rbg"], default="threefry2x32",
                    help="rbg: slower on one core but its normal generation parallelises "
                         "far better on CPU (threefry limits scaling to ~2.3x)")
    a = ap.parse_args()

    import jax
    if a.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_prng_impl", a.prng)
    import h5py
    import jax.numpy as jnp
    import numpy as np
    from jax import lax, random

    with h5py.File(a.params, "r") as h:
        mu, cov = h["mu"][()], h["cov"][()]
    n = len(mu)
    dt = a.T / a.steps
    dtype = jnp.dtype(a.dtype)
    L = jnp.asarray(np.linalg.cholesky(cov), dtype)
    drift = jnp.asarray((mu - 0.5 * np.diag(cov)) * dt, dtype)
    sdt = jnp.asarray(np.sqrt(dt), dtype)
    w = a.value / n
    chunk = int(min(a.paths, max(1_000, a.mem_mb * 2**20 / (3 * dtype.itemsize * n))))
    n_chunks = -(-a.paths // chunk)

    @jax.jit
    def chunk_losses(key):
        def step(s, logS):
            z = random.normal(random.fold_in(key, s), (chunk, n), dtype)
            return logS + drift + sdt * (z @ L.T)
        logS = lax.fori_loop(0, a.steps, step, jnp.zeros((chunk, n), dtype))
        return a.value - w * jnp.exp(logS).sum(axis=1)

    base = random.PRNGKey(a.seed)
    t0 = time.perf_counter()
    chunk_losses(base).block_until_ready()   # compile (and one discarded run)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [chunk_losses(random.fold_in(base, c)) for c in range(n_chunks)]
    loss = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])[: a.paths]
    elapsed = time.perf_counter() - t0

    def var_cvar(alpha):
        k = int(np.ceil(alpha * len(loss))) - 1
        v = np.partition(loss, k)[k]
        return float(v), float(loss[loss >= v].mean())

    v95, c95 = var_cvar(0.95)
    v99, c99 = var_cvar(0.99)
    print(json.dumps({
        "impl": f"jax-{a.dtype}-{a.prng}", "workers": len(os.sched_getaffinity(0)), "mode": "var",
        "paths": a.paths, "steps": a.steps, "assets": n, "chunk": chunk, "elapsed": elapsed,
        "compile": t_compile, "value": a.value, "mean_loss": float(loss.mean()),
        "VaR95": v95, "CVaR95": c95, "VaR99": v99, "CVaR99": c99,
    }))


if __name__ == "__main__":
    main()
