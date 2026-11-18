"""Correctness checks for the simulators. Exit code 0 = all checks passed.

1. Option mode: every single-asset European call is within 4 standard errors
   of Black-Scholes (one-step and weekly-step discretisation, 3 strikes).
   Run on the first 8 portfolio assets to keep it fast.
2. Single-asset VaR matches the closed-form lognormal quantile.
3. OpenMP == serial and MPI == serial on the full portfolio: identical VaR/CVaR
   (same paths), mean loss equal up to summation order.
4. C vs NumPy, C vs JAX and C vs JAX-rbg (independent RNGs and implementations): VaR99 agrees within 3 %.
5. JAX on 1 core == JAX on all cores (up to GEMM summation order).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PARAMS = ROOT / "data/stats/params.txt"
failures = 0


def run(cmd, env=None):
    out = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=True,
                         env={**os.environ, **(env or {})})
    return json.loads(out.stdout.strip().splitlines()[-1])


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def subset_params(k: int) -> Path:
    lines = PARAMS.read_text().split("\n")
    n = int(lines[0])
    head = lines[1:1 + k]
    cov = [" ".join(lines[1 + n + i].split()[:k]) for i in range(k)]
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("\n".join([str(k), *head, *cov]) + "\n")
    f.close()
    return Path(f.name)


def main():
    subprocess.run(["make", "-s", "-C", str(ROOT / "sim")], check=True)
    n_assets = int(PARAMS.read_text().split("\n", 1)[0])
    print(f"portfolio: {n_assets} assets")

    sub = subset_params(min(8, n_assets))
    for steps in (1, 52):
        for K in (0.9, 1.0, 1.1):
            r = run([BUILD / "mc_omp", "--mode", "option", "--paths", 400_000, "--steps", steps,
                     "--T", 1, "--K", K, "--params", sub])
            worst = max(abs(o["mc"] - o["bs"]) / o["se"] for o in r["options"] if "bs" in o)
            check(f"Black-Scholes steps={steps:<3} K={K}", worst < 4, f"max |MC-BS|/SE = {worst:.2f}")
    sub.unlink()

    # Single asset: loss_alpha = V (1 - exp((mu - s^2/2) T + s sqrt(T) z_{1-alpha}))
    S0, mu, sig, T, V = 50_000.0, 0.08, 0.3, 10 / 252, 1e8
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"1\nTEST {S0} {mu}\n{sig**2}\n")
    r = run([BUILD / "mc_omp", "--paths", 2_000_000, "--steps", 1, "--T", T, "--value", V,
             "--params", f.name])
    for a in (0.95, 0.99):
        exact = V * (1 - math.exp((mu - sig**2 / 2) * T + sig * math.sqrt(T) * NormalDist().inv_cdf(1 - a)))
        got = r[f"VaR{round(a * 100)}"]
        check(f"single-asset VaR{round(a * 100)} closed form", abs(got / exact - 1) < 0.01,
              f"mc={got:,.0f} exact={exact:,.0f}")
    Path(f.name).unlink()

    paths = 20_000
    s = run([BUILD / "mc_serial", "--paths", paths, "--params", PARAMS])

    def same(o):
        exact = all(o[k] == s[k] for k in ("VaR95", "CVaR95", "VaR99", "CVaR99"))
        return exact and abs(o["mean_loss"] - s["mean_loss"]) <= 1e-9 * abs(s["mean_loss"]) + 1e-6

    for th in (2, 7, 16):
        o = run([BUILD / "mc_omp", "--paths", paths, "--params", PARAMS], {"OMP_NUM_THREADS": str(th)})
        check(f"OpenMP({th:>2} threads) == serial", same(o))
    for np_ in (2, 3, 8):
        o = run(["mpirun", "-np", np_, "--oversubscribe", BUILD / "mc_mpi", "--paths", paths,
                 "--params", PARAMS])
        check(f"MPI({np_:>2} ranks)    == serial", same(o))

    c = run([BUILD / "mc_omp", "--paths", 200_000, "--params", PARAMS])
    npy = run([sys.executable, ROOT / "sim/mc_numpy.py", "--paths", 200_000])
    rel = abs(npy["VaR99"] / c["VaR99"] - 1)
    check("C vs NumPy VaR99 (independent RNG)", rel < 0.03,
          f"C={c['VaR99']:,.0f} NumPy={npy['VaR99']:,.0f} rel diff={rel:.4f}")

    jax_cmd = [sys.executable, ROOT / "sim/mc_jax.py", "--paths", 200_000]
    jx = run(jax_cmd)
    rel = abs(jx["VaR99"] / c["VaR99"] - 1)
    check("C vs JAX VaR99 (independent RNG)", rel < 0.03,
          f"C={c['VaR99']:,.0f} JAX={jx['VaR99']:,.0f} rel diff={rel:.4f}")
    j1 = run(["taskset", "-c", "0", *jax_cmd])
    rel = max(abs(j1[k] / jx[k] - 1) for k in ("VaR95", "CVaR95", "VaR99", "CVaR99"))
    check("JAX(1 core) == JAX(all cores)", rel < 1e-9, f"max rel diff={rel:.1e}")

    jr = run([*jax_cmd, "--prng", "rbg"])
    rel = abs(jr["VaR99"] / c["VaR99"] - 1)
    check("C vs JAX(rbg PRNG) VaR99", rel < 0.03,
          f"C={c['VaR99']:,.0f} JAX-rbg={jr['VaR99']:,.0f} rel diff={rel:.4f}")

    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
