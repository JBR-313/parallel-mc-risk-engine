"""Strong- and weak-scaling benchmarks for the parallel simulators.

  python bench/scaling.py --impl omp                # OpenMP threads 1..16
  python bench/scaling.py --impl omp --quick        # small sizes, for a smoke test

Strong scaling: fixed problem (paths), vary workers -> speedup S(p) = T1/Tp,
                efficiency E(p) = S(p)/p.
Weak scaling  : paths = base * p                   -> efficiency E(p) = T1/Tp.
Each point is the median of --repeat runs. Results go to results/<impl>_*.csv
and a figure results/<impl>_scaling.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
PARAMS = ROOT / "data/stats/params.txt"


def run_once(impl: str, workers: int, paths: int, steps: int) -> dict:
    args = ["--paths", str(paths), "--steps", str(steps), "--params", str(PARAMS)]
    env = dict(os.environ)
    if impl == "omp":
        env.update(OMP_NUM_THREADS=str(workers), OMP_PROC_BIND="close", OMP_PLACES="cores"
                   if workers <= os.cpu_count() // 2 else "threads")
        cmd = [str(BUILD / "mc_omp"), *args]
    elif impl == "mpi":
        # 8 physical cores / 16 hardware threads: count hardware threads as slots
        cmd = ["mpirun", "-np", str(workers), "--use-hwthread-cpus", str(BUILD / "mc_mpi"), *args]
    elif impl == "serial":
        cmd = [str(BUILD / "mc_serial"), *args]
    else:
        raise ValueError(impl)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def measure(impl, workers, paths, steps, repeat):
    runs = [run_once(impl, workers, paths, steps) for _ in range(repeat)]
    t = statistics.median(r["elapsed"] for r in runs)
    return t, runs[0]


def plot(impl, strong, weak, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    s1, s2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=surf)
    p = [r["workers"] for r in strong]
    panels = [
        ("Strong scaling: speedup", p, [r["speedup"] for r in strong], p, "speedup  T1 / Tp"),
        ("Strong scaling: efficiency", p, [r["efficiency"] for r in strong], [1] * len(p), "efficiency  S(p) / p"),
        ("Weak scaling: efficiency", [r["workers"] for r in weak], [r["efficiency"] for r in weak],
         [1] * len(weak), "efficiency  T1 / Tp"),
    ]
    for ax, (title, x, y, ideal, ylabel) in zip(axes, panels):
        ax.set_facecolor(surf)
        ax.plot(x, ideal, ls="--", lw=1.2, color=ink2, label="ideal")
        ax.plot(x, y, lw=2, color=s1 if "Strong" in title else s2, marker="o", ms=6,
                markeredgecolor=surf, markeredgewidth=1.5, label="measured")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlabel("threads" if impl == "omp" else "processes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True, color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(-4, 8),
                    ha="right", color=ink, fontsize=9)
        if "efficiency" in title:
            ax.set_ylim(0, 1.15)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=surf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["omp", "mpi"], default="omp")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16])
    ap.add_argument("--paths", type=int, default=100_000, help="strong-scaling problem size")
    ap.add_argument("--weak-base", type=int, default=10_000, help="paths per worker (weak scaling)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.paths, a.weak_base, a.repeat = 10_000, 1_000, 1

    subprocess.run(["make", "-s", "-C", str(ROOT / "sim")], check=True)
    RESULTS.mkdir(exist_ok=True)

    t_serial, _ = measure("serial", 1, a.paths, a.steps, a.repeat)
    print(f"serial baseline: {t_serial:.3f}s for {a.paths:,} paths x {a.steps} steps")

    strong, t1 = [], None
    for p in a.workers:
        t, r = measure(a.impl, p, a.paths, a.steps, a.repeat)
        t1 = t1 or t
        row = {"workers": p, "paths": a.paths, "steps": a.steps, "time_s": t,
               "speedup": t1 / t, "efficiency": t1 / t / p, "speedup_vs_serial": t_serial / t,
               "VaR99": r["VaR99"]}
        strong.append(row)
        print(f"strong p={p:>2}  {t:7.3f}s  S={row['speedup']:5.2f}  E={row['efficiency']:.2f}")

    weak, t1 = [], None
    for p in a.workers:
        n = a.weak_base * p
        t, _ = measure(a.impl, p, n, a.steps, a.repeat)
        t1 = t1 or t
        weak.append({"workers": p, "paths": n, "steps": a.steps, "time_s": t, "efficiency": t1 / t})
        print(f"weak   p={p:>2}  {t:7.3f}s  E={t1 / t:.2f}  ({n:,} paths)")

    for name, rows in (("strong", strong), ("weak", weak)):
        with open(RESULTS / f"{a.impl}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    meta = {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
            "serial_time_s": t_serial, **vars(a)}
    (RESULTS / f"{a.impl}_meta.json").write_text(json.dumps(meta, indent=2))
    plot(a.impl, strong, weak, RESULTS / f"{a.impl}_scaling.png")
    print(f"wrote results/{a.impl}_strong.csv, {a.impl}_weak.csv, {a.impl}_scaling.png")


if __name__ == "__main__":
    main()
