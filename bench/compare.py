"""Compare the OpenMP, MPI and JAX (threefry / rbg PRNG) scaling runs.

Reads results/{omp,mpi,jax,jax_rbg}_{strong,weak}.csv and *_meta.json (written by
bench/scaling.py) and writes
  results/compare.png   speedup vs serial C, strong efficiency, weak efficiency
  results/summary.md    Markdown table (pasted into README.md)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMPLS = [("omp", "OpenMP (C)", "#2a78d6"), ("mpi", "MPI (C)", "#eb6834"),
         ("jax", "JAX, threefry PRNG", "#1baf7a"), ("jax_rbg", "JAX, rbg PRNG", "#eda100")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(impl):
    def rows(kind):
        with open(RESULTS / f"{impl}_{kind}.csv") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rows("strong"), rows("weak"), json.loads((RESULTS / f"{impl}_meta.json").read_text())


def main():
    data = {impl: load(impl) for impl, _, _ in IMPLS}
    t_serial = statistics_median([m["serial_time_s"] for _, _, m in data.values()])
    meta = data["omp"][2]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURF)
    titles = ["Speedup vs serial C  (T_serial / T_p)", "Strong-scaling efficiency  T1 / (p T_p)",
              "Weak-scaling efficiency  T1 / T_p"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor(SURF)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("workers (threads / ranks / cores)")

    p_all = [int(r["workers"]) for r in data["omp"][0]]
    axes[0].plot(p_all, p_all, ls="--", lw=1.2, color=INK2, label="ideal (linear)")
    for ax in axes[1:]:
        ax.axhline(1.0, ls="--", lw=1.2, color=INK2, label="ideal")
        ax.set_ylim(0, 1.15)
    for ax in axes:
        ax.axvline(8.5, color=GRID, lw=6, zorder=0)
    axes[0].text(8.7, 0.6, "SMT region\n(>8 physical cores)", color=INK2, fontsize=8, va="bottom")

    end_labels = []
    for impl, label, color in IMPLS:
        strong, weak, _ = data[impl]
        p = [int(r["workers"]) for r in strong]
        series = [
            (axes[0], p, [t_serial / r["time_s"] for r in strong]),
            (axes[1], p, [r["efficiency"] for r in strong]),
            (axes[2], [int(r["workers"]) for r in weak], [r["efficiency"] for r in weak]),
        ]
        for ax, x, y in series:
            ax.plot(x, y, lw=2, color=color, marker="o", ms=6, markeredgecolor=SURF,
                    markeredgewidth=1.5, label=label)
        end_labels.append((series[0][2][-1], series[0][1][-1]))
    # End labels on the speedup panel, nudged apart so close values do not overlap.
    placed = []
    for y, x in sorted(end_labels):
        y_txt = max(y, placed[-1] + 0.85) if placed else y
        placed.append(y_txt)
        axes[0].annotate(f"{y:.1f}x", (x, y), xytext=(x + 0.4, y_txt), va="center",
                         color=INK, fontsize=9)
    for ax in axes:
        ax.set_xticks(p_all)
        ax.set_xlim(0, max(p_all) + 1.8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Monte Carlo VaR, top-200 KRX portfolio (real data): {int(meta['paths']):,} paths x "
                 f"{int(meta['steps'])} steps, serial C = {t_serial:.1f} s  |  Ryzen 7 7840HS, 8C/16T",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(RESULTS / "compare.png", dpi=150, facecolor=SURF)

    lines = [
        f"Real data, top-200 portfolio, {int(meta['paths']):,} paths x {int(meta['steps'])} steps "
        f"(strong) / {int(meta['weak_base']):,} paths per worker (weak); median of {int(meta['repeat'])} runs. "
        f"Serial C baseline: {t_serial:.2f} s.",
        "",
        "| Implementation | 1 worker | 8 workers | 16 workers | Speedup vs serial C (8 / 16) "
        "| Strong eff. (8 / 16) | Weak eff. (8 / 16) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for impl, label, _ in IMPLS:
        strong, weak, _ = data[impl]
        s = {int(r["workers"]): r for r in strong}
        w = {int(r["workers"]): r for r in weak}
        lines.append(
            f"| {label} | {s[1]['time_s']:.2f} s | {s[8]['time_s']:.2f} s | {s[16]['time_s']:.2f} s "
            f"| {t_serial / s[8]['time_s']:.1f}x / {t_serial / s[16]['time_s']:.1f}x "
            f"| {s[8]['efficiency']:.2f} / {s[16]['efficiency']:.2f} "
            f"| {w[8]['efficiency']:.2f} / {w[16]['efficiency']:.2f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote results/compare.png, results/summary.md")


def statistics_median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


if __name__ == "__main__":
    main()
