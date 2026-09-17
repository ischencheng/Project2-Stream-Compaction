"""Regenerate report figures from measured CSV samples. Requires matplotlib."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "img"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def read_samples(name):
    groups = defaultdict(list)
    with (ROOT / "results" / name).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["algorithm"], int(row["n"]), int(row["block"])
            groups[key].append((float(row["algorithm_ms"]), float(row["wall_ms"])))
    return groups


def series(groups, algorithm, column=0):
    rows = sorted((n, samples) for (a, n, _), samples in groups.items() if a == algorithm)
    x = [n for n, _ in rows]
    med = [statistics.median(t[column] for t in s) for _, s in rows]
    lo = [sorted(t[column] for t in s)[len(s) // 4] for _, s in rows]
    hi = [sorted(t[column] for t in s)[3 * len(s) // 4] for _, s in rows]
    return x, med, lo, hi


def median(algorithm, n, column=0):
    values = next(values for (a, size, _), values in samples.items()
                  if a == algorithm and size == n)
    return statistics.median(value[column] for value in values)


def annotate(ax, point, label, position):
    ax.annotate(label, xy=point, xytext=position, textcoords="axes fraction",
                fontsize=9, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#777777", "alpha": 0.96},
                arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.1},
                zorder=5)


def curve(name, title, algorithms, column=0, notes=()):
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    for algorithm in algorithms:
        x, med, lo, hi = series(samples, algorithm, column)
        line, = ax.plot(x, med, marker="o", markersize=4, label=algorithm)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.13)
    ax.set(xscale="log", yscale="log", xlabel="Array size (integers)", ylabel="Median time (ms)", title=title)
    ax.grid(True, which="major", alpha=0.23)
    ax.legend(fontsize=10)
    for algorithm, n, label, position in notes:
        annotate(ax, (n, median(algorithm, n, column)), label, position)
    fig.text(0.5, -0.02, "RTX 2050 | Release | 3 warmups + 15 samples | Shading: interquartile range",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


samples = read_samples("benchmark.csv")
tuning = read_samples("tuning.csv")
large = 1048576
largest = max(n for _, n, _ in samples)
curve("scan-performance.png", "Exclusive scan: required implementations", ["CPU", "Naive", "Efficient", "Thrust"],
      notes=[("Efficient", large,
              f"At n = {large:,}\nGlobal tree: {median('Efficient', large) / median('CPU', large):.2f}x CPU time",
              (0.49, 0.86)),
             ("Thrust", largest,
              f"At n = {largest:,}\nThrust: {median('CPU', largest) / median('Thrust', largest):.2f}x CPU speedup\n(algorithm interval)",
              (0.80, 0.24))])
curve("scan-optimizations.png", "Scan optimizations: measured algorithm time",
      ["CPU", "Efficient unoptimized", "Efficient", "Shared naive", "Shared Blelloch", "Thrust"],
      notes=[("Efficient", large,
              f"Active-thread indexing: {median('Efficient unoptimized', large) / median('Efficient', large):.2f}x faster\nthan full-grid baseline at n = {large:,}",
              (0.50, 0.87)),
             ("Shared Blelloch", large,
              f"Shared Blelloch: {median('Efficient', large) / median('Shared Blelloch', large):.2f}x faster\nthan global tree at n = {large:,}",
              (0.78, 0.22))])
curve("compaction-performance.png", "Stable compaction: approximately 75% retained",
      ["CPU direct", "CPU scan", "Efficient compact", "Shared compact"],
      notes=[("Shared compact", large,
              f"At n = {large:,}\nShared compaction: {median('CPU direct', large) / median('Shared compact', large):.2f}x CPU direct speedup\n(algorithm interval; excludes transfers)",
              (0.73, 0.21))])
curve("scan-wall-time.png", "Scan: complete host API call including allocation and transfers",
      ["CPU", "Naive", "Efficient", "Shared naive", "Shared Blelloch", "Thrust"], column=1,
      notes=[("Thrust", largest,
              f"At n = {largest:,}\nThrust: {median('Thrust', largest, 1):.2f} ms; CPU: {median('CPU', largest, 1):.2f} ms\nTransfers and allocation change the ranking",
              (0.76, 0.24))])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for ax, n in zip(axes, [10000, 1048576]):
    for algorithm in ["Naive", "Efficient", "Efficient unoptimized", "Shared naive", "Shared Blelloch", "Shared unpadded"]:
        rows = sorted((b, statistics.median(t[0] for t in s)) for (a, size, b), s in tuning.items() if a == algorithm and size == n)
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=algorithm)
    ax.set(xscale="log", yscale="log", xlabel="Threads per block", ylabel="Median algorithm time (ms)", title=f"n = {n:,}")
    ax.set_xticks([32, 64, 128, 256, 512, 1024], [32, 64, 128, 256, 512, 1024])
    ax.grid(True, alpha=0.2)
for algorithm, position in [("Shared naive", (0.29, 0.48)), ("Shared Blelloch", (0.70, 0.34))]:
    rows = [(b, statistics.median(t[0] for t in values))
            for (a, n, b), values in tuning.items() if a == algorithm and n == large]
    block, time = min(rows, key=lambda row: row[1])
    annotate(axes[1], (block, time), f"{algorithm}: best at {block} threads\n{time:.4f} ms", position)
fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
fig.suptitle("Block-size sweep on RTX 2050 (3 warmups + 15 samples)")
fig.savefig(OUT / "block-size-tuning.png", dpi=180)
plt.close(fig)

with (ROOT / "results" / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["algorithm", "n", "block", "median_algorithm_ms", "q1_algorithm_ms", "q3_algorithm_ms", "median_wall_ms"])
    for (algorithm, n, block), values in sorted(samples.items()):
        algorithm_times = sorted(v[0] for v in values)
        writer.writerow([algorithm, n, block, statistics.median(algorithm_times),
                         algorithm_times[len(values) // 4], algorithm_times[3 * len(values) // 4],
                         statistics.median(v[1] for v in values)])
print("Wrote five report figures and results/summary.csv.")
