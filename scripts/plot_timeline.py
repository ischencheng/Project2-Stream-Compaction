"""Export/plot a compact CUDA timeline from an Nsight Systems SQLite capture.

Usage: python scripts/plot_timeline.py build/profiles/thrust.sqlite
Without a SQLite argument, redraw from the committed timeline CSV.
"""
import csv
import re
import sqlite3
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results" / "thrust-timeline.csv"
if len(sys.argv) == 2:
    db = sqlite3.connect(sys.argv[1])
    events = []
    for start, end, name in db.execute(
        "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
    ):
        if "Profiler" in name or "GetName" in name:
            continue
        events.append(("CUDA API", start, end, re.sub(r"_v\d+$", "", name), 0))
    for start, end, name in db.execute(
        "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.shortName ORDER BY k.start"
    ):
        events.append(("GPU kernels", start, end, name, 0))
    for start, end, size, name in db.execute(
        "SELECT m.start,m.end,m.bytes,e.label FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
        "JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id ORDER BY m.start"
    ):
        events.append(("GPU copies", start, end, name, size))
    origin = min(e[1] for e in events)
    with PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["lane", "start_ms", "duration_ms", "name", "bytes"])
        for lane, start, end, name, size in sorted(events, key=lambda e: e[1]):
            writer.writerow([lane, (start - origin) / 1e6, (end - start) / 1e6, name, size])

with PATH.open() as stream:
    rows = list(csv.DictReader(stream))

def color(name):
    if name in ("cudaMalloc", "cudaFree"):
        return "#E69F00"
    if "Synchronize" in name:
        return "#999999"
    if name == "cudaMemcpy" or "Device" in name and "to" in name:
        return "#009E73"
    if "Scan" in name:
        return "#CC79A7"
    return "#0072B2"

fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
lanes = {"CUDA API": 2, "GPU kernels": 1, "GPU copies": 0}
for row in rows:
    start, duration = float(row["start_ms"]), float(row["duration_ms"])
    lane = lanes[row["lane"]]
    ax.broken_barh([(start, duration)], (lane - 0.2, 0.4), facecolors=color(row["name"]))

annotations = {"static_kernel": "Vector initialization (2 kernels)",
               "DeviceScanKernel": "CUB scan",
               "Host-to-Device": "H2D: 4 MiB", "Device-to-Host": "D2H: 4 MiB"}
seen = set()
for row in rows:
    name = row["name"]
    if name in annotations and name not in seen:
        start, duration = float(row["start_ms"]), float(row["duration_ms"])
        y = lanes[row["lane"]]
        label = f"CUB scan: {duration * 1000:.1f} us" if name == "DeviceScanKernel" else annotations[name]
        ax.annotate(label, (start + duration / 2, y + 0.22),
                    xytext=(start + duration / 2, y + 0.48), ha="center", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": "#555555"})
        seen.add(name)
ax.set(yticks=list(lanes.values()), yticklabels=list(lanes), ylim=(-0.4, 2.5),
       xlabel="Milliseconds since first allocation after capture start",
       title="Nsight Systems: one warmed Thrust host call, 1,048,576 integers")
ax.grid(axis="x", alpha=0.2)
ax.legend(handles=[Patch(color=c, label=t) for c, t in [
    ("#E69F00", "Allocate/free"), ("#999999", "Synchronize"),
    ("#009E73", "Copy"), ("#0072B2", "Launch/event/fill"), ("#CC79A7", "Scan kernels")]],
    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=9)
fig.savefig(ROOT / "img" / "thrust-timeline.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote results/thrust-timeline.csv and img/thrust-timeline.png.")
