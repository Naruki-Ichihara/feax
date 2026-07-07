"""Plot benchmark results (DOF sweep) from one or more results CSVs.

Produces a two-panel figure:
  * left  — mean wall time per call vs DOF (log-log), std as a shaded band
  * right — throughput (MDOF/s) vs DOF (log-x)

One line per (solver, mode, batch) series. Rows with no timing (build/OOM
failures) are dropped but reported to stdout.

Usage:
    python plot_bench.py results_direct_1M.csv
    python plot_bench.py results_direct_1M.csv --out cudss.png --title "cuDSS eager"
    python plot_bench.py a.csv b.csv --out combined.png
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(paths):
    """series[(solver, mode, batch)] = list of dict(dof, mean, std, thru, note)."""
    series = defaultdict(list)
    dropped = []
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                dof = _f(r.get("dof"))
                mean = _f(r.get("mean_s"))
                key = (r.get("solver", "?"), r.get("mode", "?"), int(_f(r.get("batch")) or 1))
                if dof is None or mean is None:
                    dropped.append((key, r.get("dof"), r.get("note", "")))
                    continue
                series[key].append(dict(
                    dof=dof, mean=mean, std=_f(r.get("std_s")) or 0.0,
                    thru=_f(r.get("throughput_dofps")), note=r.get("note", "")))
    for pts in series.values():
        pts.sort(key=lambda p: p["dof"])
    return series, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="results CSV path(s)")
    ap.add_argument("--out", default=None, help="output PNG (default: <first csv>.png)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    series, dropped = load(args.csv)
    if not series:
        print("No timed rows found in:", args.csv)
        return 1

    out = args.out or (os.path.splitext(args.csv[0])[0] + ".png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    for i, (key, pts) in enumerate(sorted(series.items())):
        solver, mode, batch = key
        label = f"{solver}/{mode}" + (f" (b{batch})" if batch > 1 else "")
        m = markers[i % len(markers)]
        x = [p["dof"] for p in pts]
        y = [p["mean"] for p in pts]
        lo = [max(1e-9, p["mean"] - p["std"]) for p in pts]
        hi = [p["mean"] + p["std"] for p in pts]
        line, = ax1.plot(x, y, marker=m, label=label)
        ax1.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15)

        xt = [p["dof"] for p in pts if p["thru"]]
        yt = [p["thru"] / 1e6 for p in pts if p["thru"]]
        if xt:
            ax2.plot(xt, yt, marker=m, label=label)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("DOF"); ax1.set_ylabel("mean wall time per call [s]")
    ax1.set_title("Solve time vs DOF")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel("DOF"); ax2.set_ylabel("throughput [MDOF/s]")
    ax2.set_title("Throughput vs DOF")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(fontsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")
    if dropped:
        print(f"Dropped {len(dropped)} untimed row(s):")
        for key, dof, note in dropped:
            print(f"  {key}  dof={dof}  note={note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
