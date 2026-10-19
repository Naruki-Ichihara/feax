#!/usr/bin/env python3
"""Sync benchmark CSVs into the docs benchmark dashboard.

Reads every result file — the local, git-ignored ``bench.csv`` next to this
script plus the tracked ``results/*.csv`` — converts the rows to the compact
JSON the interactive dashboard embeds, and rewrites two literals in
``docs/static/benchmarks/dashboard.html``:

* ``const DATA = [...]``  — the rows;
* ``const MODES = [...]`` — the execution modes present in the data, in the
  canonical order below, so a new mode shows up in the controls without
  editing the HTML by hand (its label / dash / marker fall back to defaults
  unless ``MODE_LABEL`` etc. in the dashboard know it).

Run this after appending benchmark rows (e.g. via ``run_guarded.sh``), then
rebuild the docs (``cd docs && npm run build``) or let the dev server pick it up.

    python examples/benchmark/sync_docs_dashboard.py
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "bench.csv")
RESULTS = sorted(glob.glob(os.path.join(HERE, "results", "*.csv")))
MODE_ORDER = ["eager", "jit", "grad", "vmap-lhs", "vmap-rhs"]
DASH = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "static",
                                     "benchmarks", "dashboard.html"))

# Collapse the two physical H100 cards into one label (they carry complementary
# solver runs). Extend as new devices appear.
DEVICE_SHORT = {
    "NVIDIA GB10": "GB10",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 PCIe": "H100",
    "NVIDIA A100-SXM4-40GB": "A100",
    "NVIDIA RTX PRO 4500 Blackwell": "RTX 4500",
}


def _f(row, key):
    try:
        v = float(row[key])
        return None if v != v else v          # drop NaN
    except (TypeError, ValueError, KeyError):
        return None


def load_rows(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "dev": DEVICE_SHORT.get(r["device"], r["device"]),
                "plat": r.get("platform", "gpu"),
                "solver": r["solver"], "mode": r["mode"],
                "dof": int(r["dof"]), "batch": int(r["batch"]),
                "compile": _f(r, "compile_s"), "mean": _f(r, "mean_s"),
                "std": _f(r, "std_s"), "thru": _f(r, "throughput_dofps"),
                "amort": _f(r, "amortized_s"),
            })
    return out


def main():
    sources = ([CSV] if os.path.exists(CSV) else []) + RESULTS
    if not sources:
        sys.exit(f"no result CSVs: neither {CSV} nor results/*.csv exist")
    if not os.path.exists(DASH):
        sys.exit(f"dashboard not found at {DASH}")

    rows = []
    for src in sources:
        rows.extend(load_rows(src))
    data_js = json.dumps(rows, separators=(",", ":"))
    modes = sorted({r["mode"] for r in rows},
                   key=lambda m: (MODE_ORDER.index(m) if m in MODE_ORDER else 99, m))
    modes_js = json.dumps(modes)

    html = open(DASH).read()
    new_html, n = re.subn(r"const DATA = \[.*?\];",
                          "const DATA = " + data_js + ";", html,
                          count=1, flags=re.S)
    if n != 1:
        sys.exit("could not find `const DATA = [...]` in the dashboard")
    new_html, n = re.subn(r"const MODES = \[[^\]]*\];",
                          "const MODES = " + modes_js + ";", new_html, count=1)
    if n != 1:
        sys.exit("could not find `const MODES = [...]` in the dashboard")
    open(DASH, "w").write(new_html)
    print("sources:", ", ".join(os.path.relpath(s, HERE) for s in sources))
    print("modes:", modes)

    dev = Counter(r["dev"] for r in rows)
    combo = Counter(r["solver"] + "/" + r["mode"] for r in rows)
    print(f"synced {len(rows)} rows -> {DASH}")
    print("  devices:", dict(dev))
    print("  solver/mode:", dict(sorted(combo.items())))
    print("Rebuild docs to publish: cd docs && npx docusaurus build")


if __name__ == "__main__":
    main()
