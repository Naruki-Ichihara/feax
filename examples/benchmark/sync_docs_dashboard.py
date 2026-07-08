#!/usr/bin/env python3
"""Sync bench.csv into the docs benchmark dashboard.

Reads ``bench.csv`` next to this script, converts it to the compact JSON the
interactive dashboard embeds, and rewrites the ``const DATA = [...]`` literal in
``docs/static/benchmarks/dashboard.html``. Run this after appending new
benchmark rows (e.g. via ``run_cudss_guarded.sh`` / ``run_krylov_guarded.sh``),
then rebuild the docs (``cd docs && npx docusaurus build``) or let the dev
server pick it up.

    python examples/benchmark/sync_docs_dashboard.py
"""
import csv
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "bench.csv")
DASH = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "static",
                                     "benchmarks", "dashboard.html"))

# Collapse the two physical H100 cards into one label (they carry complementary
# solver runs). Extend as new devices appear.
DEVICE_SHORT = {
    "NVIDIA GB10": "GB10",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 PCIe": "H100",
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
                "solver": r["solver"], "mode": r["mode"],
                "dof": int(r["dof"]), "batch": int(r["batch"]),
                "compile": _f(r, "compile_s"), "mean": _f(r, "mean_s"),
                "std": _f(r, "std_s"), "thru": _f(r, "throughput_dofps"),
                "amort": _f(r, "amortized_s"),
            })
    return out


def main():
    if not os.path.exists(CSV):
        sys.exit(f"bench.csv not found at {CSV}")
    if not os.path.exists(DASH):
        sys.exit(f"dashboard not found at {DASH}")

    rows = load_rows(CSV)
    data_js = json.dumps(rows, separators=(",", ":"))

    html = open(DASH).read()
    new_html, n = re.subn(r"const DATA = \[.*?\];",
                          "const DATA = " + data_js + ";", html,
                          count=1, flags=re.S)
    if n != 1:
        sys.exit("could not find `const DATA = [...]` in the dashboard")
    open(DASH, "w").write(new_html)

    dev = Counter(r["dev"] for r in rows)
    combo = Counter(r["solver"] + "/" + r["mode"] for r in rows)
    print(f"synced {len(rows)} rows -> {DASH}")
    print("  devices:", dict(dev))
    print("  solver/mode:", dict(sorted(combo.items())))
    print("Rebuild docs to publish: cd docs && npx docusaurus build")


if __name__ == "__main__":
    main()
