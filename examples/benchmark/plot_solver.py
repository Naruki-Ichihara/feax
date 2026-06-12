"""Plot solver-throughput benchmark results (bench_solver.py).

Left: solve time vs DoFs (log-log, with O(N) reference).
Right: throughput (MDoF/s) vs DoFs. Solid = jit, dashed = eager.

Usage: python plot_solver.py [solver_results.csv] [-o solver_benchmark.png]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="solver_results.csv")
    p.add_argument("-o", "--out", default="solver_benchmark.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv).sort_values("n_dofs")
    fig, (ax_t, ax_tp) = plt.subplots(1, 2, figsize=(11, 4.5))

    style = {
        "jit": dict(color="tab:blue", marker="o", ls="-", label="jit"),
        "eager": dict(color="tab:red", marker="s", ls="--", label="eager"),
    }
    for mode, g in df.groupby("mode"):
        g = g.sort_values("n_dofs")
        ax_t.loglog(g["n_dofs"], g["time_s"], **style[mode])
        ax_tp.semilogx(g["n_dofs"], g["throughput_mdofs"], **style[mode])

    x0, x1 = df["n_dofs"].min(), df["n_dofs"].max()
    y0 = df[df["mode"] == "jit"]["time_s"].min()
    ax_t.loglog([x0, x1], [y0, y0 * x1 / x0], "k:", lw=1, label="O(N)")

    ax_t.set_xlabel("DoFs")
    ax_t.set_ylabel("solve time [s]")
    ax_t.set_title("full solve (assembly + cuDSS factorize + solve)")
    ax_t.legend()
    ax_t.grid(True, which="both", alpha=0.3)

    ax_tp.set_xlabel("DoFs")
    ax_tp.set_ylabel("throughput [MDoF/s]")
    ax_tp.set_title("solver throughput")
    ax_tp.grid(True, which="both", alpha=0.3)

    device = df["device"].iloc[0]
    dtype = df["dtype"].iloc[0]
    fig.suptitle(
        f"feax direct-solver throughput — linear elasticity cantilever ({device}, {dtype})"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
