"""Draw local-rule to population-phenomenon mechanism map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_BASE = Path(__file__).with_name("local_to_global_mechanism_map")


def add_box(ax, xy, text, fc, ec="#333333", width=0.24, height=0.095, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        fc=fc,
        ec=ec,
        lw=0.9,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True, zorder=4)


def add_arrow(ax, start, end, color="#555555", style="-", label=None, rad=0.0):
    sx, sy = start
    ex, ey = end
    if sx < ex:
        start = (sx + 0.13, sy)
        end = (ex - 0.14, ey)
    elif sx > ex:
        start = (sx - 0.13, sy)
        end = (ex + 0.14, ey)
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.4,
        color=color,
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="center", fontsize=7, color=color)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.16, 0.96, "Local rules", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.96, "Intermediate observables", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.84, 0.96, "Population phenomena", ha="center", va="center", fontsize=14, fontweight="bold")

    local = {
        "history": (0.16, 0.84, "Self-history\nrefractoriness / memory"),
        "local": (0.16, 0.70, "Local coupling\nneighbor kernels"),
        "latent": (0.16, 0.56, "Common latent / behavior\nstate drive"),
        "rho": (0.16, 0.42, "Near-critical normalized A\nrho close to 1"),
        "sym": (0.16, 0.28, "Symmetry / asymmetry\nfeedback structure"),
        "ei": (0.16, 0.14, "E/I delays and inhibition\nphase feedback"),
    }
    mid = {
        "glm": (0.50, 0.78, "Nested GLM gains\nhistory/local/global"),
        "corr": (0.50, 0.64, "Auto/cross correlation\nand population coupling"),
        "lyap": (0.50, 0.50, "Lyapunov covariance\nSigma = A Sigma A^T + Q"),
        "dmd": (0.50, 0.36, "DMD eigenmodes\nreal vs complex modes"),
        "avalanche": (0.50, 0.22, "Avalanche statistics\nbranching and tails"),
        "energy": (0.50, 0.08, "Information / activity /\nwiring-cost proxy"),
    }
    global_nodes = {
        "h1": (0.84, 0.80, "H1 supported:\nhistory + local coupling"),
        "h2": (0.84, 0.62, "H2 supported:\nlong-tail eigenspectrum"),
        "h3": (0.84, 0.44, "H3 weak:\nPSD/PLV without\ncomplex/reset evidence"),
        "h4": (0.84, 0.26, "H4 supported:\ncritical branching proxy"),
        "h5": (0.84, 0.08, "H5 supported:\nsparse + few long-range\nnear-stable optimum"),
    }

    for x, y, text in local.values():
        add_box(ax, (x, y), text, "#dbe9f6")
    for x, y, text in mid.values():
        add_box(ax, (x, y), text, "#e6f4df")
    for key, (x, y, text) in global_nodes.items():
        color = "#f8e0c6" if key == "h3" else "#d9f0d3"
        add_box(ax, (x, y), text, color, width=0.26, height=0.115)

    solid = "#3b6ea8"
    weak = "#d95f02"
    add_arrow(ax, local["history"][:2], mid["glm"][:2], solid)
    add_arrow(ax, local["local"][:2], mid["glm"][:2], solid)
    add_arrow(ax, local["latent"][:2], mid["glm"][:2], "#7f7f7f", style="--", label="confound/control")
    add_arrow(ax, mid["glm"][:2], global_nodes["h1"][:2], solid)
    add_arrow(ax, mid["corr"][:2], global_nodes["h1"][:2], solid, rad=0.08)

    add_arrow(ax, local["rho"][:2], mid["lyap"][:2], solid)
    add_arrow(ax, local["sym"][:2], mid["lyap"][:2], solid)
    add_arrow(ax, mid["lyap"][:2], global_nodes["h2"][:2], solid)
    add_arrow(ax, mid["dmd"][:2], global_nodes["h2"][:2], solid, rad=-0.08)

    add_arrow(ax, local["ei"][:2], mid["dmd"][:2], weak)
    add_arrow(ax, mid["dmd"][:2], global_nodes["h3"][:2], weak, label="missing near-unit complex")
    add_arrow(ax, mid["corr"][:2], global_nodes["h3"][:2], weak, style="--", label="not sufficient")

    add_arrow(ax, local["rho"][:2], mid["avalanche"][:2], solid, rad=0.12)
    add_arrow(ax, mid["avalanche"][:2], global_nodes["h4"][:2], solid)
    add_arrow(ax, mid["energy"][:2], global_nodes["h5"][:2], solid)
    add_arrow(ax, local["local"][:2], mid["energy"][:2], solid, rad=0.12)

    ax.text(
        0.50,
        0.015,
        "Solid blue: implemented evidence path. Orange: partial/weak path. Dashed: control or insufficiency warning.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
