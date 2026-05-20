from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from .utils import fmt_seconds_compact


def plot_heatmap(
    data: dict,
    title: str,
    file: str,
    show: bool = True,
    key: str = "completes",
    zmin: int | None = 0,
    zmax: int | None = 8,
    zero_red: bool = True,
) -> None:
    x_axis = sorted(set(k[0] for k in data))
    y_axis = sorted(set(k[1] for k in data))

    z_raw = [
        [(data[(x, y)].get(key, None) if (x, y) in data else None) for x in x_axis]
        for y in y_axis
    ]
    if key == "first_completion_time":
        text = [
            [fmt_seconds_compact(float(v)) if v is not None else "" for v in row]
            for row in z_raw
        ]
    else:
        text = [[str(val) if val is not None else "" for val in row] for row in z_raw]

    z = np.array(
        [[v if v is not None else np.nan for v in row] for row in z_raw],
        dtype=float,
    )
    masked = np.ma.masked_invalid(z)

    vmin = zmin if zmin is not None else np.nanmin(z)
    vmax = zmax if zmax is not None else np.nanmax(z)

    if zero_red:
        blues = plt.cm.Blues(np.linspace(0.3, 1.0, 256))
        cmap = ListedColormap(np.vstack([[[1, 0, 0, 1]], blues]))
        boundaries = [vmin - 0.5, 0.5] + list(np.linspace(0.5, vmax, 256))
        norm = BoundaryNorm(boundaries, cmap.N)
    else:
        cmap = plt.cm.Blues
        norm = None

    cell_px = 50
    dpi = 150
    fig_w = max(len(x_axis) * cell_px / dpi + 1.5, 3)
    fig_h = max(len(y_axis) * cell_px / dpi + 1.5, 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(
        masked,
        aspect="equal",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        interpolation="nearest",
    )

    mid = (vmin + vmax) / 2
    for i, row in enumerate(text):
        for j, val in enumerate(row):
            if val:
                cell_val = z_raw[i][j]
                dark_bg = cell_val is not None and (
                    (zero_red and cell_val == 0) or cell_val > mid
                )
                color = "white" if dark_bg else "black"
                fs = 5 if key == "first_completion_time" else 6
                ax.text(j, i, val, ha="center", va="center", fontsize=fs, color=color)

    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis, rotation=90, fontsize=6)
    ax.set_yticks(range(len(y_axis)))
    ax.set_yticklabels(y_axis, fontsize=6)
    ax.set_title(title, fontsize=9)

    fig.savefig(file, dpi=dpi, bbox_inches="tight")
    if "all" in title and show:
        plt.show()
    plt.close(fig)
