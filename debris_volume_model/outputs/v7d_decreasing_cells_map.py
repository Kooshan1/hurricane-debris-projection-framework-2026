"""
Map the spatial distribution of decreasing-volume cells (FEMA36 storm, the most
representative scenario) and color them by their net 2020->2040 delta.
"""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PRED_DIR = Path("outputs/final_debris_volume_output")
OUT_DIR = Path("debris_volume_model/outputs/v7d_decreasing_cells")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_map(storm: str):
    p20 = gpd.read_file(PRED_DIR / storm / "2019" / "V14_physics_v7d_predictions.shp")
    p40 = gpd.read_file(PRED_DIR / storm / "2040" / "V14_physics_v7d_predictions.shp")
    p20 = p20[["FID", "geometry", "pred_m3"]].rename(columns={"pred_m3": "pred_2020"})
    p40 = p40[["FID", "pred_m3"]].rename(columns={"pred_m3": "pred_2040"})
    g = p20.merge(p40, on="FID", how="inner")
    g["d_total"] = g["pred_2040"] - g["pred_2020"]
    g["active"] = g["pred_2020"] > 1

    # Project for nicer maps
    if str(g.crs).lower() != "epsg:3857":
        g = g.to_crs("EPSG:3857")

    # Render the map: only ACTIVE cells, colored by d_total
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none"); ax.axis("off")
    ax.set_aspect("equal")

    inactive = g[~g["active"]]
    inactive.plot(ax=ax, color="#eeeeee", edgecolor="none", linewidth=0,
                  zorder=1, alpha=0.6)

    active = g[g["active"]].copy()
    vmax = float(np.percentile(np.abs(active["d_total"]), 95))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    active.plot(ax=ax, column="d_total", cmap="RdBu", norm=norm,
                edgecolor="none", linewidth=0, zorder=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                       pad=0.02, extend="both")
    cb.set_label(f"v7d ΔDebris (2040 − 2020) [m³ / 250 m cell]   |   "
                 f"{storm.upper()}, active baseline cells only",
                 size=11)
    cb.ax.tick_params(labelsize=9)

    ax.set_title("")
    out = OUT_DIR / f"map_decreasing_cells_{storm}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)
    print(f"  wrote {out}")

    # Also write the GeoDataFrame so it can be inspected in QGIS
    out_shp = OUT_DIR / f"per_cell_deltas_{storm}.shp"
    active[["FID", "pred_2020", "pred_2040", "d_total", "geometry"]].to_file(out_shp)
    print(f"  wrote {out_shp}")


if __name__ == "__main__":
    for s in ["ike", "fema33", "fema36"]:
        print(f"\n=== {s} ===")
        make_map(s)
