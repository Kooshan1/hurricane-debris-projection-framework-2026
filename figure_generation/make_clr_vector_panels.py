"""
Vector merged panels for the supplementary CLR uncertainty maps
(S2 mean, S3 std, S4 P(CLR>0.30), S5 P(CLR>0.50)).

Redesign rationale
------------------
`make_merged_map_panel` in _style.py rendered each map to a PNG, PIL-rotated
the bitmap, then imshow'd the bitmaps into the grid -- three resampling steps
that softened the thin tract boundaries -- and the colorbar was a SEPARATE PNG
rendered in matplotlib's default font (DejaVu, not STIX Two). This module
instead builds ONE matplotlib figure per metric: the tract/county geometry is
rotated once (so it stays VECTOR), all nine choropleths are drawn as vector
subplots, and a single native colorbar is added -- everything under STIX Two.
The resulting PDF is fully vector (crisp boundaries at any zoom), small, and
font-consistent. Colormaps, the shared PowerNorm year-contrast, and the
moderate legend size are unchanged from 02_make_uncertainty_maps.py.

Outputs: REVISION_FIG_ROOT/S-F2_vector/{S-F2_CLR_mean,S-F2_CLR_std,
         S-F2_CLR_pexceed_030,S-F2_CLR_pexceed_050}.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DISPLAY_YEARS, HURRICANES, HURRICANE_LABELS,
                    YEAR_DISPLAY_TO_FILE, MC_DIR_700, REVISION_FIG_ROOT,
                    FIG_WIDTH_DOUBLE, FONT_SIZE_SMALL, MAP_ROTATION_DEG,
                    load_county_gdf, load_tract_gdf, boundary_extent,
                    register_custom_cmaps, ensure_font, MidpointNormalize)

DERIVED_DIR = MC_DIR_700 / "derived_uncertainty"
OUT_DIR = REVISION_FIG_ROOT / "S-F2_vector"
SENS_DIR = REVISION_FIG_ROOT / "S-F8_sensitivity"   # holds sensitivity_data_{storm}.csv
SENS_VLIM = 1.0
RASTER_DPI = 800    # DPI for the non-vector (raster) twin PDFs

CLR_NORM_GAMMA = 0.6     # PowerNorm gamma — expands the skewed low-CLR range
CLR_VMAX_PCTL = 97.5     # cap vmax at this pooled percentile (vs global max ~0.96)

# (out_stem, value_col, cmap_name, colorbar_label) -- original restored cmaps
METRICS = [
    ("S-F2_CLR_mean",        "mean",          "baseline_reds_clr",     "Mean CLR"),
    ("S-F2_CLR_std",         "std",           "uncertainty_purples",   "Std. dev. of CLR"),
    ("S-F2_CLR_pexceed_030", "p_exceed_0.30", "baseline_orrd_network", "P(CLR > 0.30)"),
    ("S-F2_CLR_pexceed_050", "p_exceed_0.50", "baseline_orrd_network", "P(CLR > 0.50)"),
]


def pooled_vmax(value_col, pctl=CLR_VMAX_PCTL):
    vals = []
    for storm in HURRICANES:
        for dy in DISPLAY_YEARS:
            fy = YEAR_DISPLAY_TO_FILE[dy]
            csv = DERIVED_DIR / f"clr_metrics_{storm}_{fy}.csv"
            if csv.is_file():
                vals.append(pd.read_csv(csv)[value_col].dropna().to_numpy())
    if not vals:
        return 1.0
    v = float(np.percentile(np.concatenate(vals), pctl))
    return v if (np.isfinite(v) and v > 0) else 1.0


def _save_raster_pdf(png_path, pdf_path, dpi):
    """Embed a high-DPI PNG into a single-page raster PDF at the correct physical
    size -- the non-vector twin of the same figure (white background)."""
    import fitz
    pix = fitz.Pixmap(str(png_path))
    w_pt, h_pt = pix.width / dpi * 72.0, pix.height / dpi * 72.0
    doc = fitz.open()
    page = doc.new_page(width=w_pt, height=h_pt)
    page.insert_image(fitz.Rect(0, 0, w_pt, h_pt), filename=str(png_path))
    doc.save(str(pdf_path), deflate=True)
    doc.close()


def _save_outputs(fig, out_pdf):
    """Save the SAME figure three ways: a vector PDF, a high-DPI raster PNG, and
    a non-vector (raster) PDF twin (<stem>_raster.pdf)."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(out_pdf, transparent=True, bbox_inches="tight", pad_inches=0.02)
    png = out_pdf.with_suffix(".png")
    fig.savefig(png, dpi=RASTER_DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    _save_raster_pdf(png, out_pdf.with_name(out_pdf.stem + "_raster.pdf"), RASTER_DPI)
    plt.close(fig)


def make_panel(tract_rot, county_rot, extent, value_col, cmap_name, label, out_pdf):
    register_custom_cmaps()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#dddddd", alpha=1.0)
    vmax = pooled_vmax(value_col)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)   # LINEAR colorbar (uniform ticks)

    minx, miny, maxx, maxy = extent
    aspect = (maxy - miny) / max(maxx - minx, 1e-9)
    ncols, nrows = 3, 3
    cell_w = FIG_WIDTH_DOUBLE / ncols
    cell_h = cell_w * aspect
    cbar_band_in = 0.50                       # reserved bottom band for the colorbar
    fig_h = nrows * cell_h + cbar_band_in

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_DOUBLE, fig_h))
    fig.patch.set_alpha(0.0)
    # Negative wspace overlaps the empty (transparent) corners of the rotated
    # maps so the panels pack tightly and the maps render LARGER; a small hspace
    # leaves only enough room for the one-line panel titles.
    fig.subplots_adjust(left=0.002, right=0.998, top=0.988,
                        bottom=(cbar_band_in / fig_h) + 0.004,
                        wspace=-0.30, hspace=0.04)

    for r, storm in enumerate(HURRICANES):
        for c, dy in enumerate(DISPLAY_YEARS):
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            fy = YEAR_DISPLAY_TO_FILE[dy]
            csv = DERIVED_DIR / f"clr_metrics_{storm}_{fy}.csv"
            if csv.is_file():
                df = (pd.read_csv(csv)[["FID", value_col]]
                      .rename(columns={value_col: "value"}))
                df["FID"] = df["FID"].astype(np.int64)
                g = tract_rot.merge(df, on="FID", how="left")
                g.plot(column="value", cmap=cmap, norm=norm, ax=ax,
                       linewidth=0.015, edgecolor="black", legend=False,
                       missing_kwds={"color": "#dddddd", "edgecolor": "black",
                                     "linewidth": 0.015})
            county_rot.plot(ax=ax, facecolor="none", edgecolor="black",
                            linewidth=0.025)
            ax.set_title(f"{HURRICANE_LABELS[storm]} {dy}",
                         fontsize=FONT_SIZE_SMALL, pad=2.5)

    # One shared, native, horizontal colorbar (vector text -> STIX Two).
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    bar_w_frac = 1.8 / FIG_WIDTH_DOUBLE        # ~1.8 in long (was ~2.8 in)
    bar_h_frac = 0.08 / fig_h                  # ~0.08 in thick (was ~0.17 in)
    cax = fig.add_axes([(1.0 - bar_w_frac) / 2.0, 0.30 / fig_h, bar_w_frac, bar_h_frac])
    ticks = np.round(np.linspace(0.0, vmax, 5), 2)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=FONT_SIZE_SMALL, labelpad=1.5)
    cb.ax.tick_params(labelsize=FONT_SIZE_SMALL, length=1.8, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    try:
        cb.solids.set_edgecolor("face")
    except Exception:
        pass

    _save_outputs(fig, out_pdf)
    print(f"  wrote {out_pdf.name} (+ _raster.pdf)  (vmax={vmax:.3f})")


def make_sensitivity_panel(tract_rot, county_rot, extent, out_pdf):
    """S8: per-tract elasticity dCLR/dSurge (2020->2040), 1x3 vector panel.

    Reads the precomputed sensitivity_data_{storm}.csv (FID, sensitivity);
    tracts with |dSD|<0.02 m have blank sensitivity and are shown light grey.
    """
    register_custom_cmaps()
    cmap = plt.get_cmap("change_rdbu_9_r")
    # Symmetric range -> identical to MidpointNormalize(-1,1,0) but avoids that
    # custom norm's int64 fill bug inside the colorbar's minor-tick locator.
    norm = mcolors.Normalize(vmin=-SENS_VLIM, vmax=SENS_VLIM)

    minx, miny, maxx, maxy = extent
    aspect = (maxy - miny) / max(maxx - minx, 1e-9)
    ncols = 3
    cell_w = FIG_WIDTH_DOUBLE / ncols
    cell_h = cell_w * aspect
    cbar_band_in = 0.5
    fig_h = cell_h + cbar_band_in

    fig, axes = plt.subplots(1, ncols, figsize=(FIG_WIDTH_DOUBLE, fig_h))
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.965,
                        bottom=(cbar_band_in / fig_h) + 0.004, wspace=-0.30)

    for c, storm in enumerate(HURRICANES):
        ax = axes[c]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        tract_rot.plot(ax=ax, facecolor="#e0e0e0", edgecolor="black",
                       linewidth=0.015, zorder=0)
        csv = SENS_DIR / f"sensitivity_data_{storm}.csv"
        if csv.is_file():
            df = (pd.read_csv(csv)[["FID", "sensitivity"]]
                  .rename(columns={"sensitivity": "value"}))
            df["FID"] = df["FID"].astype(np.int64)
            g = tract_rot.merge(df, on="FID", how="left")
            valid = g[g["value"].notna()]
            if not valid.empty:
                valid.plot(column="value", cmap=cmap, norm=norm, ax=ax,
                           linewidth=0.02, edgecolor="black", legend=False, zorder=1)
        county_rot.plot(ax=ax, facecolor="none", edgecolor="black",
                        linewidth=0.025, zorder=10)
        ax.set_title(HURRICANE_LABELS[storm], fontsize=FONT_SIZE_SMALL, pad=2.5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    bar_w_frac = 1.8 / FIG_WIDTH_DOUBLE
    bar_h_frac = 0.08 / fig_h
    cax = fig.add_axes([(1.0 - bar_w_frac) / 2.0, 0.30 / fig_h, bar_w_frac, bar_h_frac])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.set_label(r"$\Delta$CLR / $\Delta$Surge (1/m)", fontsize=FONT_SIZE_SMALL,
                 labelpad=2)
    cb.ax.tick_params(labelsize=FONT_SIZE_SMALL, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    try:
        cb.solids.set_edgecolor("face")
    except Exception:
        pass

    _save_outputs(fig, out_pdf)
    print(f"  wrote {out_pdf.name} (+ _raster.pdf)")


def main():
    ensure_font()              # STIX Two for ALL text (titles + colorbar)
    register_custom_cmaps()
    tract = load_tract_gdf()
    county = load_county_gdf()

    # Rotate the geometry once (clockwise MAP_ROTATION_DEG) about one shared
    # origin so tracts and county stay aligned; everything remains vector.
    try:
        origin = county.geometry.union_all().centroid
    except Exception:
        origin = county.geometry.unary_union.centroid
    # A sub-pixel simplification tolerance (~15 EPSG:3857 units ~ 13 m, well
    # below one print pixel at this cell size) keeps the boundaries visually
    # identical while cutting the vector vertex count (and PDF size) sharply.
    SIMPLIFY_TOL = 15.0
    tract_rot = tract.copy()
    tract_rot["geometry"] = (tract.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
                             .rotate(-MAP_ROTATION_DEG, origin=origin))
    tract_rot = tract_rot.set_geometry("geometry")
    county_rot = county.copy()
    county_rot["geometry"] = (county.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
                              .rotate(-MAP_ROTATION_DEG, origin=origin))
    county_rot = county_rot.set_geometry("geometry")

    minx, miny, maxx, maxy = boundary_extent(tract_rot)
    px, py = (maxx - minx) * 0.04, (maxy - miny) * 0.04
    extent = (minx - px, miny - py, maxx + px, maxy + py)

    print(f"Writing vector CLR panels to {OUT_DIR}")
    for stem, col, cmap_name, label in METRICS:
        make_panel(tract_rot, county_rot, extent, col, cmap_name, label,
                   OUT_DIR / f"{stem}.pdf")

    # S8 sensitivity (same vector approach, 1x3, diverging colorbar)
    make_sensitivity_panel(tract_rot, county_rot, extent,
                           OUT_DIR / "S-F8_sensitivity.pdf")


if __name__ == "__main__":
    main()
