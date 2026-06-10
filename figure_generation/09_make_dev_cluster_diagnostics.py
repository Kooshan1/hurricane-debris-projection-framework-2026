"""
S-F7: Developed-land cluster diagnostics.

Produces two side-by-side panels at standardised journal quality:
  (a) ALE curves for the three sub-classes (DT, DM, DO) on the observed
      data manifold.
  (b) Joint cluster effect on debris volume (one curve per "all three
      sub-classes scaled jointly" sweep).

The source CSVs come from
  debris_volume_model/outputs/run_v7d_hazard_interactions/evaluation/
and are produced by the model-evaluation pipeline. This script merely
re-renders them into a clean, journal-quality figure consistent with
the rest of the revision figures.

Outputs (both PNG and PDF; rendered independently):
  S-F7_dev_cluster_diagnostics/S-F7_dev_cluster_merged.png
  S-F7_dev_cluster_diagnostics/S-F7_dev_cluster_merged.pdf
  S-F7_dev_cluster_diagnostics/dev_cluster_diagnostics_data.csv (the
      input ALE / joint-effect data merged into a single CSV)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DPI, FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE,
                    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
                    OUTPUTS_ROOT, REVISION_FIG_ROOT, ensure_font)

# Source data: ALE and joint-effect CSVs from the model-evaluation run.
EVAL_DIR = OUTPUTS_ROOT.parent / "debris_volume_model" / "outputs" \
           / "run_v7d_hazard_interactions" / "evaluation"
OUT_DIR = REVISION_FIG_ROOT / "S-F7_dev_cluster_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBCLASS_COLOURS = {
    "DT": "#d62728",   # red --- total developed
    "DM": "#1f77b4",   # blue --- medium-intensity developed
    "DO": "#2ca02c",   # green --- open-space developed
}
SUBCLASS_LABEL = {
    "DT": "DT (total developed)",
    "DM": "DM (medium-intensity developed)",
    "DO": "DO (open-space developed)",
}


def _load_ale_curves():
    """Load DT / DM / DO ALE curves from the evaluation directory."""
    out = {}
    for sub in ("DT", "DM", "DO"):
        csv = EVAL_DIR / f"ale_{sub}.csv"
        if csv.is_file():
            out[sub] = pd.read_csv(csv)
    return out


def _load_joint_effect():
    csv = EVAL_DIR / "dev_cluster_joint_effect.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return None


def _xy_columns_ale(df: pd.DataFrame) -> tuple[str, str]:
    """ALE CSVs use columns: bin_midpoint_original_scale (x), ale_centred_m3 (y), bin_count."""
    return "bin_midpoint_original_scale", "ale_centred_m3"


def _xy_columns_joint(df: pd.DataFrame) -> tuple[str, str]:
    """Joint-effect CSV: cluster_axis_mid (x), mean_pred_m3 (y)."""
    return "cluster_axis_mid", "mean_pred_m3"


def _plot_merged(out_path: Path):
    """Four-panel layout: (a) DT ALE, (b) DM ALE, (c) DO ALE, (d) joint cluster.

    Rationale: the three sub-classes (DT, DM, DO) have very different
    effect magnitudes, so plotting them on a single shared y-axis squashes
    the smaller-magnitude curves into a flat line. We instead use a 2x2
    grid where each ALE curve gets its own panel with auto-scaled y-axis,
    making per-sub-class trends visible. The fourth panel (bottom-right)
    shows the joint cluster effect (all three sub-classes scaled
    together), so all four panels are visually consistent in size and
    styling.
    """
    ensure_font()
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_DOUBLE, 4.6),
        gridspec_kw={"wspace": 0.32, "hspace": 0.45},
    )
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10)

    ale_curves = _load_ale_curves()

    def _style_axes(ax):
        ax.set_facecolor("none")
        ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=FONT_SIZE_NORMAL)

    # ---- Panels (a), (b), (c): individual sub-class ALE curves ----
    panel_specs = [
        ("DT", axes[0, 0], "(a)"),
        ("DM", axes[0, 1], "(b)"),
        ("DO", axes[1, 0], "(c)"),
    ]
    for sub, ax, letter in panel_specs:
        df = ale_curves.get(sub)
        if df is None:
            continue
        x_col, y_col = _xy_columns_ale(df)
        # ALE is binned with only ~9-11 points per sub-class; show markers
        # so the reader can see the discrete observations.
        ax.plot(df[x_col], df[y_col],
                color=SUBCLASS_COLOURS[sub], linewidth=1.8,
                marker="o", markersize=4)
        _style_axes(ax)
        ax.set_xlabel(f"{sub} cell fraction", fontsize=FONT_SIZE_NORMAL)
        ax.set_ylabel(r"ALE effect (m$^{3}$/cell)",
                      fontsize=FONT_SIZE_NORMAL)
        ax.set_title(f"{letter} {SUBCLASS_LABEL[sub]}",
                     fontsize=FONT_SIZE_LARGE, loc="left", pad=4)

    # ---- Panel (d): joint cluster effect ----
    ax_joint = axes[1, 1]
    joint = _load_joint_effect()
    if joint is not None and len(joint):
        x_col, y_col = _xy_columns_joint(joint)
        ax_joint.plot(joint[x_col], joint[y_col],
                      color="#7b3294", linewidth=1.8, marker="o", markersize=4)
    _style_axes(ax_joint)
    ax_joint.set_xlabel("Joint cluster scaling factor",
                       fontsize=FONT_SIZE_NORMAL)
    ax_joint.set_ylabel(r"Mean predicted debris (m$^{3}$/cell)",
                       fontsize=FONT_SIZE_NORMAL)
    ax_joint.set_title("(d) Joint cluster (all three sub-classes scaled together)",
                       fontsize=FONT_SIZE_LARGE, loc="left", pad=4)

    # Independent PNG (high DPI raster) + PDF (matplotlib PDF backend with
    # embedded fonts via pdf.fonttype=42) so neither output is derived from
    # the other.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=1200, bbox_inches="tight",
                transparent=True, facecolor="none", pad_inches=0.04)
    pdf_path = out_path.with_suffix(".pdf")
    prev = plt.rcParams.get("pdf.fonttype")
    plt.rcParams["pdf.fonttype"] = 42
    try:
        fig.savefig(pdf_path, dpi=600, bbox_inches="tight",
                    transparent=True, facecolor="none", pad_inches=0.04)
    finally:
        if prev is not None:
            plt.rcParams["pdf.fonttype"] = prev
    plt.close(fig)
    print(f"  wrote {out_path.name} (+ pdf)")


def main():
    if not EVAL_DIR.is_dir():
        print(f"[skip] evaluation dir missing: {EVAL_DIR}")
        return
    _plot_merged(OUT_DIR / "S-F7_dev_cluster_merged.png")

    # Also export a single merged CSV for transparency / data deposit.
    pieces = []
    for sub, df in _load_ale_curves().items():
        x_col, y_col = _xy_columns_ale(df)
        out = df[[x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})
        out["series"] = f"ALE_{sub}"
        pieces.append(out)
    joint = _load_joint_effect()
    if joint is not None and len(joint):
        x_col, y_col = _xy_columns_joint(joint)
        out = joint[[x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})
        out["series"] = "joint_cluster"
        pieces.append(out)
    if pieces:
        all_data = pd.concat(pieces, ignore_index=True)
        all_data.to_csv(OUT_DIR / "dev_cluster_diagnostics_data.csv", index=False)
        print(f"  wrote dev_cluster_diagnostics_data.csv")
    print(f"\nS-F7 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
