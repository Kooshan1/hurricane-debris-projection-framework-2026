"""
Generate paper-quality visualisations of the feature-set changes from the
original NN (32 inputs, all unconstrained) to a physics-informed variant
(v3 or v4b).

CLI:
    python plot_feature_transformations.py [--variant v3|v4b]
        defaults to v3 for backwards compatibility.

Outputs:
  outputs/run_<variant>/feature_documentation/
    feature_transformation_table.csv      -- machine-readable change log
    feature_overview.png                  -- 4-group panel: kept / dropped / engineered / direction
    feature_priors.png                    -- coloured 3-column INC/DEC/FREE table
    feature_priors_compact.png            -- horizontal coloured bar
    feature_flow.png                      -- Sankey-style 32 -> N flow
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent

# variant selection set in main(); module-level OUT_DIR is rebound there.
OUT_DIR = THIS_DIR / "outputs" / "run_v3" / "feature_documentation"

# ============================================================================
# Reference data
# ============================================================================
ORIGINAL_FEATURES = [
    # Multi-hazard intensity (8)
    ("SD", "Multi-hazard", "Maximum surge depth (m)"),
    ("WH", "Multi-hazard", "Maximum significant wave height (m)"),
    ("WaveD", "Multi-hazard", "Mean wave direction (deg)"),
    ("WV_X", "Multi-hazard", "Maximum wave velocity, x-component (m/s)"),
    ("WV_Y", "Multi-hazard", "Maximum wave velocity, y-component (m/s)"),
    ("WindD", "Multi-hazard", "Mean wind direction (deg)"),
    ("WS", "Multi-hazard", "Mean wind speed (m/s)"),
    ("MF", "Multi-hazard", "Multi-hazard damage factor (-)"),
    # Built environment (7)
    ("NumB", "Built env", "Number of buildings (-)"),
    ("TBFA", "Built env", "Total building footprint area (m^2)"),
    ("NAS", "Built env", "Number of accessory structures (-)"),
    ("TAAS", "Built env", "Total area of accessory structures (m^2)"),
    ("NumMH", "Built env", "Number of mobile homes (-)"),
    ("WPF_1", "Built env", "Weighted performance factor 1 (-)"),
    ("WPF_2", "Built env", "Weighted performance factor 2 (-)"),
    # Natural environment (8)
    ("OW", "Natural env", "Open-water fraction (-)"),
    ("DO", "Natural env", "Developed open-space fraction (-)"),
    ("DM", "Natural env", "Developed medium intensity fraction (-)"),
    ("DT", "Natural env", "Developed high intensity fraction (-)"),
    ("RD", "Natural env", "Road density (km/km^2)"),
    ("ME", "Natural env", "Minimum elevation (m)"),
    ("ADS", "Natural env", "Distance to protective structures (m)"),
    ("UL", "Natural env", "Urban land fraction (-)"),
    # Human / socioeconomic (7)
    ("PD", "Socioeconomic", "Population density (1/km^2)"),
    ("NumH", "Socioeconomic", "Number of households (-)"),
    ("MHI", "Socioeconomic", "Median household income (USD)"),
    ("NumHU", "Socioeconomic", "Number of housing units (-)"),
    ("OHU", "Socioeconomic", "Owner-occupied housing units (-)"),
    ("VHU", "Socioeconomic", "Vacant housing units (-)"),
    ("PR", "Socioeconomic", "Poverty rate (-)"),
    # Spatial (2)
    ("centroid_x", "Spatial", "Cell-centroid longitude (deg)"),
    ("centroid_y", "Spatial", "Cell-centroid latitude (deg)"),
]

# v3 feature classification (from feature_groups_v3.json)
V3_INC = {"SD", "WH", "WS", "MF", "WPF_1", "WPF_2",
          "NumB", "mean_bldg_size_m2", "NAS", "TAAS", "NumMH",
          "NumHU", "PD"}
V3_DEC = {"ME", "dist_to_coast_m"}
V3_FREE = {"WaveD", "WindD", "WV_X", "WV_Y",
           "OW", "DO", "DM", "DT", "RD",
           "ADS", "UL",
           "MHI", "PR"}
V3_DROPPED = {"centroid_x", "centroid_y", "TBFA", "VHU", "OHU", "NumH"}

# v4b feature classification (from feature_groups_v4b.json):
#   - WaveD, WindD dropped (storm-track-localized)
#   - OHU, VHU restored as monotone-INC (LCC artifacts suppressed by
#     monotone-sum architecture)
#   - NumH stays dropped (numerically identical to OHU)
V4B_INC = {"SD", "WH", "WS", "MF", "WPF_1", "WPF_2",
           "NumB", "mean_bldg_size_m2", "NAS", "TAAS", "NumMH",
           "NumHU", "OHU", "VHU", "PD"}
V4B_DEC = {"ME", "dist_to_coast_m"}
V4B_FREE = {"WV_X", "WV_Y",
            "OW", "DO", "DM", "DT", "RD",
            "ADS", "UL",
            "MHI", "PR"}
V4B_DROPPED = {"centroid_x", "centroid_y", "TBFA", "NumH", "WaveD", "WindD"}

V3_ENGINEERED = [
    ("mean_bldg_size_m2", "Built env", "TBFA / max(NumB, 1) -- mean building size (m^2)"),
    ("dist_to_coast_m", "Natural env", "Distance from cell centroid to nearest open-water cell (m)"),
]
V4B_ENGINEERED = V3_ENGINEERED  # same engineered features in both variants

# v7d (production): v4c minus the FID-16765 outlier in training PLUS 3 hazard-x-
# building interaction features as monotone-INC. Same 27 v4c features carry over;
# only the 3 interactions are added.
V7D_INC = V4B_INC | {"SD_NumB", "WH_NumB", "MF_NumB"}
V7D_DEC = V4B_DEC
V7D_FREE = V4B_FREE - {"UL"}                       # UL dropped (v4c heritage)
V7D_DROPPED = V4B_DROPPED | {"UL"}                 # UL dropped (v4c heritage)
V7D_ENGINEERED = V4B_ENGINEERED + [
    ("SD_NumB", "Hazard x Building",
     "SD * NumB -- surge depth times number of buildings (exposed stock x hazard)"),
    ("WH_NumB", "Hazard x Building",
     "WH * NumB -- wave height times number of buildings"),
    ("MF_NumB", "Hazard x Building",
     "MF * NumB -- momentum flux times number of buildings"),
]

REASONS_V3 = {
    "centroid_x": "Region-specific position memorisation; not physical",
    "centroid_y": "Region-specific position memorisation; not physical",
    "TBFA":       "r=0.74 with NumB; opposite-sign PDPs in original NN. Replaced with mean_bldg_size_m2",
    "VHU":        "Decomposition of NumHU; LCC's 2030 vacancy dip propagates non-physically",
    "OHU":        "Decomposition of NumHU; redundant",
    "NumH":       "Numerically identical to OHU in LCC inputs; redundant",
}
REASONS_V4B = {
    "centroid_x": "Region-specific position memorisation; not physical",
    "centroid_y": "Region-specific position memorisation; not physical",
    "TBFA":       "r=0.74 with NumB; opposite-sign PDPs in original NN. Replaced with mean_bldg_size_m2",
    "NumH":       "Numerically identical to OHU in LCC inputs; redundant",
    "WaveD":      "Storm-track-specific (Hurricane-Ike-Galveston only); does not generalize",
    "WindD":      "Storm-track-specific (Hurricane-Ike-Galveston only); does not generalize",
}

JUSTIFICATIONS = {
    # monotone-increasing
    "SD":    "more surge -> more inundation/buoyancy damage",
    "WH":    "more wave height -> more wave damage",
    "WS":    "more wind speed -> more wind damage",
    "MF":    "multi-hazard damage factor; higher -> more damage",
    "WPF_1": "performance factor; higher vulnerability -> more debris",
    "WPF_2": "performance factor; higher vulnerability -> more debris",
    "NumB":  "more buildings -> more debris sources",
    "mean_bldg_size_m2": "larger buildings -> more debris per damaged unit",
    "NAS":   "more accessory structures -> more debris sources",
    "TAAS":  "larger accessory structure area -> more debris",
    "NumMH": "more mobile homes -> more vulnerable debris sources",
    "NumHU": "more housing units -> more building stock at risk",
    "OHU":   "more occupied housing units -> more building stock at risk",
    "VHU":   "more vacant housing units -> more poorly-maintained building stock",
    "PD":    "higher population density -> denser development",
    # monotone-decreasing
    "ME":    "higher elevation -> less inundation -> less debris",
    "dist_to_coast_m": "further from coast -> less surge exposure",
    # v7d hazard-x-building interactions (mono-INC)
    "SD_NumB": "amount of building stock exposed to surge depth -> more debris",
    "WH_NumB": "amount of building stock exposed to wave height -> more debris",
    "MF_NumB": "amount of building stock exposed to momentum flux -> more debris",
}

# ============================================================================
# Build the transformation table
# ============================================================================
def build_table(variant: str = "v3") -> pd.DataFrame:
    if variant == "v4b":
        INC, DEC, FREE, DROPPED = V4B_INC, V4B_DEC, V4B_FREE, V4B_DROPPED
        REASONS = REASONS_V4B
        ENGINEERED = V4B_ENGINEERED
    elif variant == "v4c":
        # v4c = v4b but with UL also dropped (highly redundant with DT, r=0.93)
        INC = V4B_INC
        DEC = V4B_DEC
        FREE = V4B_FREE - {"UL"}
        DROPPED = V4B_DROPPED | {"UL"}
        REASONS = {**REASONS_V4B,
                   "UL": "Urban-lag is highly redundant with DT (r=0.934) and DO (r=0.765); dropped to break the multicollinearity that drove DT's anomalous negative PDP slope in v4b."}
        ENGINEERED = V4B_ENGINEERED
    elif variant == "v7d":
        # v7d (production) = v4c + 3 hazard-x-building interactions (mono-INC),
        # trained on the no-outlier dataset (FID-16765 removed)
        INC, DEC, FREE, DROPPED = V7D_INC, V7D_DEC, V7D_FREE, V7D_DROPPED
        REASONS = {**REASONS_V4B,
                   "UL": "Urban-lag is highly redundant with DT (r=0.934) and DO (r=0.765); dropped to break the multicollinearity that drove DT's anomalous negative PDP slope in v4b/v4c heritage."}
        ENGINEERED = V7D_ENGINEERED
    else:
        INC, DEC, FREE, DROPPED = V3_INC, V3_DEC, V3_FREE, V3_DROPPED
        REASONS = REASONS_V3
        ENGINEERED = V3_ENGINEERED

    rows = []
    for feat, group, desc in ORIGINAL_FEATURES:
        if feat in DROPPED:
            status = "DROPPED"
            v3_group = "n/a"
            direction = "n/a"
            rationale = REASONS.get(feat, "")
        elif feat in INC:
            status = "KEPT"
            v3_group = "monotone-INC"
            direction = "non-decreasing"
            rationale = JUSTIFICATIONS.get(feat, "")
        elif feat in DEC:
            status = "KEPT"
            v3_group = "monotone-DEC"
            direction = "non-increasing"
            rationale = JUSTIFICATIONS.get(feat, "")
        elif feat in FREE:
            status = "KEPT"
            v3_group = "free"
            direction = "unconstrained"
            rationale = "directional / categorical / sign-dependent"
        else:
            status = "?"
            v3_group = "?"
            direction = "?"
            rationale = ""
        rows.append({
            "feature": feat,
            "original_group": group,
            "description": desc,
            "v3_status": status,
            "v3_group": v3_group,
            "v3_direction": direction,
            "rationale": rationale,
        })
    # add engineered features
    for feat, group, desc in ENGINEERED:
        if feat in INC:
            v3_group = "monotone-INC"
            direction = "non-decreasing"
        elif feat in DEC:
            v3_group = "monotone-DEC"
            direction = "non-increasing"
        else:
            v3_group = "free"
            direction = "unconstrained"
        rows.append({
            "feature": feat,
            "original_group": "n/a (added)",
            "description": desc,
            "v3_status": "ADDED (engineered)",
            "v3_group": v3_group,
            "v3_direction": direction,
            "rationale": JUSTIFICATIONS.get(feat, ""),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Visualisation 1: Feature overview matrix
# ============================================================================
def plot_overview(df: pd.DataFrame, out_path: Path):
    """A grid where each row = group; columns = features; cell colour = status."""
    # Status colour scheme
    color_kept_inc = "#1b7837"   # dark green
    color_kept_dec = "#9970ab"   # purple
    color_kept_free = "#4393c3"  # blue
    color_dropped = "#b2182b"    # red
    color_added = "#fb6a4a"      # orange-red

    def status_color(row):
        if row["v3_status"].startswith("DROPPED"):
            return color_dropped
        if row["v3_status"].startswith("ADDED"):
            return color_added
        if row["v3_group"] == "monotone-INC":
            return color_kept_inc
        if row["v3_group"] == "monotone-DEC":
            return color_kept_dec
        return color_kept_free

    def status_marker(row):
        if row["v3_status"].startswith("DROPPED"):
            return "×"
        if row["v3_group"] == "monotone-INC":
            return "↑"
        if row["v3_group"] == "monotone-DEC":
            return "↓"
        return "~"

    # Order groups for the figure
    group_order = ["Multi-hazard", "Built env", "Natural env", "Socioeconomic", "Spatial",
                   "n/a (added)"]
    group_label_pretty = {
        "Multi-hazard": "Multi-hazard intensity",
        "Built env": "Built environment",
        "Natural env": "Natural environment",
        "Socioeconomic": "Human / socioeconomic",
        "Spatial": "Spatial coordinates",
        "n/a (added)": "Engineered (added)",
    }

    # For each group, list features in order
    fig, ax = plt.subplots(figsize=(11, 6.0))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")

    row_step = 1.5  # vertical spacing between groups (was 1; too tight for rotated labels)
    y = 0
    yticks = []
    yticklabels = []
    for grp in group_order:
        sub = df[df["original_group"] == grp]
        if sub.empty:
            continue
        feats = sub["feature"].tolist()
        for i, feat in enumerate(feats):
            row = sub.iloc[i]
            col = status_color(row)
            mk = status_marker(row)
            ax.scatter(i, y, s=260, color=col, alpha=0.85,
                       edgecolors="black", linewidths=0.4, zorder=2)
            ax.text(i, y, mk, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=3)
            ax.text(i, y - 0.30, feat, ha="center", va="top",
                    fontsize=7, rotation=30, rotation_mode="anchor")
        yticks.append(y)
        yticklabels.append(f"{group_label_pretty[grp]} ({len(feats)})")
        y -= row_step

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xticks([])
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.set_xlim(-0.6, max(df.groupby("original_group").size().max(), 8) - 0.4)
    ax.set_ylim(y + 0.9, 0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color_kept_inc, label="Kept, monotone-INC (↑)"),
        mpatches.Patch(color=color_kept_dec, label="Kept, monotone-DEC (↓)"),
        mpatches.Patch(color=color_kept_free, label="Kept, free / unconstrained (~)"),
        mpatches.Patch(color=color_dropped, label="Dropped (×)"),
        mpatches.Patch(color=color_added, label="Added (engineered)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.32), framealpha=0.0,
              ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# Visualisation 2: v3 directional priors (the active features)
# ============================================================================
def plot_priors(df: pd.DataFrame, out_path: Path):
    """Three vertical columns for v3's INC / DEC / FREE features with descriptions."""
    inc_features = df[(df["v3_group"] == "monotone-INC")][["feature", "rationale"]]
    dec_features = df[(df["v3_group"] == "monotone-DEC")][["feature", "rationale"]]
    free_features = df[(df["v3_group"] == "free")][["feature", "rationale"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [1.3, 0.8, 1.1]})
    fig.patch.set_alpha(0.0)

    palette = {
        "monotone-INC": "#1b7837",
        "monotone-DEC": "#9970ab",
        "free":          "#4393c3",
    }

    # Use a common y-scale so rows in DEC don't get stretched out
    common_n = max(len(inc_features), len(dec_features), len(free_features))

    def render_column(ax, df_col, title, color, arrow, label_x, rationale_x):
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        n = len(df_col)
        ax.set_xlim(0, 1)
        # Force a common y-range so vertical line spacing is the same across columns
        ax.set_ylim(-common_n - 0.5, 0.5)
        # Title
        ax.text(0.5, 0.4, f"{title}  {arrow}  ({n})",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=color)
        for i, (_, row) in enumerate(df_col.iterrows()):
            y = -i - 0.5
            # Bullet marker
            ax.scatter(0.03, y, s=80, color=color, alpha=0.85,
                        edgecolors="black", linewidths=0.4)
            ax.text(label_x, y, row["feature"], ha="left", va="center",
                    fontsize=9, fontweight="bold")
            ax.text(rationale_x, y, row["rationale"], ha="left", va="center",
                    fontsize=7.5, color="#444444")

    # Wider label column where long names live (mean_bldg_size_m2, dist_to_coast_m)
    render_column(axes[0], inc_features, "Monotone-INCREASING",
                  palette["monotone-INC"], "↑",
                  label_x=0.08, rationale_x=0.46)
    render_column(axes[1], dec_features, "Monotone-DECREASING",
                  palette["monotone-DEC"], "↓",
                  label_x=0.08, rationale_x=0.62)
    render_column(axes[2], free_features, "Free / unconstrained",
                  palette["free"], "~",
                  label_x=0.08, rationale_x=0.30)

    fig.suptitle("v3 input features and their directional priors",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# Visualisation 3: A compact horizontal bar showing v3's 28 features colored
# by direction
# ============================================================================
def plot_priors_compact(df: pd.DataFrame, out_path: Path):
    palette = {"monotone-INC": "#1b7837",
               "monotone-DEC": "#9970ab",
               "free":          "#4393c3"}
    df_v3 = df[df["v3_group"].isin(palette)].copy()
    # Sort: inc first, then dec, then free; alphabetical within each
    order = {"monotone-INC": 0, "monotone-DEC": 1, "free": 2}
    df_v3["sort_key"] = df_v3["v3_group"].map(order)
    df_v3 = df_v3.sort_values(["sort_key", "feature"]).reset_index(drop=True)

    n = len(df_v3)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), 2.0))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")

    for i, (_, row) in enumerate(df_v3.iterrows()):
        col = palette[row["v3_group"]]
        ax.bar(i, 1.0, color=col, alpha=0.85, edgecolor="black", linewidth=0.4)
        # arrow / tilde inside
        sym = {"monotone-INC": "↑", "monotone-DEC": "↓", "free": "~"}[row["v3_group"]]
        ax.text(i, 0.55, sym, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        ax.text(i, 1.05, row["feature"], ha="center", va="bottom",
                fontsize=8, rotation=60, rotation_mode="anchor")

    # Group separators
    n_inc = (df_v3["v3_group"] == "monotone-INC").sum()
    n_dec = (df_v3["v3_group"] == "monotone-DEC").sum()
    ax.axvline(n_inc - 0.5, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(n_inc + n_dec - 0.5, color="black", linewidth=0.4, alpha=0.4)
    # group labels
    ax.text((n_inc - 1) / 2, -0.25, f"Monotone-INC ↑ ({n_inc})",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=palette["monotone-INC"])
    ax.text(n_inc + (n_dec - 1) / 2, -0.25, f"Mono-DEC ↓ ({n_dec})",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=palette["monotone-DEC"])
    ax.text(n_inc + n_dec + ((n - n_inc - n_dec - 1) / 2), -0.25,
            f"Free ~ ({n - n_inc - n_dec})",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=palette["free"])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(-0.6, 2.0)
    ax.set_xlim(-0.6, n - 0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# Visualisation 4: Sankey-style flow showing original 32 -> v3 28
# ============================================================================
def plot_flow(df: pd.DataFrame, out_path: Path):
    """A simple side-by-side flow showing where each original feature went."""
    # Counts
    n_kept_inc = ((df["v3_status"] == "KEPT") & (df["v3_group"] == "monotone-INC") &
                  (df["original_group"] != "n/a (added)")).sum()
    n_kept_dec = ((df["v3_status"] == "KEPT") & (df["v3_group"] == "monotone-DEC") &
                  (df["original_group"] != "n/a (added)")).sum()
    n_kept_free = ((df["v3_status"] == "KEPT") & (df["v3_group"] == "free")).sum()
    n_dropped = (df["v3_status"] == "DROPPED").sum()
    n_added_inc = ((df["v3_status"] == "ADDED (engineered)") & (df["v3_group"] == "monotone-INC")).sum()
    n_added_dec = ((df["v3_status"] == "ADDED (engineered)") & (df["v3_group"] == "monotone-DEC")).sum()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Left side: original (32)
    left = mpatches.FancyBboxPatch((0.05, 0.05), 0.18, 0.9,
                                    boxstyle="round,pad=0.02",
                                    facecolor="#cccccc", edgecolor="black",
                                    linewidth=0.6)
    ax.add_patch(left)
    ax.text(0.14, 0.92, "Original NN", ha="center", va="top",
            fontsize=10, fontweight="bold")
    ax.text(0.14, 0.85, "32 inputs", ha="center", va="top", fontsize=9)
    ax.text(0.14, 0.78, "(none constrained)", ha="center", va="top",
            fontsize=8, color="#666666", style="italic")

    # Right side: v3 (28) — three sub-boxes
    box_w = 0.20
    # INC
    inc_box = mpatches.FancyBboxPatch((0.55, 0.65), box_w, 0.30,
                                       boxstyle="round,pad=0.015",
                                       facecolor="#1b7837", edgecolor="black",
                                       linewidth=0.4, alpha=0.85)
    ax.add_patch(inc_box)
    ax.text(0.65, 0.90, "Monotone-INC ↑", ha="center", va="top",
            fontsize=9, fontweight="bold", color="white")
    ax.text(0.65, 0.78, f"{n_kept_inc + n_added_inc} features", ha="center", va="top",
            fontsize=8.5, color="white")
    ax.text(0.65, 0.71, f"({n_kept_inc} kept + {n_added_inc} engineered)",
            ha="center", va="top", fontsize=7, color="white", style="italic")
    # DEC
    dec_box = mpatches.FancyBboxPatch((0.55, 0.34), box_w, 0.26,
                                       boxstyle="round,pad=0.015",
                                       facecolor="#9970ab", edgecolor="black",
                                       linewidth=0.4, alpha=0.85)
    ax.add_patch(dec_box)
    ax.text(0.65, 0.55, "Monotone-DEC ↓", ha="center", va="top",
            fontsize=9, fontweight="bold", color="white")
    ax.text(0.65, 0.46, f"{n_kept_dec + n_added_dec} features", ha="center", va="top",
            fontsize=8.5, color="white")
    ax.text(0.65, 0.39, f"({n_kept_dec} kept + {n_added_dec} engineered)",
            ha="center", va="top", fontsize=7, color="white", style="italic")
    # FREE
    free_box = mpatches.FancyBboxPatch((0.55, 0.05), box_w, 0.24,
                                        boxstyle="round,pad=0.015",
                                        facecolor="#4393c3", edgecolor="black",
                                        linewidth=0.4, alpha=0.85)
    ax.add_patch(free_box)
    ax.text(0.65, 0.25, "Free ~", ha="center", va="top",
            fontsize=9, fontweight="bold", color="white")
    ax.text(0.65, 0.16, f"{n_kept_free} features", ha="center", va="top",
            fontsize=8.5, color="white")
    ax.text(0.65, 0.10, "(directional/socio.)",
            ha="center", va="top", fontsize=7, color="white", style="italic")

    # Dropped pile (orphan)
    dropped_box = mpatches.FancyBboxPatch((0.30, 0.30), 0.16, 0.40,
                                           boxstyle="round,pad=0.015",
                                           facecolor="#b2182b", edgecolor="black",
                                           linewidth=0.4, alpha=0.85)
    ax.add_patch(dropped_box)
    ax.text(0.38, 0.65, "Dropped ×", ha="center", va="top",
            fontsize=9, fontweight="bold", color="white")
    ax.text(0.38, 0.55, f"{n_dropped} features", ha="center", va="top",
            fontsize=8.5, color="white")
    # Build dropped-feature list dynamically; split across two lines
    dropped_feats = df.loc[df["v3_status"] == "DROPPED", "feature"].tolist()
    half = (len(dropped_feats) + 1) // 2
    line1 = ", ".join(dropped_feats[:half])
    line2 = ", ".join(dropped_feats[half:])
    ax.text(0.38, 0.50, line1 + ("," if line2 else ""), ha="center", va="top",
            fontsize=6.5, color="white")
    if line2:
        ax.text(0.38, 0.45, line2, ha="center", va="top",
                fontsize=6.5, color="white")

    # Arrows
    arrow_kw = dict(head_width=0.015, head_length=0.012, fc="black",
                    ec="black", linewidth=0.3, length_includes_head=True)
    # Original -> INC, DEC, FREE
    ax.annotate("", xy=(0.55, 0.80), xytext=(0.23, 0.80),
                arrowprops=dict(arrowstyle="->", color="#1b7837", linewidth=1.0))
    ax.annotate("", xy=(0.55, 0.47), xytext=(0.23, 0.47),
                arrowprops=dict(arrowstyle="->", color="#9970ab", linewidth=1.0))
    ax.annotate("", xy=(0.55, 0.17), xytext=(0.23, 0.17),
                arrowprops=dict(arrowstyle="->", color="#4393c3", linewidth=1.0))
    # Original -> Dropped
    ax.annotate("", xy=(0.30, 0.50), xytext=(0.23, 0.50),
                arrowprops=dict(arrowstyle="->", color="#b2182b", linewidth=1.0))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v3", "v4b", "v4c", "v7d"], default="v3",
                   help="model variant; routes outputs to outputs/run_<variant>/feature_documentation/")
    args = p.parse_args()

    # Route v7d output to the actual production folder name
    # (run_v7d_hazard_interactions), parallel to where v3/v4b/v4c live.
    folder_for_variant = {
        "v3":  "run_v3",
        "v4b": "run_v4b",
        "v4c": "run_v4c",
        "v7d": "run_v7d_hazard_interactions",
    }
    out_dir = THIS_DIR / "outputs" / folder_for_variant[args.variant] / "feature_documentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(variant=args.variant)
    print(f"Built feature transformation table for variant={args.variant}: {len(df)} rows")
    print(df[["feature", "v3_status", "v3_group", "v3_direction"]].to_string(index=False))

    csv_path = out_dir / "feature_transformation_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    plot_overview(df, out_dir / "feature_overview.png")
    plot_priors(df, out_dir / "feature_priors.png")
    plot_priors_compact(df, out_dir / "feature_priors_compact.png")
    plot_flow(df, out_dir / "feature_flow.png")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
