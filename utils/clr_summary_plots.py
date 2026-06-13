import os
from pathlib import Path
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


HURRICANE_LABELS = ["Ike", "FEMA33", "FEMA36"]
# Map plot labels to ID tokens used in file names
HURRICANE_IDS = {"Ike": "ike", "FEMA33": "fema33", "FEMA36": "fema36"}

# Year mapping: display year -> file year
YEAR_MAP = {2020: 2020, 2030: 2030, 2040: 2040}


def _ensure_font(font_name: str = "STIX Two Text", fallback: str = "STIXGeneral", size_pt: int = 8):
    """Try to use the requested font; if missing, fall back gracefully."""
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.size"] = size_pt
        return font_name
    except Exception:
        # Try fallback STIX family that ships with Matplotlib
        try:
            fm.findfont(fallback, fallback_to_default=False)
            plt.rcParams["font.family"] = fallback
            plt.rcParams["font.size"] = size_pt
            print(f"Warning: '{font_name}' not found. Using '{fallback}' instead.")
            return fallback
        except Exception:
            # last resort: default font
            plt.rcParams["font.size"] = size_pt
            print(f"Warning: Neither '{font_name}' nor '{fallback}' found. Using Matplotlib default font.")
            return None


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_clr_csv_path(config: dict, hurricane_id: str, year: int) -> Path:
    try:
        template = config["data_sources"]["clr"]["input_path_template"]
    except Exception:
        raise KeyError("Could not find data_sources.clr.input_path_template in config.")
    path = template.format(hurricane_name=hurricane_id, year=year)
    return Path(path)


def _load_clr_values_from_csv(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns ['FID', 'CLR'] averaged across sample columns.
    If CSV has first column as FID and remaining as sample values, average across samples per FID.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CLR CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CLR CSV is empty: {csv_path}")

    # Assume first column is FID (or rename to FID)
    fid_col = df.columns[0]
    if str(fid_col).upper() != "FID":
        df = df.rename(columns={fid_col: "FID"})

    value_cols = df.columns[1:]
    if len(value_cols) == 0:
        raise ValueError(f"No value columns found in CLR CSV (expected samples from col 1+): {csv_path}")

    try:
        values = df[value_cols].astype(float)
    except Exception:
        # Coerce errors to NaN and still compute means
        values = df[value_cols].apply(pd.to_numeric, errors="coerce")

    clr = values.mean(axis=1)
    out = pd.DataFrame({"FID": df["FID"], "CLR": clr})
    # Clip to [0,1] just in case of slight numerical artifacts
    out["CLR"] = out["CLR"].clip(lower=0, upper=1)
    return out


def _build_long_dataframe(config: dict) -> pd.DataFrame:
    """Build a long-form DataFrame with columns: ['Year','Hurricane','CLR'].
    Reads CSVs using the config template; file years match display years.
    """
    records = []
    for year_display in [2020, 2030, 2040]:
        year_file = YEAR_MAP[year_display]
        for hurricane in HURRICANE_LABELS:
            h_id = HURRICANE_IDS.get(hurricane, hurricane.lower())
            csv_path = _get_clr_csv_path(config, h_id, year_file)
            try:
                df_vals = _load_clr_values_from_csv(csv_path)
            except Exception as e:
                print(f"Warning: Skipping {hurricane} {year_display} due to error: {e}")
                continue
            for val in df_vals["CLR"].values:
                records.append({"Year": year_display, "Hurricane": hurricane, "CLR": float(val) if pd.notna(val) else np.nan})

    long_df = pd.DataFrame.from_records(records)
    # drop NaNs
    long_df = long_df.dropna(subset=["CLR"]).reset_index(drop=True)
    return long_df


def _compute_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute stats per (Year, Hurricane)."""
    if long_df.empty:
        return pd.DataFrame(columns=[
            "Year", "Hurricane", "count", "mean", "std", "min", "q1", "median", "q3", "max"
        ])
    grouped = long_df.groupby(["Year", "Hurricane"])['CLR']
    stats = grouped.agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max')
    ]).reset_index()
    # quantiles
    q = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    q.columns = ["Year", "Hurricane", "q1", "q3"]
    stats = stats.merge(q, on=["Year", "Hurricane"], how="left")
    # reorder columns
    stats = stats[["Year", "Hurricane", "count", "mean", "std", "min", "q1", "median", "q3", "max"]]
    return stats


def _hurricane_colors() -> dict:
    # Professional, distinct palette; fixed across years
    return {
        "Ike": "#1f77b4",     # blue
        "FEMA33": "#2ca02c",  # green
        "FEMA36": "#d62728",  # red
    }


def _group_positions():
    """Return x positions for each (Year, Hurricane) and group centers for tick labels.
    We keep hurricanes close within a year, and year groups spaced modestly.
    """
    years = [2020, 2030, 2040]
    n_h = len(HURRICANE_LABELS)
    # within-group spacing
    dx = 0.22
    # group base spacing
    group_gap = 1.2  # moderately spaced groups
    x_positions = {}
    group_centers = {}
    for i, y in enumerate(years):
        base = i * group_gap
        offsets = [(-dx), 0.0, (+dx)] if n_h == 3 else np.linspace(-dx, +dx, n_h)
        xs = [base + off for off in offsets]
        for h, x in zip(HURRICANE_LABELS, xs):
            x_positions[(y, h)] = x
        group_centers[y] = base
    return x_positions, group_centers


def _apply_common_axes_style(ax, group_centers):
    # X axis labels at group centers only (years)
    years = [2020, 2030, 2040]
    ax.set_xticks([group_centers[y] for y in years], [str(y) for y in years])
    # Y axis log scale with safe lower bound
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.0)
    ax.set_ylabel("Connectivity loss ratio")
    # Reduce height aesthetics: handled by figsize externally
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.3)


def _legend(ax, colors):
    handles = []
    for h in HURRICANE_LABELS:
        handles.append(plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[h], markeredgecolor='none', label=h))
    leg = ax.legend(handles=handles, loc='upper right', framealpha=0.35, facecolor='white', edgecolor='none', title="Hurricane")
    if leg is not None and leg.get_frame() is not None:
        leg.get_frame().set_alpha(0.35)


def plot_points_plus_box(long_df: pd.DataFrame, out_path: Path):
    colors = _hurricane_colors()
    _ensure_font()
    x_pos, centers = _group_positions()
    fig, ax = plt.subplots(figsize=(7, 3))

    # Scatter points per combination
    for (year, hurricane), sub in long_df.groupby(["Year", "Hurricane"]):
        if sub.empty: continue
        x = x_pos[(year, hurricane)]
        # jitter to avoid overplot
        jitter = (np.random.rand(len(sub)) - 0.5) * 0.06
        y = sub["CLR"].clip(lower=1e-4, upper=1).values
        ax.scatter(np.full(len(sub), x) + jitter, y, s=10, color=colors[hurricane], alpha=0.5, edgecolors='none', zorder=1)

    # Box plots per combination
    positions = [x_pos[(year, h)] for year in [2020, 2030, 2040] for h in HURRICANE_LABELS]
    data = [long_df[(long_df.Year == year) & (long_df.Hurricane == h)]["CLR"].clip(lower=1e-4, upper=1).values
            for year in [2020, 2030, 2040] for h in HURRICANE_LABELS]
    bp = ax.boxplot(data, positions=positions, widths=0.12, patch_artist=True, manage_ticks=False, whis=1.5, showfliers=False)
    # Color boxes by hurricane
    for i, patch in enumerate(bp['boxes']):
        year = [2020,2030,2040][(i // len(HURRICANE_LABELS))]
        h = HURRICANE_LABELS[i % len(HURRICANE_LABELS)]
        patch.set_facecolor(colors[h]); patch.set_alpha(0.35); patch.set_edgecolor(colors[h])
    for elem in ['whiskers', 'caps', 'medians']:
        for line in bp[elem]:
            line.set_color('#444444'); line.set_alpha(0.8); line.set_linewidth(0.8)

    _apply_common_axes_style(ax, centers)
    _legend(ax, colors)
    ax.set_title("Connectivity loss ratio across years and hurricanes")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_plus_means(long_df: pd.DataFrame, out_path: Path):
    colors = _hurricane_colors()
    _ensure_font()
    x_pos, centers = _group_positions()
    fig, ax = plt.subplots(figsize=(7, 3))

    # scatter points
    for (year, hurricane), sub in long_df.groupby(["Year", "Hurricane"]):
        if sub.empty: continue
        x = x_pos[(year, hurricane)]
        jitter = (np.random.rand(len(sub)) - 0.5) * 0.06
        y = sub["CLR"].clip(lower=1e-4, upper=1).values
        ax.scatter(np.full(len(sub), x) + jitter, y, s=8, color=colors[hurricane], alpha=0.35, edgecolors='none', zorder=1)

    # means
    means = long_df.groupby(["Year", "Hurricane"])['CLR'].mean().reset_index()
    for _, row in means.iterrows():
        x = x_pos[(int(row['Year']), row['Hurricane'])]
        y = max(row['CLR'], 1e-4)
        # mean marker smaller than Plot 1 points
        ax.scatter(x, y, s=8, color='white', edgecolors=colors[row['Hurricane']], linewidths=1.0, zorder=3)

    _apply_common_axes_style(ax, centers)
    _legend(ax, colors)
    ax.set_title("Connectivity loss ratio (points and means)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_box_only(long_df: pd.DataFrame, out_path: Path):
    colors = _hurricane_colors()
    _ensure_font()
    x_pos, centers = _group_positions()
    fig, ax = plt.subplots(figsize=(7, 3))

    positions = [x_pos[(year, h)] for year in [2020, 2030, 2040] for h in HURRICANE_LABELS]
    data = [long_df[(long_df.Year == year) & (long_df.Hurricane == h)]["CLR"].clip(lower=1e-4, upper=1).values
            for year in [2020, 2030, 2040] for h in HURRICANE_LABELS]

    bp = ax.boxplot(data, positions=positions, widths=0.18, patch_artist=True, manage_ticks=False, whis=1.5, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        h = HURRICANE_LABELS[i % len(HURRICANE_LABELS)]
        patch.set_facecolor(colors[h]); patch.set_alpha(0.5); patch.set_edgecolor(colors[h])
    for elem in ['whiskers', 'caps', 'medians']:
        for line in bp[elem]:
            line.set_color('#444444'); line.set_alpha(0.8); line.set_linewidth(0.8)

    _apply_common_axes_style(ax, centers)
    _legend(ax, colors)
    ax.set_title("Connectivity loss ratio (box summaries)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def generate_clr_summary_plots(config_path: str = "./config/config_plots.yaml",
                               output_dir: str = "./outputs/figure/clr_plot") -> Path:
    """Main entrypoint to generate the three CLR summary plots and stats CSV.

    Returns the directory path containing outputs.
    """
    config = _load_config(Path(config_path))
    long_df = _build_long_dataframe(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save stats
    stats = _compute_stats(long_df)
    stats_path = out_dir / "clr_summary_stats.csv"
    stats.to_csv(stats_path, index=False)

    if long_df.empty:
        print("Warning: No CLR values found to plot. Saved empty stats and skipped figures.")
        return out_dir

    # Plots
    plot_points_plus_box(long_df, out_dir / "clr_points_plus_box.png")
    plot_scatter_plus_means(long_df, out_dir / "clr_scatter_plus_means.png")
    plot_box_only(long_df, out_dir / "clr_box_only.png")

    print(f"CLR summary outputs saved in: {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_clr_summary_plots()
