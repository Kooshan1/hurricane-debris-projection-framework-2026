"""
Shared styling helpers for the revision figures.

Mirrors the conventions used by utils/PlotGenerator.py and
utils/clr_summary_plots.py so the new revision figures look identical to
the existing paper figures (transparent background, no axes, no titles,
DPI 600, custom ColorBrewer-derived colormaps, county-boundary overlay).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# ----------------------------------------------------------------------------
# Project paths (relative to NSF_Debris_Project/)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_ROOT = PROJECT_ROOT.parent / "inputs"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

TRACT_SHP = INPUTS_ROOT / "debris_impact_data" / "Tracts_Galveston_County_Without_Bolivar_Peninsula.shp"
COUNTY_SHP = INPUTS_ROOT / "debris_impact_data" / "galveston_county_border.shp"
GRID_SHP = INPUTS_ROOT / "shapefiles" / "Grid_250_Without_Bolivar_Peninsula.shp"

NETWORK_SHP = OUTPUTS_ROOT / "debris_impact_output" / "monte_carlo_result" / "Network_Condition_results.shp"

# v7d override: point the "MC_DIR_700" symbol at the v7d 1000-sample MC outputs.
# The variable name is kept for drop-in compatibility with the original scripts.
MC_DIR_700 = PROJECT_ROOT / "debris_volume_model" / "outputs" / "v7d_runs"
MC_DIR_100 = OUTPUTS_ROOT / "debris_impact_output" / "monte_carlo_result" / "old_results"
# Suffix used in v7d filenames: result_summary_<storm>_<file_year>_v7d_seeds0-999.csv
MC_FILENAME_SUFFIX = "v7d_seeds0-999"

# Paper figures are written under outputs/figure/paper_figures/.
REVISION_FIG_ROOT = OUTPUTS_ROOT / "figure" / "paper_figures"

TARGET_CRS = "EPSG:3857"

# ----------------------------------------------------------------------------
# Professional figure configuration (journal standards)
# ----------------------------------------------------------------------------
FIG_WIDTH_SINGLE = 3.5    # inches — single-column figure
FIG_WIDTH_DOUBLE = 7.0    # inches — double-column (full page width)
FONT_SIZE_SMALL = 7       # pt — legends, footnotes, secondary labels
FONT_SIZE_NORMAL = 8      # pt — axis labels, tick labels
FONT_SIZE_LARGE = 9       # pt — titles, primary annotations
DPI = 600

# --- Merged map-panel legend readability (S-F2 / S-F8 tract maps) ---
# The shared horizontal colorbar is rendered as a standalone bitmap and then
# placed (downscaled) into the merged panel at LEGEND_BAR_WIDTH_IN inches.
# To make every piece of text in the merged image visually consistent, we want
# the on-page colorbar label/tick size to match the per-panel scenario titles,
# which render at FONT_SIZE_SMALL (7 pt) on the full-width page. Because the
# legend bitmap (figsize width 5 in) is shrunk to LEGEND_BAR_WIDTH_IN inches,
# its render-time font must be scaled up by ~ (5 / LEGEND_BAR_WIDTH_IN) so that
# the *placed* type reads at ~7 pt. This is a MODERATE bump (well below the
# previous 22-24 pt) chosen purely for size consistency, not to enlarge.
LEGEND_BAR_WIDTH_IN = 2.6      # placed width of the legend bar in the merged figure
LEGEND_LABEL_SIZE = 14         # pt at the 5 in render width (~7 pt placed, == panel title)
LEGEND_TICK_SIZE = 14          # pt at the 5 in render width (~7 pt placed, == panel title)

# Map rotation (degrees clockwise) applied to individual map images before merging.
MAP_ROTATION_DEG = 25

# Match the file year vs display year mapping used in clr_summary_plots.py
# (display year 2020 corresponds to the 2019 simulation files on disk).
DISPLAY_YEARS = (2020, 2030, 2040)
YEAR_DISPLAY_TO_FILE = {2020: 2019, 2030: 2030, 2040: 2040}
HURRICANES = ("ike", "fema33", "fema36")
HURRICANE_LABELS = {"ike": "Ike", "fema33": "FEMA33", "fema36": "FEMA36"}

# Hurricane colors (consistent with clr_summary_plots.py)
HURRICANE_COLORS = {
    "Ike": "#1f77b4",     # blue
    "FEMA33": "#2ca02c",  # green
    "FEMA36": "#d62728",  # red
}

# ----------------------------------------------------------------------------
# Custom colormaps (copied from utils/PlotGenerator.py CustomColormaps)
# ----------------------------------------------------------------------------
_PRGN_9 = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7',
           '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
_RDBU_9 = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7',
           '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
_BLUES_9 = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
            '#4292c6', '#2171b5', '#08519c', '#08306b']
_ORANGES_9 = ['#FFFFFF', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c',
              '#f16913', '#d94801', '#a63603', '#7f2704']
_REDS_9 = ['#FFFFFF', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
           '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
_ORRD_9 = ['#cccccc', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59',
           '#ef6548', '#d7301f', '#b30000', '#7f0000']
_PURPLES_9 = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8',
              '#807dba', '#6a51a3', '#54278f', '#3f007d']
_GREYS_9 = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696',
            '#737373', '#525252', '#252525', '#000000']

_CUSTOM_CMAPS = {
    'change_prgn_9': _PRGN_9,
    'change_prgn_9_r': _PRGN_9[::-1],
    'change_rdbu_9': _RDBU_9,
    'change_rdbu_9_r': _RDBU_9[::-1],
    'baseline_blues_surge': _BLUES_9,
    'baseline_oranges_debris': _ORANGES_9,
    'baseline_reds_clr': _REDS_9,
    'baseline_orrd_network': _ORRD_9,
    'uncertainty_purples': _PURPLES_9,
    'uncertainty_greys': _GREYS_9,
}


def register_custom_cmaps() -> None:
    """Register custom colormaps once (idempotent)."""
    for name, hex_colors in _CUSTOM_CMAPS.items():
        try:
            plt.colormaps.get_cmap(name)
        except ValueError:
            plt.colormaps.register(mcolors.ListedColormap(hex_colors, name=name))


# ----------------------------------------------------------------------------
# Diverging midpoint normalisation (verbatim from PlotGenerator.MidpointNormalize)
# ----------------------------------------------------------------------------
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x_in = [self.vmin, self.midpoint, self.vmax]
        y_out = [0.0, 0.5, 1.0]
        masked = np.ma.masked_invalid(value)
        return np.ma.masked_array(
            np.interp(masked.filled(np.nan), x_in, y_out, left=y_out[0], right=y_out[-1]),
            mask=masked.mask,
        )


# ----------------------------------------------------------------------------
# Font (mirrors clr_summary_plots._ensure_font)
# ----------------------------------------------------------------------------
def ensure_font(family: str = "STIX Two Text", fallback: str = "STIXGeneral",
                size_pt: int | None = None) -> None:
    if size_pt is None:
        size_pt = FONT_SIZE_NORMAL
    try:
        fm.findfont(family, fallback_to_default=False)
        plt.rcParams["font.family"] = family
    except Exception:
        try:
            fm.findfont(fallback, fallback_to_default=False)
            plt.rcParams["font.family"] = fallback
        except Exception:
            pass
    plt.rcParams["font.size"] = size_pt
    plt.rcParams["axes.labelsize"] = FONT_SIZE_NORMAL
    plt.rcParams["xtick.labelsize"] = FONT_SIZE_NORMAL
    plt.rcParams["ytick.labelsize"] = FONT_SIZE_NORMAL
    plt.rcParams["legend.fontsize"] = FONT_SIZE_SMALL
    plt.rcParams["legend.title_fontsize"] = FONT_SIZE_NORMAL


# ----------------------------------------------------------------------------
# Geographic helpers
# ----------------------------------------------------------------------------
def load_tract_gdf():
    import geopandas as gpd
    gdf = gpd.read_file(TRACT_SHP)
    if str(gdf.crs).lower() != TARGET_CRS.lower():
        gdf = gdf.to_crs(TARGET_CRS)
    if "FID" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "FID"})
    gdf["FID"] = gdf["FID"].astype(np.int64)
    if not gdf.geometry.is_valid.all():
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def load_county_gdf():
    import geopandas as gpd
    gdf = gpd.read_file(COUNTY_SHP)
    if str(gdf.crs).lower() != TARGET_CRS.lower():
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def load_network_gdf():
    import geopandas as gpd
    gdf = gpd.read_file(NETWORK_SHP).reset_index(drop=True)
    if str(gdf.crs).lower() != TARGET_CRS.lower():
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def boundary_extent(tract_gdf):
    geoms = tract_gdf.geometry
    if not geoms.is_valid.all():
        geoms = geoms.buffer(0)
    union = unary_union(geoms)
    return union.bounds  # (minx, miny, maxx, maxy)


# ----------------------------------------------------------------------------
# Map renderer matching the existing paper-figure style
# ----------------------------------------------------------------------------
def render_choropleth_map(
    gdf,
    *,
    value_col: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    midpoint: float | None,
    out_path: Path,
    legend_path: Path | None,
    legend_label: str,
    county_gdf=None,
    tract_gdf=None,
    figsize=(6, 6),
    dpi: int = DPI,
    polygon_line_width: float = 0.2,
    polygon_edge_color: str = "black",
    is_line_geometry: bool = False,
    line_width: float = 0.6,
    axis_padding_factor: float = 0.05,
    norm=None,
):
    """Render a single choropleth (polygon) or line map in the paper's style.

    figsize defaults to (6,6) to match the original paper figure generator
    (PlotGenerator.py GENERAL_SETTINGS). Do NOT change this default — it
    controls the visual weight of polygon edges and must stay consistent
    with the 36 base paper maps in generated_maps_v7d/.

    If a pre-built ``norm`` is passed (e.g. a matplotlib PowerNorm used to
    stretch a skewed, washed-out value range so year-to-year differences are
    visible), it overrides the vmin/vmax/midpoint normalisation. The same norm
    object should also be handed to save_horizontal_legend so the colorbar
    matches the map.
    """
    register_custom_cmaps()

    cmap = plt.get_cmap(cmap_name)
    try:
        cmap.set_bad(alpha=0.0)
    except Exception:
        pass

    if norm is not None:
        pass  # caller-supplied normalisation takes precedence
    elif midpoint is not None:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # plot the data
    if is_line_geometry:
        gdf.plot(
            column=value_col, cmap=cmap, norm=norm,
            linewidth=line_width, ax=ax, legend=False,
            missing_kwds={"color": "none"}, zorder=1,
        )
    else:
        gdf.plot(
            column=value_col, cmap=cmap, norm=norm,
            linewidth=polygon_line_width, edgecolor=polygon_edge_color,
            ax=ax, legend=False,
            missing_kwds={"color": "none", "edgecolor": "none"},
            zorder=1,
        )

    # set the axis extent based on the tract boundary union (not data)
    if tract_gdf is not None:
        minx, miny, maxx, maxy = boundary_extent(tract_gdf)
        pad_x = (maxx - minx) * axis_padding_factor
        pad_y = (maxy - miny) * axis_padding_factor
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    # county overlay
    if county_gdf is not None:
        county_gdf.plot(
            ax=ax, facecolor="none", edgecolor="black",
            linewidth=0.2, zorder=10,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                transparent=True, facecolor="none", edgecolor="none",
                pad_inches=0.01)
    plt.close(fig)

    if legend_path is not None:
        save_horizontal_legend(cmap, norm, legend_path, label=legend_label, dpi=dpi)


def save_figure(fig, out_path: Path, *, dpi: int = DPI, pad_inches: float = 0.03,
                also_pdf: bool = True):
    """Save fig to PNG (always) and PDF (alongside) for submission-ready outputs.

    Per ASCE Author Instructions, final figures should be TIFF/EPS/PDF with
    embedded fonts. We save PDF here using matplotlib's pdf backend, which
    embeds fonts automatically when fonttype 42 (TrueType) is requested via
    rcParams.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                transparent=True, facecolor="none", edgecolor="none",
                pad_inches=pad_inches)
    if also_pdf and out_path.suffix.lower() == ".png":
        pdf_path = out_path.with_suffix(".pdf")
        prev_pdf_ft = plt.rcParams.get("pdf.fonttype")
        prev_ps_ft = plt.rcParams.get("ps.fonttype")
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        # Cap the PDF DPI at 600. Matplotlib's PDF backend uses dpi for
        # rasterising embedded bitmaps (e.g. imshow of rotated maps). The
        # PNG dpi can be higher (we use 1200 for merged map tiles) without
        # blowing up file size, but a 1200 DPI PDF rasterises every embedded
        # bitmap at that resolution and produces 20+ MB files. 600 DPI is
        # ASCE-compliant (≥ 300 DPI required) and keeps PDFs ~5-8 MB.
        pdf_dpi = min(dpi, 600)
        try:
            fig.savefig(pdf_path, dpi=pdf_dpi, bbox_inches="tight",
                        transparent=True, facecolor="none", edgecolor="none",
                        pad_inches=pad_inches)
        finally:
            if prev_pdf_ft is not None:
                plt.rcParams["pdf.fonttype"] = prev_pdf_ft
            if prev_ps_ft is not None:
                plt.rcParams["ps.fonttype"] = prev_ps_ft
    plt.close(fig)


def save_horizontal_legend(cmap, norm, out_path: Path, label: str = "",
                           dpi: int = DPI, label_size: int = LEGEND_LABEL_SIZE,
                           tick_size: int = LEGEND_TICK_SIZE):
    """Save a transparent horizontal colorbar, matching PlotGenerator._save_legend.

    Font note: label_size/tick_size are set so that, after this (figsize width
    5 in) bitmap is downscaled to LEGEND_BAR_WIDTH_IN inches inside the merged
    panel, the colorbar text reads at ~the per-panel scenario-title size
    (FONT_SIZE_SMALL). Keeps all text in the merged image visually consistent.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_legend = plt.figure(figsize=(5, 1))
    ax_legend = fig_legend.add_axes([0.05, 0.4, 0.9, 0.2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if isinstance(norm, MidpointNormalize):
        midpoint = getattr(norm, "midpoint", None)
        if midpoint is not None and norm.vmin < midpoint < norm.vmax:
            ticks = sorted({norm.vmin, midpoint, norm.vmax})
        else:
            ticks = [norm.vmin, norm.vmax]
        extend = "both"
    elif isinstance(norm, mcolors.PowerNorm):
        # The bar is power-stretched, so two end-ticks alone make intermediate
        # data values hard to read. Place ticks at even data fractions of the
        # range (they land at non-uniform spacing on the stretched bar, which
        # is what visually communicates the stretch).
        v0, v1 = norm.vmin, norm.vmax
        ticks = [v0 + f * (v1 - v0) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        extend = "both"
    else:
        ticks = [norm.vmin, norm.vmax]
        extend = "both"

    cb = fig_legend.colorbar(sm, cax=ax_legend, orientation="horizontal",
                             extend=extend, ticks=ticks)
    if label:
        cb.set_label(label, size=label_size)
    cb.ax.tick_params(labelsize=tick_size)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2g}"))

    fig_legend.patch.set_alpha(0.0)
    fig_legend.savefig(out_path, dpi=dpi, bbox_inches="tight",
                       transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig_legend)


def make_merged_map_panel(
    png_paths: list[Path],
    labels: list[str],
    out_path: Path,
    ncols: int = 3,
    legend_path: Path | None = None,
):
    """Create a merged panel of individual map PNGs with 25-deg clockwise rotation.

    Quality strategy (revised): the previous implementation used PIL to
    pre-rotate each map bitmap (BICUBIC) and then matplotlib's imshow
    rasterised them again at savefig time --- two interpolation steps that
    visibly softened polygon edges relative to the source. The current
    implementation reads the high-DPI source bitmaps via PIL once, computes
    the smallest bounding-box of non-transparent pixels (so the merged
    figure does not waste space on the empty PNG margins around the polygon),
    applies the 25-deg clockwise rotation via matplotlib's Affine2D in
    imshow, and renders the merged figure at high DPI in a SINGLE
    rasterisation step. The PNG and PDF outputs are generated by separate
    savefig calls so neither is derived from the other.
    """
    from PIL import Image as PILImage
    from matplotlib.image import imread
    from matplotlib.transforms import Affine2D

    ensure_font()
    # DPI for the two independent saves. The PDF is the shipped supplementary
    # figure: at the previous 1200 DPI it rasterised the embedded rotated
    # bitmaps so finely that each map PDF ballooned to ~14 MB (63 MB across the
    # 5 figures). 400 DPI is ASCE-compliant (>= 300 required), keeps every map
    # legible at the 7 in page width, and brings each PDF down to a few MB.
    # The PNG is only used for on-screen QA here, so a moderate 600 suffices.
    merged_dpi_png = 600
    merged_dpi_pdf = 400

    def _load_and_tightcrop(path):
        """Load PNG into float-array, crop to non-fully-transparent bbox.

        We crop the source PNG (which is in NATIVE orientation, not rotated)
        to remove the wide transparent margins matplotlib leaves around the
        county polygon. This shrinks the source bitmap before rotation, but
        does NOT pre-rotate it -- rotation happens once via Affine2D inside
        imshow at savefig time.
        """
        with PILImage.open(str(path)) as pil_img:
            pil_img = pil_img.convert("RGBA")
            alpha = pil_img.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                pil_img = pil_img.crop(bbox)
            arr = np.array(pil_img).astype(np.float32) / 255.0
        return arr

    imgs = []
    for p in png_paths:
        if p.is_file():
            imgs.append(_load_and_tightcrop(p))
        else:
            imgs.append(None)

    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    # Compute the bounding box of the ROTATED image (in pixel units). The
    # native image is W x H; after rotation by theta the bounding-box becomes
    # (W*cos + H*sin) x (W*sin + H*cos). All images share the same shape
    # because they come from the same source domain (Galveston tract gdf),
    # so we use the first valid image.
    theta = np.deg2rad(MAP_ROTATION_DEG)
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    valid = [im for im in imgs if im is not None]
    if valid:
        # Use the largest source dimensions across all loaded images
        max_h = max(im.shape[0] for im in valid)
        max_w = max(im.shape[1] for im in valid)
        rot_w = max_w * c + max_h * s
        rot_h = max_w * s + max_h * c
        aspect = rot_h / max(rot_w, 1)
    else:
        max_h = max_w = rot_h = rot_w = 1
        aspect = 1.0

    cell_w = FIG_WIDTH_DOUBLE / ncols
    cell_h = cell_w * aspect

    # Legend sized as a small horizontal bar — width set to LEGEND_BAR_WIDTH_IN
    # so it is comfortably readable (its large render-time fonts shrink to the
    # panel-title size when placed) without dominating the figure.
    # Height comes from the legend image's native aspect ratio.
    legend_w_in = LEGEND_BAR_WIDTH_IN
    legend_gap_in = 0.04
    legend_h_in = 0.0
    if legend_path and legend_path.is_file():
        from matplotlib.image import imread as _imread_probe
        _leg_img = _imread_probe(str(legend_path))
        _aspect = _leg_img.shape[0] / max(_leg_img.shape[1], 1)
        legend_h_in = legend_w_in * _aspect

    total_h = nrows * cell_h
    if legend_path and legend_path.is_file():
        total_h += legend_h_in + legend_gap_in

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIG_WIDTH_DOUBLE, total_h),
        gridspec_kw={"hspace": 0.08, "wspace": -0.35},
    )
    fig.patch.set_alpha(0.0)

    # Lock maps to fill the figure top portion; reserve only the legend bar at
    # the very bottom. Negative hspace / wspace lets adjacent cells overlap
    # their transparent margins -- since each rotated-image cell has empty
    # corners that contain no map content, this overlap is harmless and
    # produces a tighter, more journal-quality grid.
    bottom_for_legend = (legend_h_in + legend_gap_in) / total_h if legend_path and legend_path.is_file() else 0.005
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995,
                        bottom=bottom_for_legend,
                        hspace=0.08, wspace=-0.35)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        ax.set_facecolor("none")
        ax.axis("off")

        if imgs[idx] is not None:
            img = imgs[idx]
            H, W = img.shape[0], img.shape[1]
            # Pre-rotate the image data with PIL (BICUBIC). matplotlib's
            # transformed imshow turned out to be fiddly to get
            # orientation right (sign of rotation interacts with axis
            # inversion choices). Pre-rotating with PIL produces a clean
            # bitmap whose orientation we control directly: a NEGATIVE
            # PIL rotate angle corresponds to a CLOCKWISE rotation in
            # natural (north-up) image-display orientation. We then crop
            # to the non-transparent alpha bbox so the cell shows just
            # the rotated content and we display it with `imshow`
            # (single resampling step at savefig time, at the merged
            # figure's high DPI).
            from PIL import Image as PILImage
            pil_in = PILImage.fromarray((img * 255).astype(np.uint8))
            pil_rot = pil_in.rotate(-MAP_ROTATION_DEG, expand=True,
                                    resample=PILImage.BICUBIC,
                                    fillcolor=(0, 0, 0, 0))
            # Crop to non-transparent bbox so we lose nothing but trim
            # the now-larger transparent margins around the rotated
            # diamond.
            alpha = pil_rot.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                pil_rot = pil_rot.crop(bbox)
            rot_arr = np.array(pil_rot).astype(np.float32) / 255.0
            ax.imshow(rot_arr, aspect="equal",
                      interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
        # pad=8 keeps a comfortable gap between the title (top of cell) and
        # the rotated map content of the row immediately above this one.
        ax.set_title(labels[idx], fontsize=FONT_SIZE_SMALL, pad=8)

    for idx in range(n, nrows * ncols):
        r, col = divmod(idx, ncols)
        axes[r, col].axis("off")

    if legend_path and legend_path.is_file():
        leg_img = imread(str(legend_path))
        leg_w_frac = legend_w_in / FIG_WIDTH_DOUBLE
        leg_h_frac = legend_h_in / total_h
        x0 = (1.0 - leg_w_frac) / 2
        y0 = 0.005  # snug at the bottom of the figure
        ax_leg = fig.add_axes([x0, y0, leg_w_frac, leg_h_frac])
        ax_leg.imshow(leg_img, aspect="equal", interpolation="lanczos")
        ax_leg.axis("off")

    # Render the merged figure independently to PNG (high-DPI raster) and
    # to PDF (vector composition with embedded bitmaps + transformation
    # matrix). Neither output is derived from the other; each goes through
    # its own savefig call with its own dpi target.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG: high DPI raster
    fig.savefig(out_path, dpi=merged_dpi_png, bbox_inches="tight",
                transparent=True, facecolor="none", edgecolor="none",
                pad_inches=0.02)
    # PDF: separate call with its own (more moderate) DPI for bitmap data;
    # matplotlib's PDF backend records the transform so the rotation is
    # preserved in vector form.
    pdf_path = out_path.with_suffix(".pdf")
    prev_pdf_ft = plt.rcParams.get("pdf.fonttype")
    plt.rcParams["pdf.fonttype"] = 42
    try:
        fig.savefig(pdf_path, dpi=merged_dpi_pdf, bbox_inches="tight",
                    transparent=True, facecolor="none", edgecolor="none",
                    pad_inches=0.02)
    finally:
        if prev_pdf_ft is not None:
            plt.rcParams["pdf.fonttype"] = prev_pdf_ft
    plt.close(fig)
    print(f"  merged panel: {out_path.name} (+ pdf)")
