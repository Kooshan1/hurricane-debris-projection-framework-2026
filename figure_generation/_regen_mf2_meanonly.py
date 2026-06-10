"""One-off: regenerate the merged M-F2 trajectory figure from the existing
aggregated CSVs, using the (now band-free) volume panel in 03_make_trajectory_plot.

The raw per-cell volume predictions are no longer on disk, so we read the
previously-saved aggregate CSVs rather than recomputing from the pipeline.
"""
import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mf2", str(HERE / "03_make_trajectory_plot.py"))
mf2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf2)

vol = pd.read_csv(mf2.OUT_DIR / "trajectory_total_debris_volume_data.csv")
clr = pd.read_csv(mf2.OUT_DIR / "trajectory_mean_clr_data.csv")

out = mf2.OUT_DIR / "M-F2_trajectory_merged.png"
mf2._plot_merged(vol, clr, out)
print("regenerated:", out, "and", out.with_suffix(".pdf"))
