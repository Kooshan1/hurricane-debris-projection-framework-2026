"""
Step 6: S-F3 — sensitivity of the trained NN to its inputs.

Outputs (no model re-training; pure inference on the existing checkpoint):

    S-F3_nn_sensitivity/
        permutation_importance.png        — bar chart of MAE-degradation
        partial_dependence_<feature>.png  — one PDP per top-K feature
        permutation_importance_data.csv
        partial_dependence_<feature>_data.csv

The test split is reproduced exactly using the same random_state (=98) used
in train_new_NN_model_v1.3.0.py, so the importance figures are computed on
the same held-out 20% the paper reports MAE/MSE/R^2 for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import OUTPUTS_ROOT, REVISION_FIG_ROOT, ensure_font

OUT_DIR = REVISION_FIG_ROOT / "S-F3_nn_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUTS_ROOT / "trained_debris_volume_model" / "V13_NN_all_parameters"
TRAIN_DATA_CSV = (OUTPUTS_ROOT / "final_input_for_debris_volume_model" / "ike"
                  / "2008" / "final_input_parameters.csv")
GRID_SHP_Y = OUTPUTS_ROOT / "debris_volume" / "Grid_250_with_debris_volume.shp"

DROPOUT_RATIO = 0.5
RANDOM_STATE = 98  # matches train_new_NN_model_v1.3.0.py


# ---------------- Model ----------------
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_weights = torch.tanh(self.attention(x))
        attention_weights = self.context_vector(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return torch.sum(x * attention_weights, dim=1)


class DebrisVolumeNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.attention = Attention(256, 128)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATIO)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        return self.fc5(x)


# ---------------- Data prep (mirrors train script exactly) ----------------
def _build_dataset():
    """Reproduce the exact train/val/test split used during training."""
    print("Loading training input + ground-truth volumes ...")
    df_new = pd.read_csv(TRAIN_DATA_CSV)
    gdf_y = gpd.read_file(GRID_SHP_Y)
    df_new["FID"] = df_new["FID"].astype(int)
    gdf_y["FID"] = gdf_y["FID"].astype(int)

    df_merged = pd.merge(df_new, gdf_y[["FID", "volume_m3", "geometry"]],
                         on="FID", how="left", suffixes=("_df", "_gdf"))
    geom_col = "geometry_gdf" if "geometry_gdf" in df_merged.columns else "geometry"
    gdf_merged = gpd.GeoDataFrame(df_merged, geometry=geom_col)
    gdf_merged["centroid_x"] = gdf_merged.geometry.centroid.x
    gdf_merged["centroid_y"] = gdf_merged.geometry.centroid.y

    df_filtered = gdf_merged.dropna()

    # Use the exact feature list saved by the training script. The checkpoint
    # was trained on the full 32-feature set (OW and DO included), so no
    # columns are dropped here.
    feature_names = joblib.load(MODEL_DIR / "features_used.pkl")
    print(f"  using saved feature list ({len(feature_names)} features)")
    X = df_filtered[feature_names].to_numpy(dtype=np.float32)
    y = df_filtered["volume_m3"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    return feature_names, X_train, X_val, X_test, y_train, y_val, y_test


# ---------------- Inference helpers ----------------
def _predict(model, X_scaled, batch_size=1024) -> np.ndarray:
    """Run the model in eval mode on a (already scaled) numpy matrix."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32)
            out = model(batch).cpu().numpy().flatten()
            preds.append(np.maximum(out, 0))
    return np.concatenate(preds)


# ---------------- Permutation importance ----------------
def permutation_importance(model, X_scaled, y_true, feature_names,
                           n_repeats: int = 10, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    base_pred = _predict(model, X_scaled)
    base_mae = mean_absolute_error(y_true.flatten(), base_pred)
    print(f"Baseline MAE on test set: {base_mae:.2f}")

    importances = np.zeros((len(feature_names), n_repeats))
    for j, name in enumerate(feature_names):
        for r in range(n_repeats):
            X_perm = X_scaled.copy()
            rng.shuffle(X_perm[:, j])  # in-place shuffle along axis 0
            perm_pred = _predict(model, X_perm)
            importances[j, r] = mean_absolute_error(y_true.flatten(), perm_pred) - base_mae
    out = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importances.mean(axis=1),
        "importance_std": importances.std(axis=1),
    }).sort_values("importance_mean", ascending=False)
    return out, base_mae


def plot_permutation_importance(df_imp: pd.DataFrame, base_mae: float,
                                 out_path: Path, top_k: int = 15):
    ensure_font()
    df = df_imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.25 * len(df) + 1.0))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.barh(df["feature"], df["importance_mean"],
            xerr=df["importance_std"], color="#6a51a3", alpha=0.85,
            edgecolor="black", linewidth=0.3,
            error_kw=dict(ecolor="#444444", lw=0.5))
    ax.set_xlabel(rf"Permutation importance ($\Delta$MAE; m$^{{3}}$)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ---------------- Partial dependence ----------------
def partial_dependence(model, X_scaled, feature_idx: int,
                        n_grid: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """1-D PDP: vary feature j across percentiles 5-95, hold others fixed."""
    grid = np.linspace(np.nanpercentile(X_scaled[:, feature_idx], 5),
                       np.nanpercentile(X_scaled[:, feature_idx], 95), n_grid)
    pdp = np.empty(n_grid, dtype=np.float32)
    base_X = X_scaled.copy()
    for k, v in enumerate(grid):
        base_X[:, feature_idx] = v
        pdp[k] = float(np.mean(_predict(model, base_X)))
    return grid, pdp


def plot_pdp(grid_scaled, pdp, feature_name, scaler, feature_idx, out_path: Path):
    """Convert the scaled grid back to the original units before plotting."""
    ensure_font()
    mean = scaler.mean_[feature_idx]
    scale = scaler.scale_[feature_idx]
    grid_orig = grid_scaled * scale + mean

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.plot(grid_orig, pdp, color="#6a51a3", linewidth=1.4)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(r"Mean predicted debris (m$^{3}$/cell)")
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


def main():
    print("Reproducing train/val/test split ...")
    feature_names, X_train, X_val, X_test, y_train, y_val, y_test = _build_dataset()
    print(f"  feature count = {len(feature_names)};  test shape = {X_test.shape}")

    # Load the same scaler that the training script used
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Load the model
    print("Loading trained NN checkpoint ...")
    model = DebrisVolumeNN(input_size=X_test.shape[1]).cpu()
    state = torch.load(MODEL_DIR / "nn_model_epoch_100.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Permutation importance
    print("Computing permutation importance (n_repeats=5) ...")
    df_imp, base_mae = permutation_importance(model, X_test_scaled, y_test,
                                              feature_names, n_repeats=5,
                                              rng_seed=42)
    df_imp.to_csv(OUT_DIR / "permutation_importance_data.csv", index=False)
    plot_permutation_importance(df_imp, base_mae,
                                 OUT_DIR / "permutation_importance.png",
                                 top_k=15)
    print("  saved permutation_importance.png")
    print("\nTop 10 features by permutation importance:")
    print(df_imp.head(10).to_string(index=False))

    # Partial dependence plots for top-3 features
    top3 = df_imp.head(3)["feature"].tolist()
    for feat in top3:
        idx = feature_names.index(feat)
        grid, pdp = partial_dependence(model, X_test_scaled, idx, n_grid=40)
        pd.DataFrame({"x_scaled": grid,
                      "x_original": grid * scaler.scale_[idx] + scaler.mean_[idx],
                      "pdp_mean_predicted_m3": pdp}).to_csv(
            OUT_DIR / f"partial_dependence_{feat}_data.csv", index=False)
        plot_pdp(grid, pdp, feat, scaler, idx,
                  OUT_DIR / f"partial_dependence_{feat}.png")
        print(f"  saved partial_dependence_{feat}.png")

    print(f"\nS-F3 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
