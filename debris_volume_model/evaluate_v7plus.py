"""
Paper-quality evaluation pipeline for the new physics-informed debris-volume
model.  Reads a trained run directory (with model_best.pth, scaler.pkl,
manifest.json) and produces:

  1. Bootstrap CIs on test MAE / RMSE / R^2 (n_bootstrap = 1000)
  2. Per-volume-bin MAE table (which y-magnitude regime is the model worst at?)
  3. Test-set predicted-vs-actual scatter (calibrated and uncalibrated)
  4. Residual histogram + residual vs predicted plot
  5. Permutation importance (n_repeats = 10) bar chart
  6. Feature-group importance (multi-hazard / built-env / nat-env / socio)
  7. 1-D partial dependence for top-K features
  8. Cell-level MAE map (CSV with FID + |y_true - y_pred|) for downstream mapping

Usage:
    python evaluate_v7plus.py --run-dir outputs/run_v7 \
        --csv engineered_input_2026_05_04.csv \
        --manifest feature_groups_v3.json \
        [--no-calibrate] [--top-k-pdp 5] [--n-bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--seed", type=int, default=98)
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the post-hoc multiplicative calibration")
    p.add_argument("--top-k-pdp", type=int, default=5)
    p.add_argument("--perm-repeats", type=int, default=10)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    return p.parse_args()


# ============================================================================
# Helpers
# ============================================================================
def _load_args_from_run(run_dir: Path) -> dict:
    """Read the saved arg manifest from the training run."""
    with open(run_dir / "manifest.json") as f:
        m = json.load(f)
    return m.get("args", {})


def load_model_from_run(run_dir: Path, n_inc: int, n_dec: int, n_free: int,
                        device: str):
    """Reconstruct the trained model with the *same* architecture used at
    training time (h_inc / h_dec / h_free), then load the checkpoint."""
    saved_args = _load_args_from_run(run_dir)
    h_inc = tuple(saved_args.get("h_inc", [32, 16]))
    h_dec = tuple(saved_args.get("h_dec", [16, 8]))
    h_free = tuple(saved_args.get("h_free", [32, 16]))
    dropout = saved_args.get("dropout", 0.1)
    print(f"  reconstructing model: h_inc={h_inc}, h_dec={h_dec}, "
          f"h_free={h_free}, dropout={dropout}")
    model = PhysicsInformedDebrisNN(
        n_inc=n_inc, n_dec=n_dec, n_free=n_free,
        h_inc=h_inc, h_dec=h_dec, h_free=h_free,
        dropout=dropout,
    ).to(device)
    ckpt = torch.load(run_dir / "model_best.pth", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, saved_args


def load_test_split(csv_path: Path, manifest_path: Path, seed: int):
    with open(manifest_path) as f:
        m = json.load(f)
    feature_order = m["monotone_inc"] + m["monotone_dec"] + m["free"]
    n_inc, n_dec = len(m["monotone_inc"]), len(m["monotone_dec"])
    df = pd.read_csv(csv_path).dropna(subset=feature_order + ["volume_m3"])
    X = df[feature_order].to_numpy(dtype=np.float32)
    y = df["volume_m3"].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    return feature_order, n_inc, n_dec, X_train, y_train, X_val, y_val, X_test, y_test


@torch.no_grad()
def predict(model, X_scaled, n_inc, n_dec, device, batch_size=4096):
    model.eval()
    preds = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        preds.append(model(x_inc, x_dec, x_free).cpu().numpy())
    return np.concatenate(preds)


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_pred": float(y_pred.mean()),
        "mean_true": float(y_true.mean()),
    }


# ============================================================================
# (1) Bootstrap CIs on test metrics
# ============================================================================
def bootstrap_test_metrics(y_true, y_pred, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(y_true[idx], y_pred[idx])
        rows.append(m)
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# (2) Per-volume-bin MAE
# ============================================================================
def per_bin_mae(y_true, y_pred,
                bins=(0, 1, 100, 500, 1000, 2500, 5000, 10000, np.inf)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n == 0:
            mae = np.nan; rmse = np.nan; mp = np.nan; mt = np.nan
        else:
            mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            mp = float(y_pred[mask].mean())
            mt = float(y_true[mask].mean())
        rows.append({
            "bin": f"[{lo:.0f}, {hi:.0f})",
            "n": n,
            "mean_true": mt,
            "mean_pred": mp,
            "mae": mae,
            "rmse": rmse,
        })
    return pd.DataFrame(rows)


# ============================================================================
# (3) Test scatter
# ============================================================================
def plot_test_scatter(y_true, y_pred, out_path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, color="#4393c3", edgecolors="none")
    lim = float(max(y_true.max(), y_pred.max()) * 1.05)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.6)
    ax.set_xlabel(r"Actual debris (m$^3$/cell)")
    ax.set_ylabel(r"Predicted debris (m$^3$/cell)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    if title_suffix:
        ax.set_title(title_suffix, fontsize=9)
    fig.tight_layout()
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# (4) Residual histogram + residual vs predicted
# ============================================================================
def plot_residuals(y_true, y_pred, out_dir: Path):
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(res, bins=80, color="#5aae61", edgecolor="black", linewidth=0.3)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel(r"Residual = actual $-$ predicted (m$^3$)")
    axes[0].set_ylabel("Count")
    axes[1].scatter(y_pred, res, s=6, alpha=0.35, color="#5aae61", edgecolors="none")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel(r"Predicted (m$^3$)")
    axes[1].set_ylabel(r"Residual (m$^3$)")
    for ax in axes:
        ax.grid(linestyle=":", alpha=0.35)
        ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# (5) Permutation importance
# ============================================================================
def permutation_importance(model, X_scaled, y, feature_names, n_inc, n_dec,
                            device, n_repeats=10, seed=42, calibration=1.0):
    rng = np.random.default_rng(seed)
    base = predict(model, X_scaled, n_inc, n_dec, device) * calibration
    base_mae = mean_absolute_error(y, base)
    rows = []
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            rng.shuffle(X_perm[:, j])
            perm_pred = predict(model, X_perm, n_inc, n_dec, device) * calibration
            deltas.append(mean_absolute_error(y, perm_pred) - base_mae)
        rows.append({"feature": name,
                     "importance_mean": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas))})
    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    return df, base_mae


def feature_group_importance(model, X_scaled, y, feature_order,
                              n_inc, n_dec, manifest, device, n_repeats=10,
                              seed=42, calibration=1.0):
    """Joint permutation of all features in a group (more robust than 1-D)."""
    monotone_inc = manifest["monotone_inc"]
    monotone_dec = manifest["monotone_dec"]
    free = manifest["free"]
    # Map to the four conceptual groups used in the paper
    hazard = ["SD", "WH", "WaveD", "WV_X", "WV_Y", "WindD", "WS", "MF",
              "WPF_1", "WPF_2"]
    built = ["NumB", "mean_bldg_size_m2", "NAS", "TAAS", "NumMH"]
    natural = ["ME", "dist_to_coast_m", "OW", "DO", "DM", "DT", "RD",
               "ADS", "UL"]
    socio = ["PD", "MHI", "PR", "NumHU", "OHU", "VHU", "NumH"]
    groups = {
        "Multi-hazard intensity": hazard,
        "Built environment": built,
        "Natural environment": natural,
        "Human / socioeconomic": socio,
    }
    name_to_idx = {n: i for i, n in enumerate(feature_order)}
    rng = np.random.default_rng(seed)
    base = predict(model, X_scaled, n_inc, n_dec, device) * calibration
    base_mae = mean_absolute_error(y, base)
    rows = []
    for gname, gfeats in groups.items():
        idxs = [name_to_idx[f] for f in gfeats if f in name_to_idx]
        if not idxs:
            continue
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            # joint permutation over the group: same row reorder for all gfeats
            new_order = rng.permutation(len(X_perm))
            X_perm[:, idxs] = X_perm[new_order, :][:, idxs]
            perm_pred = predict(model, X_perm, n_inc, n_dec, device) * calibration
            deltas.append(mean_absolute_error(y, perm_pred) - base_mae)
        rows.append({"group": gname, "n_features": len(idxs),
                     "importance_mean": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def plot_perm_importance(df_imp: pd.DataFrame, out_path: Path,
                          title: str = "", top_k: int = 15, color="#1b7837"):
    df = df_imp.head(top_k).iloc[::-1]
    label_col = "feature" if "feature" in df.columns else "group"
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(df) + 1.0))
    ax.barh(df[label_col], df["importance_mean"], xerr=df["importance_std"],
            color=color, alpha=0.85, edgecolor="black", linewidth=0.3,
            error_kw=dict(ecolor="#444444", lw=0.5))
    ax.set_xlabel(r"Permutation importance ($\Delta$MAE; m$^3$)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# (7) Partial dependence
# ============================================================================
def partial_dependence(model, X_scaled, feat_idx, n_inc, n_dec, device,
                        n_grid=40, calibration=1.0):
    grid = np.linspace(np.percentile(X_scaled[:, feat_idx], 5),
                       np.percentile(X_scaled[:, feat_idx], 95), n_grid)
    pdp = np.empty(n_grid, dtype=np.float32)
    base = X_scaled.copy()
    for k, v in enumerate(grid):
        base[:, feat_idx] = v
        pdp[k] = (predict(model, base, n_inc, n_dec, device) * calibration).mean()
    return grid, pdp


def plot_pdp(grid_orig, pdp, feature_name, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(grid_orig, pdp, color="#1b7837", linewidth=1.4)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(r"Mean predicted debris (m$^3$/cell)")
    ax.grid(linestyle=":", alpha=0.35)
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data + model from {run_dir} ...")
    feature_order, n_inc, n_dec, X_tr, y_tr, X_va, y_va, X_te, y_te = \
        load_test_split(Path(args.csv), Path(args.manifest), args.seed)
    n_free = len(feature_order) - n_inc - n_dec

    scaler = joblib.load(run_dir / "scaler.pkl")
    X_te_s = scaler.transform(X_te).astype(np.float32)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt, saved_args = load_model_from_run(run_dir, n_inc, n_dec, n_free, device)
    print(f"  best epoch = {ckpt['epoch']}")

    # calibration
    cal = 1.0
    if not args.no_calibrate:
        with open(run_dir / "final_metrics.json") as f:
            fm = json.load(f)
        cal = fm["train"]["mean_true"] / max(fm["train"]["mean_pred"], 1e-6)
        print(f"  calibration constant c = {cal:.4f}")

    # Test predictions
    y_te_flat = y_te.flatten()
    pred_te_raw = predict(model, X_te_s, n_inc, n_dec, device)
    pred_te = pred_te_raw * cal
    metrics_raw = regression_metrics(y_te_flat, pred_te_raw)
    metrics = regression_metrics(y_te_flat, pred_te)
    print(f"  raw test:        MAE={metrics_raw['mae']:.2f} R2={metrics_raw['r2']:.3f}")
    print(f"  calibrated test: MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

    # 1. Bootstrap CIs (calibrated)
    print(f"\nBootstrap CIs (n={args.n_bootstrap}) ...")
    bs = bootstrap_test_metrics(y_te_flat, pred_te, n_boot=args.n_bootstrap)
    ci = bs.quantile([0.025, 0.5, 0.975]).T
    ci.columns = ["q025", "median", "q975"]
    ci.to_csv(out_dir / "test_bootstrap_cis.csv")
    print(ci)

    # 2. Per-bin MAE
    print("\nPer-volume-bin MAE ...")
    bin_df = per_bin_mae(y_te_flat, pred_te)
    bin_df.to_csv(out_dir / "per_bin_mae.csv", index=False)
    print(bin_df.to_string(index=False))

    # 3. Scatter plots
    plot_test_scatter(y_te_flat, pred_te_raw, out_dir / "test_scatter_uncalibrated.png",
                      title_suffix=f"raw  R2={metrics_raw['r2']:.3f}  MAE={metrics_raw['mae']:.0f}")
    plot_test_scatter(y_te_flat, pred_te, out_dir / "test_scatter_calibrated.png",
                      title_suffix=f"calibrated  c={cal:.2f}  R2={metrics['r2']:.3f}  MAE={metrics['mae']:.0f}")

    # 4. Residuals (calibrated)
    plot_residuals(y_te_flat, pred_te, out_dir)

    # 5. Permutation importance (per-feature)
    print(f"\nPermutation importance (n_repeats={args.perm_repeats}) ...")
    df_imp, base_mae = permutation_importance(
        model, X_te_s, y_te_flat, feature_order, n_inc, n_dec, device,
        n_repeats=args.perm_repeats, calibration=cal)
    df_imp.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_perm_importance(df_imp, out_dir / "permutation_importance.png",
                          title="Per-feature permutation importance",
                          top_k=15, color="#1b7837")
    print(df_imp.head(10).to_string(index=False))

    # 6. Group-level importance
    with open(args.manifest) as f:
        manifest = json.load(f)
    print("\nFeature-group importance ...")
    df_grp = feature_group_importance(model, X_te_s, y_te_flat, feature_order,
                                       n_inc, n_dec, manifest, device,
                                       n_repeats=args.perm_repeats,
                                       calibration=cal)
    df_grp.to_csv(out_dir / "group_importance.csv", index=False)
    plot_perm_importance(df_grp, out_dir / "group_importance.png",
                          title="Feature-group importance",
                          top_k=10, color="#762a83")
    print(df_grp.to_string(index=False))

    # 7. PDPs for top-K features
    top_features = df_imp.head(args.top_k_pdp)["feature"].tolist()
    print(f"\nPDP for top {args.top_k_pdp} features: {top_features}")
    for feat in top_features:
        idx = feature_order.index(feat)
        grid, pdp = partial_dependence(model, X_te_s, idx, n_inc, n_dec, device,
                                        calibration=cal)
        grid_orig = grid * scaler.scale_[idx] + scaler.mean_[idx]
        pd.DataFrame({"x_orig": grid_orig, "pdp": pdp}).to_csv(
            out_dir / f"pdp_{feat}.csv", index=False)
        plot_pdp(grid_orig, pdp, feat, out_dir / f"pdp_{feat}.png")

    # Final metrics summary
    summary = {
        "best_epoch": int(ckpt["epoch"]),
        "calibration_factor": float(cal),
        "test_metrics_raw": metrics_raw,
        "test_metrics_calibrated": metrics,
        "test_bootstrap_ci_mae_95": [float(ci.loc["mae", "q025"]),
                                      float(ci.loc["mae", "q975"])],
        "test_bootstrap_ci_r2_95": [float(ci.loc["r2", "q025"]),
                                     float(ci.loc["r2", "q975"])],
        "top10_feature_importance": df_imp.head(10).to_dict(orient="records"),
        "group_importance": df_grp.to_dict(orient="records"),
        "training_args": saved_args,
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll evaluation outputs in {out_dir}")


if __name__ == "__main__":
    main()
