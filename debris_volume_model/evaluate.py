"""
Post-training evaluation for the new physics-informed test model.

Loads the saved best checkpoint, recomputes test-set metrics, and runs:

  1. Permutation importance (n_repeats=5) — same protocol used for the
     original model in 06_make_nn_sensitivity.py, so the bar charts are
     directly comparable.
  2. 1-D partial dependence for the top-K features.
  3. Test-set predicted-vs-actual scatter plot.

Outputs are written into the same directory as the checkpoint.

Usage:
    python evaluate.py --run-dir debris_volume_model/trained_model \
                       --csv debris_volume_model/training_data.csv \
                       --manifest data/feature_groups_<DATE>.json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="directory with model_best.pth, scaler.pkl, manifest.json")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--seed", type=int, default=98)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--top-k-pdp", type=int, default=5)
    return p.parse_args()


def load_test_set(csv_path, manifest_path, seed):
    with open(manifest_path) as f:
        m = json.load(f)
    feature_order = m["monotone_inc"] + m["monotone_dec"] + m["free"]
    n_inc, n_dec = len(m["monotone_inc"]), len(m["monotone_dec"])
    df = pd.read_csv(csv_path).dropna(subset=feature_order + ["volume_m3"])
    X = df[feature_order].to_numpy(dtype=np.float32)
    y = df["volume_m3"].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed)
    return feature_order, n_inc, n_dec, X_test, y_test.flatten()


@torch.no_grad()
def predict_full(model, X_scaled, n_inc, n_dec, device, batch_size=2048):
    model.eval()
    preds = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        preds.append(model(x_inc, x_dec, x_free).cpu().numpy())
    return np.concatenate(preds)


def permutation_importance(model, X_scaled, y, feature_names, n_inc, n_dec,
                           device, n_repeats=5, seed=42):
    rng = np.random.default_rng(seed)
    base = predict_full(model, X_scaled, n_inc, n_dec, device)
    base_mae = mean_absolute_error(y, base)
    print(f"  Baseline test MAE: {base_mae:.2f}")
    rows = []
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            rng.shuffle(X_perm[:, j])
            perm_pred = predict_full(model, X_perm, n_inc, n_dec, device)
            deltas.append(mean_absolute_error(y, perm_pred) - base_mae)
        rows.append({"feature": name,
                     "importance_mean": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas))})
    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    return df, base_mae


def partial_dependence(model, X_scaled, feat_idx, n_inc, n_dec, device,
                        n_grid=40):
    grid = np.linspace(np.percentile(X_scaled[:, feat_idx], 5),
                       np.percentile(X_scaled[:, feat_idx], 95), n_grid)
    pdp = np.empty(n_grid, dtype=np.float32)
    base = X_scaled.copy()
    for k, v in enumerate(grid):
        base[:, feat_idx] = v
        pdp[k] = predict_full(model, base, n_inc, n_dec, device).mean()
    return grid, pdp


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    print("Loading test split ...")
    feature_order, n_inc, n_dec, X_test, y_test = load_test_set(
        args.csv, args.manifest, args.seed)
    print(f"  test size: {X_test.shape[0]}")
    n_free = len(feature_order) - n_inc - n_dec

    scaler = joblib.load(run_dir / "scaler.pkl")
    X_test_s = scaler.transform(X_test).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = PhysicsInformedDebrisNN(n_inc=n_inc, n_dec=n_dec, n_free=n_free).to(device)
    ckpt = torch.load(run_dir / "model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded best ckpt from epoch {ckpt['epoch']}")

    # ---- Test predictions / metrics -------------------------------
    pred = predict_full(model, X_test_s, n_inc, n_dec, device)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    print(f"\nTest metrics: MAE={mae:.2f}  RMSE={rmse:.2f}  R^2={r2:.3f}")
    pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_csv(
        run_dir / "test_predictions_eval.csv", index=False)

    # ---- Permutation importance -----------------------------------
    print("\nPermutation importance ...")
    df_imp, base_mae = permutation_importance(model, X_test_s, y_test,
                                               feature_order, n_inc, n_dec,
                                               device, n_repeats=args.n_repeats)
    df_imp.to_csv(run_dir / "permutation_importance.csv", index=False)
    print("\nTop 10 features by permutation importance:")
    print(df_imp.head(10).to_string(index=False))

    # bar chart (top 15)
    top = df_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.25 * len(top) + 1.0))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="#1b7837", alpha=0.85, edgecolor="black", linewidth=0.3,
            error_kw=dict(ecolor="#444444", lw=0.5))
    ax.set_xlabel(r"Permutation importance ($\Delta$MAE; m$^3$)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(run_dir / "permutation_importance.png", dpi=300,
                bbox_inches="tight", transparent=True, facecolor="none")
    plt.close(fig)

    # ---- 1-D partial dependence for top-K --------------------------
    print(f"\nPartial dependence (top {args.top_k_pdp}) ...")
    top_features = df_imp.head(args.top_k_pdp)["feature"].tolist()
    for feat in top_features:
        idx = feature_order.index(feat)
        grid, pdp = partial_dependence(model, X_test_s, idx, n_inc, n_dec, device)
        # Convert grid back to original units
        mean = scaler.mean_[idx]; sigma = scaler.scale_[idx]
        grid_orig = grid * sigma + mean
        pd.DataFrame({"x_scaled": grid, "x_original": grid_orig,
                      "pdp_mean_pred_m3": pdp}).to_csv(
            run_dir / f"pdp_{feat}.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.5, 2.6))
        fig.patch.set_alpha(0.0); ax.set_facecolor("none")
        ax.plot(grid_orig, pdp, color="#1b7837", linewidth=1.4)
        ax.set_xlabel(feat)
        ax.set_ylabel(r"Mean predicted debris (m$^3$/cell)")
        ax.grid(linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(run_dir / f"pdp_{feat}.png", dpi=300,
                    bbox_inches="tight", transparent=True, facecolor="none")
        plt.close(fig)
        print(f"  saved pdp_{feat}.png")

    # ---- Final summary ---------------------------------------------
    summary = {
        "test_mae": float(mae), "test_rmse": rmse, "test_r2": float(r2),
        "best_epoch": int(ckpt["epoch"]),
        "n_test": int(X_test.shape[0]),
        "top10_features": df_imp.head(10).to_dict(orient="records"),
    }
    with open(run_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {run_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
