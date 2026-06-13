"""
Training script for the new physics-informed test model.

Reads the engineered CSV produced by prepare_features.py, splits 60/20/20
with random_state=98 (matches the original paper for fair comparison),
trains a partially-monotone NN with non-negative output, and logs
everything to W&B.

Usage:
    python train.py --csv grid_static_features.csv \
                    --manifest feature_groups_2026_05_04.json \
                    --out-dir ./outputs/run_<TIMESTAMP> \
                    [--epochs 300] [--batch-size 64] [--lr 5e-4] \
                    [--wandb-project hurricane-debris-projection] \
                    [--run-name physics_v1] [--seed 98]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True,
                   help="path to the training table (training_data.csv)")
    p.add_argument("--manifest", type=str, required=True,
                   help="path to feature_groups_<DATE>.json")
    p.add_argument("--out-dir", type=str, default="./outputs/run",
                   help="directory for checkpoints / scaler / metrics")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=98)
    p.add_argument("--patience", type=int, default=30,
                   help="early-stopping patience on val loss")
    p.add_argument("--lr-patience", type=int, default=8,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wandb-project", type=str,
                   default="hurricane-debris-projection")
    p.add_argument("--wandb-key", type=str, default=None,
                   help="if set, wandb.login(key=...) is called")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-branch-stats", action="store_true", default=True,
                   help="log mean / std of each branch output every epoch")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb logging (useful for debugging)")
    p.add_argument("--log1p-target", action="store_true",
                   help="train on log1p(y); predict in log-space then expm1 to recover")
    p.add_argument("--sqrt-sample-weight", action="store_true",
                   help="weight loss by sqrt(y+1) to upweight high-volume cells")
    p.add_argument("--loss", type=str, default="huber",
                   choices=["huber", "mse", "huber_wide", "tweedie", "log_mse"],
                   help="loss function: huber (SmoothL1, beta=1, ~MAE), "
                        "huber_wide (SmoothL1, beta=200, ~MSE for typical residuals), "
                        "mse (pure MSE), "
                        "tweedie (Tweedie deviance with --tweedie-power), "
                        "log_mse (MSE in log1p space; both pred and target are "
                        "raw-scale, loss = (log1p(pred) - log1p(y))^2; "
                        "appropriate for multiplicative models with --output-activation exp)")
    p.add_argument("--tweedie-power", type=float, default=1.5,
                   help="power index p in (1, 2) for Tweedie loss; "
                        "p=1.5 is compound Poisson-Gamma (insurance/debris claims)")
    p.add_argument("--h-inc", type=int, nargs=2, default=[32, 16],
                   help="hidden dims for monotone-increasing branch")
    p.add_argument("--h-dec", type=int, nargs=2, default=[16, 8],
                   help="hidden dims for monotone-decreasing branch")
    p.add_argument("--h-free", type=int, nargs=2, default=[32, 16],
                   help="hidden dims for free / unconstrained branch")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="dropout rate in the free branch")
    p.add_argument("--output-activation", type=str, default="softplus",
                   choices=["softplus", "exp"],
                   help="output activation: softplus (additive, default) or "
                        "exp (log-linear / multiplicative; pair with --log1p-target)")
    p.add_argument("--smart-bias-init", action="store_true",
                   help="initialise the bias to mean(y_train) (raw-scale) "
                        "or log1p(mean(y_train)) (with --log1p-target). "
                        "Use with --loss mse to encourage tail fit.")
    p.add_argument("--grad-clip", type=float, default=10.0,
                   help="max-norm gradient clip; raise to e.g. 1000 to "
                        "effectively disable when training with MSE on "
                        "raw-scale targets that have a heavy upper tail.")
    return p.parse_args()


# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data loading
# ============================================================================
@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list = field(default_factory=list)
    n_inc: int = 0
    n_dec: int = 0
    n_free: int = 0


def load_splits(csv_path: Path, manifest_path: Path, seed: int) -> Splits:
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    monotone_inc = manifest["monotone_inc"]
    monotone_dec = manifest["monotone_dec"]
    free = manifest["free"]
    feature_order = monotone_inc + monotone_dec + free
    n_inc, n_dec, n_free = len(monotone_inc), len(monotone_dec), len(free)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    # match the original training: drop rows with any NaN in the inputs / target
    cols_needed = feature_order + ["volume_m3"]
    df = df.dropna(subset=cols_needed)
    print(f"  rows after dropna: {len(df)}")

    X = df[feature_order].to_numpy(dtype=np.float32)
    y = df["volume_m3"].to_numpy(dtype=np.float32).reshape(-1, 1)
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"  y stats: min={y.min():.1f}, median={np.median(y):.1f}, "
          f"mean={y.mean():.1f}, p95={np.percentile(y, 95):.1f}, max={y.max():.1f}")

    # 60 / 20 / 20 split, identical strategy to the paper
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return Splits(X_train=X_train, y_train=y_train,
                  X_val=X_val, y_val=y_val,
                  X_test=X_test, y_test=y_test,
                  feature_names=feature_order,
                  n_inc=n_inc, n_dec=n_dec, n_free=n_free)


# ============================================================================
# Metrics
# ============================================================================
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_pred": float(y_pred.mean()),
        "mean_true": float(y_true.mean()),
    }


@torch.no_grad()
def evaluate(model: PhysicsInformedDebrisNN, X: torch.Tensor, y: torch.Tensor,
             n_inc: int, n_dec: int, batch_size: int = 1024,
             device: str = "cuda", log1p_target: bool = False):
    """Run inference on X, return predictions and metrics ON THE ORIGINAL SCALE.

    If `log1p_target` is True, the model outputs log-space predictions which
    are converted back to the original scale via expm1 before computing
    metrics. The reference y is also assumed to be in original scale.
    """
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        pred = model(x_inc, x_dec, x_free).cpu().numpy()
        if log1p_target:
            # Clip log-space predictions to a numerically safe range before
            # exponentiating.  log1p(max observed) ~ 9.93 (max y ~ 20566);
            # 16 gives expm1 ~ 8.9e6 which is far above any plausible target
            # but well within float32.  This only matters for early-epoch
            # numerical sanity; well-trained models stay <= 12.
            pred = np.clip(pred, 0.0, 16.0)
            pred = np.expm1(pred)  # back to original scale
            pred = np.maximum(pred, 0.0)  # clip residual negatives from numerical noise
        preds.append(pred)
    preds = np.concatenate(preds)
    metrics = regression_metrics(y.cpu().numpy(), preds)
    return preds, metrics


# ============================================================================
# Training loop
# ============================================================================
def train(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- W&B init -----------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        run_name = args.run_name or f"physics_v1_{datetime.now():%Y%m%d_%H%M}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # ---- Data ---------------------------------------------------------
    splits = load_splits(Path(args.csv), Path(args.manifest), args.seed)

    # ---- Scale --------------------------------------------------------
    scaler = StandardScaler().fit(splits.X_train)
    X_train_s = scaler.transform(splits.X_train).astype(np.float32)
    X_val_s = scaler.transform(splits.X_val).astype(np.float32)
    X_test_s = scaler.transform(splits.X_test).astype(np.float32)

    joblib.dump(scaler, out_dir / "scaler.pkl")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"feature_names": splits.feature_names,
                   "n_inc": splits.n_inc, "n_dec": splits.n_dec, "n_free": splits.n_free,
                   "args": vars(args)}, f, indent=2)

    # ---- Model --------------------------------------------------------
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    # When training on log1p(y), initialise the model bias so that the
    # softplus output starts near log1p(mean(y_train)).  This avoids the
    # pathology where random branch outputs + zero bias give large
    # softplus(branch_sum) values that overflow when expm1'd.
    if args.log1p_target:
        mean_y_train = float(np.maximum(splits.y_train, 0).mean())
        init_b = float(np.log1p(mean_y_train))
        init_o = -3.0
        init_s = 1.0
        print(f"  log1p target: bias=log1p({mean_y_train:.1f})={init_b:.3f}, "
              f"branch init_offset={init_o} -> initial effective weights ~ "
              f"softplus({init_o})={float(np.log1p(np.exp(init_o))):.3f}")
    elif args.output_activation == "exp" and args.smart_bias_init:
        # For exp output: y = exp(raw_clip) - 1.  Want initial y ~ mean_y, so
        # raw_clip ~ log1p(mean_y), so bias ~ log1p(mean_y).
        # Use a strongly-negative branch offset so initial branches are
        # ~zero contribution -- exp output is highly sensitive to branch
        # excursions because small raw differences become large prediction
        # differences via exp.
        mean_y_train = float(np.maximum(splits.y_train, 0).mean())
        init_b = float(np.log1p(mean_y_train))
        init_o = -3.0  # effective initial weights ~ softplus(-3) ~ 0.05
        init_s = 1.0
        print(f"  exp output + smart bias init: bias=log1p({mean_y_train:.1f})="
              f"{init_b:.3f}, branch init_offset={init_o}")
    elif args.smart_bias_init:
        mean_y_train = float(np.maximum(splits.y_train, 0).mean())
        init_b = mean_y_train  # softplus(b) ~ b for b >> 0
        init_o = -1.5  # initial effective weights ~ softplus(-1.5) = 0.20
        init_s = 1.0
        print(f"  smart bias init (raw scale): bias={init_b:.1f}, "
              f"branch init_offset={init_o} -> initial effective weights ~ "
              f"softplus({init_o})={float(np.log1p(np.exp(init_o))):.3f}")
    else:
        init_b = 0.0
        init_s = 1.0
        init_o = 0.0
    model = PhysicsInformedDebrisNN(
        n_inc=splits.n_inc, n_dec=splits.n_dec, n_free=splits.n_free,
        h_inc=tuple(args.h_inc), h_dec=tuple(args.h_dec), h_free=tuple(args.h_free),
        dropout=args.dropout,
        init_bias=init_b, init_scale=init_s, init_offset=init_o,
        output_activation=args.output_activation,
    ).to(device)
    n_params = model.n_parameters()
    print(f"Model parameters: {n_params}")
    if use_wandb:
        import wandb
        wandb.config.update({"n_parameters": n_params,
                             "n_inc": splits.n_inc,
                             "n_dec": splits.n_dec,
                             "n_free": splits.n_free,
                             "n_features_total": splits.n_inc + splits.n_dec + splits.n_free,
                             "n_train": len(X_train_s),
                             "n_val": len(X_val_s),
                             "n_test": len(X_test_s)})
        wandb.watch(model, log="all", log_freq=50)

    # ---- Tensors ------------------------------------------------------
    # Always store y on the ORIGINAL scale; we transform inside the loop
    # if log1p_target is requested.
    X_train_t = torch.tensor(X_train_s)
    y_train_t = torch.tensor(splits.y_train, dtype=torch.float32).squeeze(-1)
    X_val_t = torch.tensor(X_val_s)
    y_val_t = torch.tensor(splits.y_val, dtype=torch.float32).squeeze(-1)
    X_test_t = torch.tensor(X_test_s)
    y_test_t = torch.tensor(splits.y_test, dtype=torch.float32).squeeze(-1)

    # The TARGET fed to the loss is either the raw y or log1p(y).
    if args.log1p_target:
        y_train_loss = torch.log1p(y_train_t.clamp(min=0))
    else:
        y_train_loss = y_train_t
    # Per-sample loss weights for upweighting the heavy tail.
    if args.sqrt_sample_weight:
        sample_w = torch.sqrt(y_train_t.clamp(min=0) + 1.0)
        # normalise so the average weight is 1 (keeps lr in the same regime)
        sample_w = sample_w / sample_w.mean()
    else:
        sample_w = torch.ones_like(y_train_t)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_loss, sample_w),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # ---- Optimizer / loss --------------------------------------------
    # We need per-element loss to apply sample weights, so use reduction='none'
    if args.loss == "mse":
        criterion = nn.MSELoss(reduction="none")
        print("Loss: MSE (mean-biased)")
    elif args.loss == "huber_wide":
        # Quadratic regime extends to +/- 200 (covers most of the bulk of
        # residuals on the natural scale of the target). Beyond that,
        # transitions to L1 to remain robust to extreme outliers.
        criterion = nn.SmoothL1Loss(reduction="none", beta=200.0)
        print("Loss: Huber (beta=200, mean-biased on the bulk; L1 in the tail)")
    elif args.loss == "log_mse":
        # MSE in log1p space.  pred and target are both in raw (non-negative)
        # scale; the loss compares their log1p-transforms so that *relative*
        # errors are penalised equally regardless of magnitude.  Pair with
        # --output-activation exp for a clean log-linear / multiplicative model.
        class _LogMSECrit:
            def __call__(self, pred, target):
                return (torch.log1p(torch.clamp(pred, min=0))
                        - torch.log1p(torch.clamp(target, min=0))) ** 2
        criterion = _LogMSECrit()
        print("Loss: log-MSE  ((log1p(pred) - log1p(y))^2; relative-error loss)")
    elif args.loss == "tweedie":
        # Tweedie deviance with power p in (1,2): principled loss for
        # non-negative heavy-tailed data with zero-inflation (compound
        # Poisson-Gamma).  Variance of y scales as mu^p, so high-mean cells
        # naturally get more loss weight.  Standard in actuarial /
        # claims-modelling literature (Tweedie 1984; Smyth & Jorgensen 2002).
        p_pwr = args.tweedie_power
        eps = 1e-6
        def tweedie_dev(pred, target):
            pred_safe = torch.clamp(pred, min=eps)
            target_safe = torch.clamp(target, min=eps)
            # canonical Tweedie deviance, per-element (positive)
            term1 = target * (target_safe.pow(1 - p_pwr) - pred_safe.pow(1 - p_pwr)) / (1 - p_pwr)
            term2 = (target_safe.pow(2 - p_pwr) - pred_safe.pow(2 - p_pwr)) / (2 - p_pwr)
            return 2.0 * (term1 - term2)
        # wrap in a "criterion-like" object with __call__ matching nn.MSELoss
        class _TweedieCrit:
            def __init__(self):
                pass
            def __call__(self, pred, target):
                return tweedie_dev(pred, target)
        criterion = _TweedieCrit()
        print(f"Loss: Tweedie deviance (power p={p_pwr}, "
              "principled for non-negative heavy-tailed targets with point mass at 0)")
    else:
        criterion = nn.SmoothL1Loss(reduction="none")  # default Huber
        print("Loss: SmoothL1 (Huber, beta=1, median-biased)")
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    # ---- Loop ---------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_grad_norm = 0.0
        n_batches = 0
        # for branch-output stats
        sum_h_inc = 0.0; sum_h_dec = 0.0; sum_h_free = 0.0; sum_raw = 0.0; sum_y = 0.0
        n_samples_seen = 0

        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            x_inc, x_dec, x_free = split_inputs(xb, splits.n_inc, splits.n_dec)
            optimizer.zero_grad()
            pred, br = model(x_inc, x_dec, x_free, return_branches=True)
            # criterion has reduction='none' -> per-sample loss; weight + average
            loss = (criterion(pred, yb) * wb).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            running_grad_norm += float(grad_norm)
            n_batches += 1
            n_samples_seen += xb.size(0)
            sum_h_inc += float(br["h_inc"].sum().item())
            sum_h_dec += float(br["h_dec"].sum().item())
            sum_h_free += float(br["h_free"].sum().item())
            sum_raw += float(br["raw"].sum().item())
            sum_y += float(pred.sum().item())

        # ---- Train metrics on full set (no aug) -----------------------
        train_pred, train_metrics = evaluate(
            model, X_train_t, y_train_t, splits.n_inc, splits.n_dec,
            batch_size=1024, device=device, log1p_target=args.log1p_target)
        val_pred, val_metrics = evaluate(
            model, X_val_t, y_val_t, splits.n_inc, splits.n_dec,
            batch_size=1024, device=device, log1p_target=args.log1p_target)
        test_pred, test_metrics = evaluate(
            model, X_test_t, y_test_t, splits.n_inc, splits.n_dec,
            batch_size=1024, device=device, log1p_target=args.log1p_target)

        # validation huber on the LOSS scale (log-space if log1p, else raw)
        # For ReduceLROnPlateau and early stopping we use val RMSE on the
        # RAW target scale.  RMSE penalises large residuals quadratically,
        # which is exactly what we want when the goal is to fit the heavy
        # upper tail of the distribution rather than just the bulk.
        # The Huber series is also reported for backwards-compatibility.
        train_huber = float(val_metrics["rmse"]) if False else float(train_metrics["rmse"])
        train_huber = float(train_metrics["rmse"])
        val_huber = float(val_metrics["rmse"])
        test_huber = float(test_metrics["rmse"])

        scheduler.step(val_huber)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log = {
            "epoch": epoch,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            "running/avg_loss_per_sample": running_loss / max(n_samples_seen, 1),
            "running/avg_grad_norm": running_grad_norm / max(n_batches, 1),
            "train/huber_loss": train_huber,
            "train/mae": train_metrics["mae"],
            "train/rmse": train_metrics["rmse"],
            "train/r2": train_metrics["r2"],
            "val/huber_loss": val_huber,
            "val/mae": val_metrics["mae"],
            "val/rmse": val_metrics["rmse"],
            "val/r2": val_metrics["r2"],
            "test/huber_loss": test_huber,
            "test/mae": test_metrics["mae"],
            "test/rmse": test_metrics["rmse"],
            "test/r2": test_metrics["r2"],
            "branch/h_inc_mean": sum_h_inc / max(n_samples_seen, 1),
            "branch/h_dec_mean": sum_h_dec / max(n_samples_seen, 1),
            "branch/h_free_mean": sum_h_free / max(n_samples_seen, 1),
            "branch/raw_mean": sum_raw / max(n_samples_seen, 1),
            "branch/y_mean": sum_y / max(n_samples_seen, 1),
        }

        print(f"epoch {epoch:>3} | t={epoch_time:.2f}s | lr={current_lr:.2e} | "
              f"train MAE={train_metrics['mae']:.1f}  R2={train_metrics['r2']:.3f} | "
              f"val MAE={val_metrics['mae']:.1f}  R2={val_metrics['r2']:.3f} | "
              f"test MAE={test_metrics['mae']:.1f}  R2={test_metrics['r2']:.3f}")

        if use_wandb:
            import wandb
            wandb.log(log, step=epoch)

        # ---- Early stopping ------------------------------------------
        if val_huber < best_val_loss - 1e-6:
            best_val_loss = val_huber
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_huber,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }, out_dir / "model_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best epoch was {best_epoch}, val Huber = {best_val_loss:.4f})")
                break

    total_time = time.time() - t_start
    print(f"\nTraining done in {total_time/60:.1f} minutes. "
          f"Best epoch = {best_epoch}.")

    # ---- Final eval with best checkpoint ------------------------------
    ckpt = torch.load(out_dir / "model_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    train_pred, train_metrics = evaluate(model, X_train_t, y_train_t,
                                         splits.n_inc, splits.n_dec, device=device,
                                         log1p_target=args.log1p_target)
    val_pred, val_metrics = evaluate(model, X_val_t, y_val_t,
                                     splits.n_inc, splits.n_dec, device=device,
                                     log1p_target=args.log1p_target)
    test_pred, test_metrics = evaluate(model, X_test_t, y_test_t,
                                       splits.n_inc, splits.n_dec, device=device,
                                       log1p_target=args.log1p_target)

    final = {
        "best_epoch": best_epoch,
        "total_time_min": total_time / 60.0,
        "train": train_metrics, "val": val_metrics, "test": test_metrics,
    }
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    # save predictions for the test set (for later figures)
    test_df = pd.DataFrame({
        "y_true": y_test_t.numpy(),
        "y_pred": test_pred,
        "residual": y_test_t.numpy() - test_pred,
    })
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print("\n========== FINAL METRICS (best ckpt) ==========")
    print(f"Train: MAE={train_metrics['mae']:.2f}  RMSE={train_metrics['rmse']:.2f}  R2={train_metrics['r2']:.3f}")
    print(f"Val  : MAE={val_metrics['mae']:.2f}  RMSE={val_metrics['rmse']:.2f}  R2={val_metrics['r2']:.3f}")
    print(f"Test : MAE={test_metrics['mae']:.2f}  RMSE={test_metrics['rmse']:.2f}  R2={test_metrics['r2']:.3f}")

    if use_wandb:
        import wandb
        wandb.log({"final/" + k: v for k, v in train_metrics.items()},
                  step=ckpt["epoch"])
        wandb.summary.update({
            "best_epoch": best_epoch,
            "total_time_min": total_time / 60.0,
            "final_train_mae": train_metrics["mae"],
            "final_train_r2": train_metrics["r2"],
            "final_val_mae": val_metrics["mae"],
            "final_val_r2": val_metrics["r2"],
            "final_test_mae": test_metrics["mae"],
            "final_test_r2": test_metrics["r2"],
        })

        # Predicted vs actual scatter (test)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(test_df["y_true"], test_df["y_pred"], s=4, alpha=0.4,
                       color="#4393c3", edgecolors="none")
            lim = max(test_df["y_true"].max(), test_df["y_pred"].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.6)
            ax.set_xlabel(r"Actual debris (m$^3$/cell)")
            ax.set_ylabel(r"Predicted debris (m$^3$/cell)")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            fig.tight_layout()
            fig_path = out_dir / "test_scatter.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            wandb.log({"final/test_scatter": wandb.Image(str(fig_path))})
        except Exception as e:
            print(f"Could not log scatter plot: {e}")

        wandb.finish()

    print(f"\nAll artefacts saved to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
