"""
Partially-monotone, physics-informed neural network for debris-volume prediction.

Architecture rationale
----------------------
Input features are split into three blocks:

    monotone-INCREASING (n_inc=11):
        SD, WH, WS, MF, WPF_1, WPF_2,
        NumB, mean_bldg_size_m2, NAS, TAAS, NumMH
        -- physically, debris should be NON-DECREASING in each of these.

    monotone-DECREASING (n_dec=2):
        ME (minimum elevation), dist_to_coast_m
        -- physically, debris should be NON-INCREASING in each.

    free (n_free=18):
        directional / categorical / socioeconomic features -- no monotonicity
        prior is asserted.

Each block is a small 2-hidden-layer MLP that produces a single scalar.
The final prediction is

    y = softplus( bias + h_inc + h_dec + h_free )

`softplus` enforces non-negativity (debris volume >= 0).

Monotonicity is enforced by construction:
  - all weights in the inc / dec branches are computed as `softplus(raw_w)`,
    so they are strictly positive,
  - all activations are non-decreasing (ReLU),
  - the dec branch additionally negates its input before the network and
    negates its output, so it is monotone non-increasing in the original
    inputs.

The free branch has no constraint.

Author note for paper justification
-----------------------------------
This is a *constrained* neural network -- the constraints are (i) positive
weights for the monotone branches and (ii) softplus output activation.
These are standard, well-cited techniques (Lang 2005; Wehenkel & Louppe
2019; You et al. 2017 "Deep Lattice Networks") and the resulting model is
GUARANTEED to satisfy the stated physical priors regardless of training
outcome.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier(in_dim, out_dim, scale: float = 1.0):
    """Xavier-uniform initialisation (returns a Parameter).

    The optional `scale` factor shrinks the initial weights so the network's
    *initial* output magnitude is small.  This is critical when training on
    a log-transformed target (log1p), where unconstrained initial outputs
    in the 30+ range expm1 to >1e13 and produce numerical chaos.
    """
    bound = (6.0 / (in_dim + out_dim)) ** 0.5 * scale
    return nn.Parameter((torch.rand(in_dim, out_dim) * 2 - 1) * bound)


class MonotoneIncreasingMLP(nn.Module):
    """Two-hidden-layer MLP with non-negative weights and ReLU activations.

    Output is monotone non-decreasing in each input.
    Output dim is 1 (scalar contribution to the additive output).

    `init_scale` shrinks initial weights, and `init_offset` shifts the raw
    weights NEGATIVE so that softplus(raw_w + offset) is small at init.
    Use init_offset=-3 for log-space training -- this gives effective
    initial weights of softplus(-3) ~ 0.05, so the network's initial
    output is dominated by the bias rather than by random branch outputs.
    """
    def __init__(self, n_in: int, h1: int = 16, h2: int = 8,
                 init_scale: float = 1.0, init_offset: float = 0.0):
        super().__init__()
        # Raw (unconstrained) parameters; we apply softplus in forward()
        # to obtain the actual non-negative weights.
        self.raw_w1 = _xavier(n_in, h1, scale=init_scale)
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.raw_w2 = _xavier(h1, h2, scale=init_scale)
        self.b2 = nn.Parameter(torch.zeros(h2))
        self.raw_w3 = _xavier(h2, 1, scale=init_scale)
        self.b3 = nn.Parameter(torch.zeros(1))
        if init_offset != 0.0:
            with torch.no_grad():
                self.raw_w1.add_(init_offset)
                self.raw_w2.add_(init_offset)
                self.raw_w3.add_(init_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = F.softplus(self.raw_w1)
        w2 = F.softplus(self.raw_w2)
        w3 = F.softplus(self.raw_w3)
        h = F.relu(x @ w1 + self.b1)
        h = F.relu(h @ w2 + self.b2)
        return (h @ w3 + self.b3).squeeze(-1)  # (B,)


class MonotoneDecreasingMLP(nn.Module):
    """Wraps MonotoneIncreasingMLP to be monotone non-INCREASING in input.

    Trick: f(x) = -MonotoneIncreasingMLP(x).  Negating the *output* of
    a monotone-non-decreasing function gives a monotone-non-increasing
    function.  (Note: negating BOTH input and output cancels and gives a
    monotone-non-decreasing function -- a common sign-flip pitfall.)
    """
    def __init__(self, n_in: int, h1: int = 8, h2: int = 4,
                 init_scale: float = 1.0, init_offset: float = 0.0):
        super().__init__()
        self.inner = MonotoneIncreasingMLP(n_in, h1=h1, h2=h2,
                                            init_scale=init_scale,
                                            init_offset=init_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.inner(x)


class FreeMLP(nn.Module):
    """Standard unconstrained 2-hidden-layer MLP, scalar output."""
    def __init__(self, n_in: int, h1: int = 16, h2: int = 8, dropout: float = 0.1,
                 init_scale: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, 1)
        self.drop = nn.Dropout(dropout)
        # If init_scale < 1, shrink the linear weights so the initial
        # output magnitude is small (matters for log-space training).
        if init_scale != 1.0:
            with torch.no_grad():
                self.fc1.weight.mul_(init_scale)
                self.fc2.weight.mul_(init_scale)
                self.fc3.weight.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.drop(F.relu(self.bn2(self.fc2(h))))
        return self.fc3(h).squeeze(-1)


class PhysicsInformedDebrisNN(nn.Module):
    """Partially-monotone, physics-informed debris-volume regressor."""

    def __init__(self, n_inc: int, n_dec: int, n_free: int,
                 h_inc=(32, 16), h_dec=(16, 8), h_free=(32, 16),
                 dropout: float = 0.1, init_bias: float = 0.0,
                 init_scale: float = 1.0, init_offset: float = 0.0,
                 output_activation: str = "softplus",
                 exp_output_clip: float = 15.0):
        """
        output_activation: "softplus" (additive, default) or "exp" (log-linear /
            multiplicative).  With "exp", the network output is exp(branches+bias)
            -- the model becomes a GLM with log link, giving multiplicative
            dynamics suitable for non-negative skewed targets.  Use with a
            log-space loss (MSE on log1p target) for stability.
        exp_output_clip: when output_activation="exp", clip the pre-exp value
            to [0, exp_output_clip] to prevent overflow.  Default 15 means
            max y ~ exp(15) = 3.27e6 (well above any plausible debris cell).
        """
        super().__init__()
        self.n_inc, self.n_dec, self.n_free = n_inc, n_dec, n_free
        self.branch_inc = MonotoneIncreasingMLP(n_inc, *h_inc,
                                                 init_scale=init_scale,
                                                 init_offset=init_offset)
        self.branch_dec = MonotoneDecreasingMLP(n_dec, *h_dec,
                                                 init_scale=init_scale,
                                                 init_offset=init_offset)
        self.branch_free = FreeMLP(n_free, *h_free, dropout=dropout,
                                    init_scale=init_scale)
        # init_bias should equal log1p(target_mean) when training on log1p
        # target, so the network starts with sensible-magnitude predictions.
        self.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))
        assert output_activation in ("softplus", "exp"), \
            f"unknown output_activation: {output_activation}"
        self.output_activation = output_activation
        self.exp_output_clip = exp_output_clip

    def forward(self, x_inc: torch.Tensor, x_dec: torch.Tensor,
                x_free: torch.Tensor, return_branches: bool = False):
        h_inc = self.branch_inc(x_inc)
        h_dec = self.branch_dec(x_dec)
        h_free = self.branch_free(x_free)
        raw = self.bias + h_inc + h_dec + h_free
        if self.output_activation == "exp":
            # Log-linear / multiplicative model: y = exp(raw).
            # Clip raw to [0, exp_output_clip] to prevent overflow.
            raw_clipped = torch.clamp(raw, min=0.0, max=self.exp_output_clip)
            y = torch.exp(raw_clipped) - 1.0  # equivalent to expm1, ensures y(0)=0
            y = torch.clamp(y, min=0.0)
        else:
            y = F.softplus(raw)  # default additive output, non-negative
        if return_branches:
            return y, {"h_inc": h_inc, "h_dec": h_dec, "h_free": h_free, "raw": raw}
        return y

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def split_inputs(x_full: torch.Tensor, n_inc: int, n_dec: int):
    """Split a (B, n_inc + n_dec + n_free) tensor into the three blocks.

    Block ordering: [inc | dec | free].
    """
    x_inc = x_full[:, :n_inc]
    x_dec = x_full[:, n_inc:n_inc + n_dec]
    x_free = x_full[:, n_inc + n_dec:]
    return x_inc, x_dec, x_free


if __name__ == "__main__":
    # Quick sanity check: compute parameter count and verify monotonicity
    model = PhysicsInformedDebrisNN(n_inc=11, n_dec=2, n_free=18)
    print(f"Parameters: {model.n_parameters()}")

    # Verify monotone-increasing branch
    x_inc = torch.randn(100, 11)
    x_dec = torch.randn(100, 2)
    x_free = torch.randn(100, 18)
    model.eval()
    with torch.no_grad():
        y0 = model(x_inc, x_dec, x_free)
        y1 = model(x_inc + 1.0, x_dec, x_free)  # increase all inc inputs
        print(f"Monotone-inc check: P(y_after_increase >= y_before) = "
              f"{(y1 >= y0 - 1e-6).float().mean():.3f} (should be ~1)")
        y2 = model(x_inc, x_dec + 1.0, x_free)  # increase dec inputs
        print(f"Monotone-dec check: P(y_after_increase <= y_before) = "
              f"{(y2 <= y0 + 1e-6).float().mean():.3f} (should be ~1)")
        # Non-negativity
        print(f"Non-neg output check: min(y) = {y0.min().item():.6f} "
              f"(should be >= 0)")
