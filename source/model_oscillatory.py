"""
Oscillatory State-Space network for RoNIN seq2seq inertial odometry.

Motivation
----------
Bipedal walking is a stable, quasi-periodic limit cycle: the IMU stream is a
forced/damped oscillation riding on the gait cycle. This model makes the
temporal core an explicit bank of learnable *forced harmonic oscillators* --
a Linear Oscillatory State-Space model (LinOSS; Rusch & Rus, "Oscillatory
State-Space Models", ICLR 2025, arXiv:2410.03943) -- rather than a generic
sequence mixer.

Why this is distinct from the crowded neighbours in inertial odometry:
  * FTIN / FDIO / MambaIO model the *static spectrum* (FFT features / Laplacian
    frequency bands). LinOSS models the *dynamics / phase* of the oscillation in
    the time domain via a stable discretised ODE.
  * Mamba is a *selective* state-space scan; LinOSS is an *oscillatory* one
    (2nd-order harmonic-oscillator dynamics, not 1st-order decay).
  * The repo's own GraphLiquidNet uses CfC/liquid cells (continuous-time but
    non-oscillatory) and, critically, an unfused per-timestep Python loop.
    LinOSS is solved with a parallel associative (prefix) scan -- fully
    parallel over time, so it does not have GraphLiquidNet's throughput problem.

Architecture:  ChannelGraphEncoder stem (reused from model_graphliquid, models
cross-axis IMU coupling) -> stack of bidirectional LinOSS layers -> per-frame
velocity head.

Input:  [B, T, input_channel]
Output: [B, T, output_channel]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_graphliquid import ChannelGraphEncoder


def parallel_affine_scan(M, F_in):
    """
    Inclusive parallel prefix scan of the linear recurrence

        s_t = M @ s_{t-1} + F_t ,   s_{-1} = 0

    where the transition M is time-invariant (one 2x2 block per oscillator).
    Uses a Hillis-Steele inclusive scan: ceil(log2(T)) doubling steps, each a
    batched matmul -- no Python loop over time.

    Composition rule for affine maps f_k(x) = M x + F_k (apply earlier `left`
    then later `right`):
        (right o left).A = right.A @ left.A
        (right o left).b = right.A @ left.b + right.b

    Args:
        M:    [P, 2, 2]        per-oscillator transition (shared over batch/time)
        F_in: [B, T, P, 2]     per-step forcing

    Returns:
        s:    [B, T, P, 2]     the running oscillator state at every step
    """
    B, T, P, _ = F_in.shape
    device, dtype = F_in.device, F_in.dtype

    # A[t] carries the accumulated transition for the segment ending at t
    # (no batch dependence); b[t] carries the accumulated forcing (batched).
    A = M.unsqueeze(0).expand(T, P, 2, 2).contiguous()   # [T, P, 2, 2]
    b = F_in.clone()                                     # [B, T, P, 2]

    eye = torch.eye(2, device=device, dtype=dtype).view(1, 1, 2, 2).expand(-1, P, -1, -1)

    d = 1
    while d < T:
        # left neighbour = element at t-d (identity / zero for the first d)
        A_left = torch.cat([eye.expand(d, P, 2, 2), A[:T - d]], dim=0)          # [T, P, 2, 2]
        b_left = torch.cat([b.new_zeros(B, d, P, 2), b[:, :T - d]], dim=1)      # [B, T, P, 2]

        # b must be updated with the *current* A (before A is overwritten)
        b = torch.einsum('tpij,btpj->btpi', A, b_left) + b
        A = torch.einsum('tpij,tpjk->tpik', A, A_left)
        d *= 2

    return b


class LinOSSLayer(nn.Module):
    """
    One bidirectional Linear Oscillatory State-Space block.

    Per feature channel we run a bank of `state_dim` forced harmonic oscillators

        x'' = -A x + (B u),    A >= 0  (diagonal, learned squared-frequencies)

    discretised with the LinOSS implicit (IM) scheme, which is provably stable
    for any A >= 0 and any timestep dt > 0 (eigenvalues of the update have
    magnitude <= 1). Writing the state s = [z, x] (z = x'), one IM step is a
    fixed linear recurrence s_t = M s_{t-1} + F_t, so the whole sequence is
    produced by `parallel_affine_scan` in log-depth instead of a time loop.

    The layer is a pre-norm SSM sublayer (readout of the oscillator positions x
    + input skip) followed by a position-wise FFN sublayer, transformer-style.
    """

    def __init__(self, d_model, state_dim, d_ff, dropout=0.1,
                 bidirectional=True, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.state_dim = state_dim
        self.bidirectional = bidirectional

        self.norm1 = nn.LayerNorm(d_model)
        self.B_mat = nn.Linear(d_model, state_dim, bias=False)      # input -> forcing per oscillator

        # A >= 0 via softplus(A_raw); init so frequencies span a modest range.
        self.A_raw = nn.Parameter(torch.rand(state_dim) * 2.0 - 1.0)
        # per-oscillator timestep dt = exp(log_dt), log-uniform in [dt_min, dt_max]
        log_dt = torch.rand(state_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        c_in = state_dim * 2 if bidirectional else state_dim
        self.C_mat = nn.Linear(c_in, d_model, bias=False)           # oscillator positions -> output
        self.D = nn.Parameter(torch.ones(d_model))                  # elementwise input skip
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def _transition(self):
        """Build the per-oscillator IM transition M [P,2,2] and forcing
        coefficients coef_z, coef_x [P] from A and dt."""
        A = F.softplus(self.A_raw)                 # [P], >= 0
        dt = torch.exp(self.log_dt)                # [P], > 0
        dt2 = dt * dt
        S = 1.0 / (1.0 + dt2 * A)                  # in (0, 1]

        M00 = 1.0 - dt2 * A * S
        M01 = -dt * A * S
        M10 = S * dt
        M11 = S
        M = torch.stack([
            torch.stack([M00, M01], dim=-1),
            torch.stack([M10, M11], dim=-1),
        ], dim=-2)                                 # [P, 2, 2]

        coef_z = dt - dt * dt2 * A * S             # forcing into z (velocity) component
        coef_x = S * dt2                           # forcing into x (position) component
        return M, coef_z, coef_x

    def _scan_positions(self, f, M, coef_z, coef_x, reverse):
        """Run the oscillator scan on forcing f [B,T,P], return positions [B,T,P]."""
        if reverse:
            f = f.flip(1)
        F_in = torch.stack([coef_z * f, coef_x * f], dim=-1)   # [B, T, P, 2]
        s = parallel_affine_scan(M, F_in)                      # [B, T, P, 2]
        x = s[..., 1]                                          # oscillator position
        if reverse:
            x = x.flip(1)
        return x

    def forward(self, u):
        """u: [B, T, d_model]"""
        z = self.norm1(u)
        f = self.B_mat(z)                                      # [B, T, P]
        M, coef_z, coef_x = self._transition()

        x = self._scan_positions(f, M, coef_z, coef_x, reverse=False)
        if self.bidirectional:
            x_bwd = self._scan_positions(f, M, coef_z, coef_x, reverse=True)
            x = torch.cat([x, x_bwd], dim=-1)                 # [B, T, 2P]

        y = self.act(self.C_mat(x) + self.D * z)              # SSM readout + input skip
        u = u + self.dropout(y)                               # SSM residual
        u = u + self.dropout(self.ff(self.norm2(u)))          # FFN residual
        return u


class OscillatoryNet(nn.Module):
    """
    Channel-graph stem + bidirectional oscillatory (LinOSS) temporal core for
    seq2seq inertial odometry.

    Input:  [B, T, input_channel]
    Output: [B, T, output_channel]
    """

    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        node_dim=48,
        num_graph_layers=2,
        graph_nhead=4,
        d_model=160,
        state_dim=160,
        num_osc_layers=4,
        d_ff=320,
        kernel_size=5,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()
        self.input_channel = input_channel

        self.channel_graph = ChannelGraphEncoder(
            input_channel, node_dim, num_graph_layers, kernel_size,
            nhead=graph_nhead, dropout=dropout,
        )
        self.graph_proj = nn.Linear(input_channel * node_dim, d_model)

        self.layers = nn.ModuleList([
            LinOSSLayer(d_model, state_dim, d_ff, dropout=dropout, bidirectional=bidirectional)
            for _ in range(num_osc_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, output_channel),
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """x: [B, T, input_channel]"""
        assert x.shape[-1] == self.input_channel, \
            f"expected last dim {self.input_channel}, got {x.shape}"
        h = self.graph_proj(self.channel_graph(x))            # [B, T, d_model]
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def _sequential_affine_scan(M, F_in):
    """Reference O(T) sequential scan, used only to validate parallel_affine_scan."""
    B, T, P, _ = F_in.shape
    s = F_in.new_zeros(B, P, 2)
    out = []
    for t in range(T):
        s = torch.einsum('pij,bpj->bpi', M, s) + F_in[:, t]
        out.append(s)
    return torch.stack(out, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) parallel scan must match the sequential reference.
    #    Use the *actual* LinOSS transition (provably contractive, |eig| <= 1) so
    #    the recurrence is well-conditioned -- a random non-contractive M would
    #    blow up like ||M||^T and make float rounding, not logic, dominate.
    B, T, P = 3, 200, 16
    ref_layer = LinOSSLayer(d_model=8, state_dim=P, d_ff=16)
    M, coef_z, coef_x = ref_layer._transition()
    M = M.detach()
    F_in = torch.randn(B, T, P, 2)
    par = parallel_affine_scan(M, F_in)
    seq = _sequential_affine_scan(M, F_in)
    max_err = (par - seq).abs().max().item()
    print(f"[scan check] max|parallel - sequential| = {max_err:.3e}")
    assert max_err < 1e-4, "parallel scan disagrees with sequential reference!"

    # 2) model forward + param count
    model = OscillatoryNet()
    n = model.get_num_params()
    x = torch.randn(2, 200, 6)
    y = model(x)
    print(f"[model] params = {n:,}  under 1M = {n < 1_000_000}")
    print(f"[model] output shape = {tuple(y.shape)} (expected (2, 200, 2))")
    assert y.shape == (2, 200, 2)

    # 3) gradient flows
    y.sum().backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"[model] all params received grad = {grads_ok}")
