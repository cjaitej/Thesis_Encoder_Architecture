"""
Oscillatory State-Space network for RoNIN seq2seq inertial odometry.

Motivation
----------
Bipedal walking is a stable, quasi-periodic limit cycle: the IMU stream is a
forced/damped oscillation riding on the gait cycle. This model makes the
temporal core an explicit bank of learnable *damped harmonic oscillators* -- an
oscillatory state-space model in the spirit of LinOSS (Rusch & Rus,
"Oscillatory State-Space Models", ICLR 2025, arXiv:2410.03943) -- rather than a
generic sequence mixer.

Diagonal (complex) parameterization
-----------------------------------
A harmonic oscillator's 2x2 real transition has a complex-conjugate eigenpair,
so each oscillator is equivalently a single *complex pole* lambda = r * e^{i*theta}
with r = |lambda| <= 1 (stable) and theta the per-step phase advance (frequency).
Working in this diagonal basis, the state update is a scalar complex recurrence

    s_t = lambda * s_{t-1} + f_t

i.e. an elementwise complex multiply-add, solved by a parallel associative
(prefix) scan. This is mathematically the same oscillatory model as the 2x2
real form but far more GPU-efficient: the real form spends the whole scan doing
millions of independent 2x2 matmuls (terrible arithmetic intensity, launch
bound -> GPU idles); the diagonal form is a few large elementwise complex ops.

Why this is distinct from the crowded neighbours in inertial odometry:
  * FTIN / FDIO / MambaIO model the *static spectrum* (FFT features / Laplacian
    frequency bands). This models the *dynamics / phase* of the oscillation via
    a stable discretised ODE with an explicit oscillatory pole per channel.
  * Mamba is a *selective* (input-gated) 1st-order state-space scan; this is an
    *oscillatory* one -- complex poles on a ring, 2nd-order dynamics.
  * The repo's own GraphLiquidNet uses CfC/liquid cells (continuous-time but
    non-oscillatory) with an unfused per-timestep Python loop. This is solved
    with a log-depth parallel scan -- no time loop.

Architecture:  ChannelGraphEncoder stem (reused from model_graphliquid, models
cross-axis IMU coupling) -> stack of bidirectional oscillatory layers ->
per-frame velocity head.

Input:  [B, T, input_channel]
Output: [B, T, output_channel]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_graphliquid import ChannelGraphEncoder


def diagonal_complex_scan(lam, f):
    """
    Inclusive parallel prefix scan of the diagonal complex recurrence

        s_t = lam * s_{t-1} + f_t ,   s_{-1} = 0

    with a time-invariant per-oscillator pole lam. Hillis-Steele inclusive scan:
    ceil(log2(T)) doubling steps, each an *elementwise* complex multiply-add
    (no matmul, no Python loop over time).

    Composition of affine maps g_k(x) = lam_k x + f_k (apply earlier `left`
    then later `right`):  (right o left).lam = right.lam * left.lam ;
    (right o left).f = right.lam * left.f + right.f.

    Args:
        lam: [P]        complex pole per oscillator (|lam| <= 1)
        f:   [B, T, P]  complex per-step forcing

    Returns:
        s:   [B, T, P]  complex oscillator state at every step
    """
    B, T, P = f.shape
    A = lam.unsqueeze(0).expand(T, P).contiguous()   # [T, P] accumulated pole
    b = f.clone()                                    # [B, T, P] accumulated forcing

    one = torch.ones(1, P, dtype=lam.dtype, device=lam.device)

    d = 1
    while d < T:
        A_left = torch.cat([one.expand(d, P), A[:T - d]], dim=0)          # [T, P]
        b_left = torch.cat([b.new_zeros(B, d, P), b[:, :T - d]], dim=1)   # [B, T, P]

        # update b with the *current* A (before A is overwritten)
        b = A * b_left + b
        A = A * A_left
        d *= 2

    return b


class OscillatoryLayer(nn.Module):
    """
    One bidirectional diagonal-oscillatory (complex-pole) state-space block.

    Per feature channel we run `state_dim` damped harmonic oscillators with
    learned poles lam = r * e^{i*theta}, r in (0,1) guaranteeing stability. The
    forcing is a learned real projection of the (pre-norm) input, injected as
    the real part; the readout uses both real and imaginary parts of the final
    oscillator state (real = displacement, imaginary ~ velocity/phase quadrature).
    An SSM sublayer (readout + input skip) is followed by a position-wise FFN
    sublayer, transformer-style.
    """

    def __init__(self, d_model, state_dim, d_ff, dropout=0.1, bidirectional=True,
                 theta_min=0.008, theta_max=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.bidirectional = bidirectional

        self.norm1 = nn.LayerNorm(d_model)
        self.B_mat = nn.Linear(d_model, state_dim, bias=False)     # input -> forcing per oscillator

        # Stable pole lam = r * exp(i*theta), r = exp(-exp(nu_log)) in (0,1).
        # r in [0.9, 0.999] -> memory horizon ~10..1000 steps (long memory, good
        # for the slow heading/velocity trends that drive ATE).
        r_low, r_high = 0.9, 0.999
        nu_low = math.log(-math.log(r_high))
        nu_high = math.log(-math.log(r_low))
        self.nu_log = nn.Parameter(torch.rand(state_dim) * (nu_high - nu_low) + nu_low)

        # theta = per-step phase advance. CRITICAL init detail: pedestrian gait is
        # LOW frequency -- a 2 Hz stride at 200 Hz is theta ~ 2*pi*2/200 ~ 0.063
        # rad/step. Initialising theta uniformly on (0, pi) (up to Nyquist) starts
        # nearly every oscillator ~50x too fast; they must then migrate a long way
        # and training stalls (this is what crippled the first osc run). Instead
        # init theta LOG-uniform over ~[0.25 Hz, 16 Hz] (theta in [0.008, 0.5]),
        # so the bank starts on the gait fundamental, its harmonics, and slow
        # velocity trends.
        log_theta = torch.rand(state_dim) * (math.log(theta_max) - math.log(theta_min)) + math.log(theta_min)
        self.theta = nn.Parameter(torch.exp(log_theta))

        readout_dim = state_dim * (4 if bidirectional else 2)      # [Re, Im] per direction
        self.C_mat = nn.Linear(readout_dim, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))                 # elementwise input skip
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def _pole(self):
        r = torch.exp(-torch.exp(self.nu_log))     # [P] in (0,1)
        lam = torch.polar(r, self.theta)           # [P] complex
        return lam, r

    def forward(self, u):
        """u: [B, T, d_model]"""
        B, T, _ = u.shape
        z = self.norm1(u)
        f_real = self.B_mat(z)                                     # [B, T, P]

        lam, r = self._pole()
        # LRU-style input normalization: keeps hidden-state variance ~constant
        # regardless of how close the pole sits to the unit circle.
        f_real = f_real * torch.sqrt(1.0 - r * r)
        f = torch.complex(f_real, torch.zeros_like(f_real))       # [B, T, P]

        if self.bidirectional:
            # run both directions in one scan by stacking on the batch axis
            f_all = torch.cat([f, f.flip(1)], dim=0)              # [2B, T, P]
            s_all = diagonal_complex_scan(lam, f_all)
            s_fwd = s_all[:B]
            s_bwd = s_all[B:].flip(1)
            feats = torch.cat([s_fwd.real, s_fwd.imag, s_bwd.real, s_bwd.imag], dim=-1)
        else:
            s = diagonal_complex_scan(lam, f)
            feats = torch.cat([s.real, s.imag], dim=-1)

        y = self.act(self.C_mat(feats) + self.D * z)             # SSM readout + input skip
        u = u + self.dropout(y)                                  # SSM residual
        u = u + self.dropout(self.ff(self.norm2(u)))             # FFN residual
        return u


class OscillatoryNet(nn.Module):
    """
    Channel-graph stem + bidirectional oscillatory (complex-pole SSM) temporal
    core for seq2seq inertial odometry.

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
        state_dim=128,
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
            OscillatoryLayer(d_model, state_dim, d_ff, dropout=dropout, bidirectional=bidirectional)
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


def _sequential_complex_scan(lam, f):
    """Reference O(T) sequential scan, used only to validate diagonal_complex_scan."""
    B, T, P = f.shape
    s = f.new_zeros(B, P)
    out = []
    for t in range(T):
        s = lam * s + f[:, t]
        out.append(s)
    return torch.stack(out, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) parallel scan must match the sequential reference on a *stable* pole
    #    (|lam| < 1). A pole outside the unit circle would blow up like |lam|^T
    #    and let float rounding, not logic, dominate.
    B, T, P = 3, 200, 32
    r = torch.rand(P) * 0.099 + 0.9                 # |lam| in [0.9, 0.999]
    theta = torch.rand(P) * math.pi
    lam = torch.polar(r, theta)
    f = torch.complex(torch.randn(B, T, P), torch.randn(B, T, P))
    par = diagonal_complex_scan(lam, f)
    seq = _sequential_complex_scan(lam, f)
    max_err = (par - seq).abs().max().item()
    print(f"[scan check] max|parallel - sequential| = {max_err:.3e}")
    assert max_err < 1e-4, "parallel scan disagrees with sequential reference!"

    # 2) model forward + param count
    model = OscillatoryNet()
    n = model.get_num_params()

    # pole frequencies should start LOW (gait band), not spread to Nyquist
    with torch.no_grad():
        th = model.layers[0].theta
        hz = th * 200.0 / (2 * math.pi)   # per-step phase -> Hz at 200 Hz sampling
        print(f"[init] layer0 theta: {th.min():.3f}..{th.max():.3f} rad "
              f"(~{hz.min():.2f}..{hz.max():.2f} Hz), median ~{hz.median():.2f} Hz")
    x = torch.randn(2, 200, 6)
    y = model(x)
    print(f"[model] params = {n:,}  under 1M = {n < 1_000_000}")
    print(f"[model] output shape = {tuple(y.shape)} (expected (2, 200, 2))")
    assert y.shape == (2, 200, 2)

    # 3) gradient flows to every parameter
    y.sum().backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"[model] all params received grad = {grads_ok}")

    # 4) quick CPU timing at a training-like batch
    import time
    xb = torch.randn(128, 200, 6)
    model(xb).sum().backward()      # warmup
    t0 = time.time()
    for _ in range(3):
        model.zero_grad(); model(xb).sum().backward()
    print(f"[timing] B=128 T=200 fwd+bwd: {(time.time() - t0) / 3:.3f}s/iter (CPU)")
