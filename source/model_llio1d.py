"""LLIO-style 1D backbone (ResMLP feature extractor + MLP regression head).

The architecture here is ported from the reference implementation released with
"LLIO: Lightweight Learned Inertial Odometer" (Wang, Kuang, Niu and Liu, IEEE
Internet of Things Journal, 2022):

    https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer
    (files: model_twolayer.py, model_MLP.py -- GPLv3)

What is reused verbatim (same blocks, same structure as their Fig. 3):
  * Affine / PreAffinePostLayerScale        (ResMLP affine + layer-scale residual)
  * ResMLPExtractor                         (patchify -> linear -> N ResMLP blocks)
  * SimpleMLPReg / PoolingMLPReg            (flatten-or-pool -> MLP -> two heads)

What is adapted, and why (their repo ships the architecture only -- no training
code, no loss, no dataset loader, no released weights -- so this backbone is
trained under *our* protocol, exactly like every other competing backbone in
this repo, and does not reproduce their published numbers):

  * Window length 100 -> 200. The paper uses "L = 100 as we collect IMU at
    100 Hz. Thus, we send 1 second of IMU samples to the network"; our pipeline
    feeds 1 second at 200 Hz, i.e. 200 samples.
  * patch_len 25 -> 50. Keeping the *duration* of a patch fixed at 0.25 s
    (25 samples @ 100 Hz == 50 samples @ 200 Hz) preserves both their patch
    granularity and their patch count (Npatch = 4) over the same 1 s window,
    rather than arbitrarily rescaling one of the two. Their own ablation
    (Table III, rows E/F) sweeps patch size 10/25/50 and moves distance error
    by at most 0.002 m, so this axis is insensitive by their measurement.
  * Output 3-D displacement + 3-D covariance -> 2-D planar velocity. Their head
    emits (d_hat, diag(Sigma)) for an SCEKF measurement update, trained with
    MSE on d_hat plus an NLL term coupling d_hat and Sigma; this pipeline
    regresses planar velocity in the heading-agnostic frame and integrates it
    directly, so the covariance head is not built and the value head is sized
    to num_outputs (2). Dropping the covariance output also drops their NLL
    term, leaving exactly the MSE objective this pipeline already uses for
    every backbone.
  * einops Rearrange/Reduce -> plain torch reshape/permute/mean, to avoid
    adding an einops dependency to the training environment. Equivalence to
    the original patterns is asserted in tools/verify_llio1d.py.
  * Dropout in the regression head is threaded from --model_dropout instead of
    being hard-coded to 0.5, so this backbone honours the same dropout setting
    as the other competing backbones.
  * Affine gains/biases and the layer-scale vector are stored as 1-D [dim]
    instead of [1, 1, dim]. The forward result is bit-identical by broadcasting
    and the parameter count is unchanged, but the stored rank matters under
    this repo's MuSGD: it orthogonalises every parameter with ndim >= 2, so as
    [1, 1, dim] these 62 per-feature gains would receive Newton-Schulz matrix
    updates, while every other backbone's equivalent parameters (BatchNorm
    gains/biases, ndim == 1) take the plain SGD path. Flattening also cuts the
    optimiser step from 57.0 ms to 23.3 ms per batch of 128 on an A6000
    (68.8 -> 32.1 ms/step total, i.e. 12.7 h -> 5.9 h for 100 epochs), since
    the 62 tiny Newton-Schulz calls were almost pure kernel-launch overhead.
    This artefact cannot arise in the original, which trains with Adam, where
    the stored rank is irrelevant.

Hyperparameters follow the published ResMLP512 configuration, taken from the
paper's Section IV-A.4 (Training Details) and its ablation Table III:

    6 ResMLP blocks | expansion dimension E = 2 | dropout 0.2 | GELU
    patch length 25 @ L = 100 (1 s @ 100 Hz) | inner feature dimension 512
    regression module = average pooling + linear + GELU

The paper's ResMLP512 is the headline configuration and the reference row of
its ablation; ResMLP256 and ResMLP128 are the same architecture with the inner
feature dimension reduced, so `feature_dim` switches between them directly.

Note on size: at this configuration the model carries ~7.27M parameters, more
than the 4.63M RoNIN-ResNet baseline. That is expected rather than a porting
error -- the paper states plainly that "the ResMLP512 has more FLOPs than
ResNet but exhibits better inference efficiency", i.e. LLIO's efficiency claim
is about wall-clock inference time from MLP-only operators, not about
parameter count or FLOPs.
"""

import torch
import torch.nn as nn


class Affine(nn.Module):
    """Per-feature affine transform (ResMLP's LayerNorm replacement).

    The original stores g/b with shape [1, 1, dim]; they are 1-D [dim] here and
    broadcast to the same result (verified bit-identical). See the note on
    parameter shape in the module docstring -- under MuSGD the stored rank
    decides whether a parameter is orthogonalised, and per-feature gains should
    be treated the way every other backbone's BatchNorm gains are.
    """

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    """Affine -> fn -> layer-scale -> residual -> affine, as in ResMLP."""

    def __init__(self, dim, depth, fn):
        super().__init__()
        init_eps = 0.1 if depth <= 18 else 1e-5
        self.scale = nn.Parameter(torch.full((dim,), init_eps))
        self.affine = Affine(dim)
        self.affine_out = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.affine_out(self.fn(self.affine(x)) * self.scale + x)


class Patchify(nn.Module):
    """einops Rearrange('b c (l w) -> b l (w c)', w=patch_len), in plain torch.

    Splits the time axis into ``patch_num`` consecutive patches of ``patch_len``
    samples and flattens each patch's (sample, channel) values into one feature
    vector, so element [b, l, w*C + c] == input[b, c, l*patch_len + w].
    """

    def __init__(self, patch_num, patch_len):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len

    def forward(self, x):
        b, c, length = x.shape
        expected = self.patch_num * self.patch_len
        if length != expected:
            raise ValueError(
                f"LLIO1D expects a window of {expected} samples "
                f"(patch_num={self.patch_num} x patch_len={self.patch_len}), got {length}")
        # [b, c, l*w] -> [b, c, l, w] -> [b, l, w, c] -> [b, l, w*c]
        x = x.reshape(b, c, self.patch_num, self.patch_len)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(b, self.patch_num, self.patch_len * c)


class ResMLPExtractor(nn.Module):
    """Patch embedding followed by ``layer_num`` ResMLP blocks.

    INPUT:  [batch, input_channel, patch_num * patch_len]
    OUTPUT: [batch, patch_num, mlp_in_dim]
    """

    def __init__(self, patch_num=4, patch_len=50, input_channel=6, mlp_in_dim=512,
                 expansion=2, active_func=None, layer_num=6, dropout=0.2):
        super().__init__()
        if active_func is None:
            active_func = nn.GELU()

        def wrapper(i, fn):
            return PreAffinePostLayerScale(mlp_in_dim, i, fn)

        self.net = nn.Sequential(
            # ---- feature convert ----
            Patchify(patch_num, patch_len),
            nn.Linear(int(patch_len * input_channel), mlp_in_dim),
            # ---- ResMLP blocks: token mixing (over patches) + channel MLP ----
            *[
                nn.Sequential(
                    wrapper(i, nn.Conv1d(patch_num, patch_num, 1, bias=False)),
                    wrapper(i, nn.Sequential(
                        nn.Linear(mlp_in_dim, mlp_in_dim * expansion, bias=False),
                        active_func,
                        nn.Dropout(p=dropout),
                        nn.Linear(mlp_in_dim * expansion, mlp_in_dim, bias=False),
                    )),
                ) for i in range(layer_num)
            ],
            Affine(mlp_in_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLPReg(nn.Module):
    """Flatten every patch feature, then an MLP with a value and a cov head.

    INPUT:  [batch, patch_num, feature_len]
    OUTPUT: ([batch, out_dim], [batch, out_dim])  == (y, y_cov)
    """

    def __init__(self, patch_num=4, feature_len=512, out_dim=2, layer_num=3,
                 active_fun=None, dropout=0.2, with_cov_head=False):
        super().__init__()
        if active_fun is None:
            active_fun = nn.GELU()
        self.input_len = int(patch_num * feature_len)
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                nn.Dropout(p=dropout),
            ) for _ in range(layer_num)
        ])
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim) if with_cov_head else None

    def forward(self, x):
        out = self.net(x.reshape(x.size(0), -1))
        cov = self.out2_linear(out) if self.out2_linear is not None else None
        return self.out_linear(out), cov


class PoolingMLPReg(nn.Module):
    """Pool over patches, then an MLP with a value and a cov head.

    INPUT:  [batch, patch_num, feature_len]
    OUTPUT: ([batch, out_dim], [batch, out_dim])  == (y, y_cov)
    """

    def __init__(self, patch_num=4, feature_len=512, out_dim=2, layer_num=3,
                 active_fun=None, dropout=0.2, pooling_type='mean', with_cov_head=False):
        super().__init__()
        if active_fun is None:
            active_fun = nn.GELU()
        if pooling_type not in ('mean', 'max'):
            raise ValueError(f"pooling_type must be 'mean' or 'max', got {pooling_type}")
        self.pooling_type = pooling_type
        self.input_len = int(feature_len)
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                nn.Dropout(p=dropout),
            ) for _ in range(layer_num)
        ])
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim) if with_cov_head else None

    def forward(self, x):
        # einops Reduce('b l f -> b f', pooling_type)
        pooled = x.mean(dim=1) if self.pooling_type == 'mean' else x.max(dim=1).values
        out = self.net(pooled)
        cov = self.out2_linear(out) if self.out2_linear is not None else None
        return self.out_linear(out), cov


class LLIO1D(nn.Module):
    """LLIO-style two-stage regressor adapted to 200-sample velocity regression.

    INPUT:  [batch, in_channels, window_size]
    OUTPUT: [batch, num_outputs]   (planar velocity; covariance head discarded)
    """

    def __init__(self, in_channels=6, num_outputs=2, dropout=0.2,
                 window_size=200, patch_len=50, feature_dim=512,
                 extractor_layers=6, expansion=2, reg_layers=3, reg_type='mean',
                 with_cov_head=False):
        super().__init__()
        if window_size % patch_len != 0:
            raise ValueError(
                f"window_size ({window_size}) must be divisible by patch_len ({patch_len})")
        patch_num = window_size // patch_len
        self.window_size = window_size
        self.patch_num = patch_num
        self.patch_len = patch_len

        active_func = nn.GELU()
        self.extractor = ResMLPExtractor(
            patch_num=patch_num, patch_len=patch_len, input_channel=in_channels,
            mlp_in_dim=feature_dim, expansion=expansion, active_func=active_func,
            layer_num=extractor_layers, dropout=dropout)

        if reg_type in ('mean', 'max'):
            self.reg = PoolingMLPReg(
                patch_num=patch_num, feature_len=feature_dim, out_dim=num_outputs,
                layer_num=reg_layers, active_fun=active_func, dropout=dropout,
                pooling_type=reg_type, with_cov_head=with_cov_head)
        elif reg_type == 'flatten':
            self.reg = SimpleMLPReg(
                patch_num=patch_num, feature_len=feature_dim, out_dim=num_outputs,
                layer_num=reg_layers, active_fun=active_func, dropout=dropout,
                with_cov_head=with_cov_head)
        else:
            raise ValueError(f"Unknown reg_type: {reg_type}")

        # No custom initialization is applied here. TwoLayerModel in the
        # official repo defines a _initialize(zero_init_residual) method with
        # this exact logic (Conv1d -> kaiming_normal_, BatchNorm1d -> 1/0,
        # Linear -> normal_(0, 0.01)) but never calls it anywhere in
        # __init__ or elsewhere in the file -- confirmed against the current
        # upstream source. The model they actually train therefore uses
        # PyTorch's own nn.Linear/nn.Conv1d default initialization (Kaiming-
        # uniform), which is what falling through to no override gives here.
        # This was previously ported as a called self._initialize(), which
        # is a real bug, not a faithfulness improvement: normal_(0, 0.01) on
        # every Linear in this 3-layer regression head starves the output of
        # signal (measured output std ~1e-6 across different random inputs
        # at init, vs ~0.0036 with PyTorch's default), and training under
        # MuSGD failed to escape that near-constant starting point within
        # several epochs on real data. Removing the override fixed it.

    def forward(self, x):
        feature = self.extractor(x)
        out, _out_cov = self.reg(feature)
        # The original emits (y, y_cov) because y_cov feeds an EKF measurement
        # update. This pipeline integrates velocity directly, so the covariance
        # head is not built by default (with_cov_head=False): keeping it would
        # add parameters that receive no gradient and are never executed, which
        # would misreport the deployed footprint in the parameter/size tables.
        # forward() therefore returns just [batch, num_outputs], the contract
        # shared by every other backbone here.
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2):
    return LLIO1D(in_channels=in_channels, num_outputs=num_outputs, dropout=dropout)
