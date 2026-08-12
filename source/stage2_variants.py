"""
source/stage2_variants.py

Stage-2 residual-correction variants for the YOLOv26-1D inertial odometry pipeline.

This file is additive: it does not modify stage 1 (the neural backbone), the
training scripts, or the existing RF pipeline. It consumes the same datasets
already produced by prepare_rf_dataset_yolo26.py (rf_train.npz / rf_val.npz) and
exposes every variant behind one interface so they are directly comparable.

Variants
--------
  A  RandomForestCorrector    incumbent (matches train_rf_postprocess_yolo26.py)
  B  RidgeCorrector           linear control - is nonlinearity needed at all?
  C  CausalFilterCorrector    1-parameter causal exponential filter
  D  MLPCorrector             small feed-forward net on the same features
  E  TemporalCorrector        dilated causal 1D conv net (TCN)

Every corrector implements:
    fit(train, val=None)                -> self
    predict_residual(data)              -> (N, 2) predicted residual
    size_bytes()                        -> serialized footprint

The predicted residual is applied with the existing correction rule from
rf_utils.apply_rf_correction, so all variants plug into the same
v~ = v^ + clip(alpha * r^) arithmetic the RF already uses.

All variants are strictly causal: no feature at timestep t depends on any
sample after t. Sequence boundaries are respected (no bleeding across
trajectories).

Usage
-----
  python source/stage2_variants.py \
      --train_npz output/rf_yolo26/rf_train.npz \
      --val_npz   output/rf_yolo26/rf_val.npz \
      --variants all --out_dir output/stage2_models
"""

import argparse
import io
import json
import os
import time
from os import path as osp

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from rf_utils import apply_rf_correction

# Feature that cannot be computed in a streaming deployment: rf_utils builds
# time_progress from ts[-1], the last timestamp of the whole sequence.
NON_CAUSAL_FEATURES = ('time_progress',)


# --------------------------------------------------------------------------- #
# data handling
# --------------------------------------------------------------------------- #

def segment_bounds(seq_name):
    """Contiguous [start, end) spans, one per trajectory.

    The npz rows are written sequence-by-sequence in temporal order, so
    contiguous runs of seq_name delimit trajectories. Using runs rather than
    np.unique preserves that order.
    """
    if len(seq_name) == 0:
        return []
    breaks = np.nonzero(seq_name[1:] != seq_name[:-1])[0] + 1
    edges = np.concatenate([[0], breaks, [len(seq_name)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def load_stage2_npz(path, drop_features=()):
    """Load an rf_*.npz export into the dict every corrector consumes."""
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d['feature_names'].tolist()]
    X = d['X'].astype(np.float32)

    keep = [i for i, n in enumerate(names) if n not in set(drop_features)]
    if len(keep) != len(names):
        X = X[:, keep]
        names = [names[i] for i in keep]

    data = {
        'X': X,
        'y': d['y'].astype(np.float32),
        'pred': d['pred'].astype(np.float32),
        'gt': d['gt'].astype(np.float32),
        'seq_name': d['seq_name'],
        'ts': d['ts'],
        'feature_names': names,
    }
    data['segments'] = segment_bounds(data['seq_name'])
    return data


class Standardizer:
    """Zero-mean unit-variance feature scaling, fitted on the train split."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = X.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #

class Stage2Corrector:
    """Common interface: features (and/or the prediction stream) -> residual."""

    name = 'base'
    kind = 'base'
    feature_names = None
    best_alpha = 1.0   # correction gain chosen on the validation split
    best_clip = None   # residual clip bound chosen on the validation split

    def fit(self, train, val=None):
        raise NotImplementedError

    def predict_residual(self, data):
        raise NotImplementedError

    def size_bytes(self):
        raise NotImplementedError

    def describe(self):
        return self.name

    def params(self):
        """Hyperparameters, written to <name>_corrector_summary.json as
        'best_params' to mirror rf_corrector_summary.json."""
        return {}

    def _remember_features(self, train):
        names = train.get('feature_names')
        self.feature_names = list(names) if names is not None else None

    def _X(self, data):
        """Feature matrix reordered/subset to match what this model was fitted on.

        Lets a corrector trained with --drop_non_causal consume the full 18-column
        matrix that rf_utils.build_sequence_features produces at evaluation time.
        """
        X = data['X']
        names = data.get('feature_names')
        if not self.feature_names or names is None:
            return X
        names = list(names)
        if names == self.feature_names:
            return X
        missing = [n for n in self.feature_names if n not in names]
        if missing:
            raise ValueError('%s needs features not present at inference: %s'
                             % (self.name, missing))
        return X[:, [names.index(n) for n in self.feature_names]]


def save_corrector(model, path):
    """Persist a fitted corrector. Torch weights are moved to CPU first so the
    artifact loads on a machine without a GPU (e.g. the Raspberry Pi)."""
    inner = getattr(model, 'model', None)
    restore = None
    if isinstance(inner, nn.Module):
        restore = model.device
        inner.to('cpu')
        model.device = torch.device('cpu')
    try:
        os.makedirs(osp.dirname(path) or '.', exist_ok=True)
        joblib.dump(model, path)
    finally:
        if restore is not None:
            model.device = restore
            inner.to(restore)
    return path


def load_corrector(path, device='auto'):
    """Load a corrector saved by save_corrector and place it on `device`."""
    model = joblib.load(path)
    inner = getattr(model, 'model', None)
    if isinstance(inner, nn.Module):
        model.device = _resolve_device(device)
        inner.to(model.device)
        inner.eval()
    return model


# --------------------------------------------------------------------------- #
# A. Random Forest (incumbent)
# --------------------------------------------------------------------------- #

class RandomForestCorrector(Stage2Corrector):
    """Matches the grid-search winner recorded in rf_corrector_summary.json."""

    kind = 'tree'

    def __init__(self, n_estimators=400, max_depth=12, min_samples_leaf=1,
                 max_features='sqrt', random_state=42, n_jobs=-1):
        self.name = 'A_rf'
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, train, val=None):
        self._remember_features(train)
        self.model.fit(train['X'], train['y'])
        return self

    def predict_residual(self, data):
        return self.model.predict(self._X(data)).astype(np.float32)

    def size_bytes(self):
        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        return buf.getbuffer().nbytes

    def params(self):
        p = self.model.get_params()
        return {k: p[k] for k in
                ('n_estimators', 'max_depth', 'min_samples_leaf', 'max_features')}

    def describe(self):
        p = self.model.get_params()
        return 'RandomForest(n_estimators=%d, max_depth=%s, min_samples_leaf=%d, max_features=%s)' % (
            p['n_estimators'], p['max_depth'], p['min_samples_leaf'], p['max_features'])


# --------------------------------------------------------------------------- #
# B. Ridge (linear control)
# --------------------------------------------------------------------------- #

class RidgeCorrector(Stage2Corrector):
    """Multi-output linear map from features to residual."""

    kind = 'linear'

    def __init__(self, alpha=1.0):
        self.name = 'B_ridge'
        self.reg_alpha = alpha
        self.scaler = Standardizer()
        self.model = Ridge(alpha=alpha)

    def fit(self, train, val=None):
        self._remember_features(train)
        self.scaler.fit(train['X'])
        self.model.fit(self.scaler.transform(train['X']), train['y'])
        return self

    def predict_residual(self, data):
        return self.model.predict(self.scaler.transform(self._X(data))).astype(np.float32)

    def size_bytes(self):
        n = self.model.coef_.size + np.size(self.model.intercept_)
        n += self.scaler.mean.size + self.scaler.std.size
        return int(n) * 4

    def params(self):
        return {'ridge_alpha': self.reg_alpha, 'n_features': int(self.model.coef_.shape[1])}

    def describe(self):
        return 'Ridge(alpha=%g, n_features=%d)' % (self.reg_alpha, self.model.coef_.shape[1])


# --------------------------------------------------------------------------- #
# C. Causal stream filter
# --------------------------------------------------------------------------- #

def _ema_stream(pred, segments, alpha):
    """Causal exponential moving average, restarted at every segment."""
    out = np.empty_like(pred)
    for a, b in segments:
        p = pred[a:b]
        acc = p[0].copy()
        for i in range(p.shape[0]):
            acc = alpha * p[i] + (1.0 - alpha) * acc
            out[a + i] = acc
    return out


class CausalFilterCorrector(Stage2Corrector):
    """Filters the predicted-velocity stream instead of regressing on features.

    One parameter (the smoothing factor), grid-searched on the validation split.
    The returned residual is filtered(pred) - pred, so the standard
    v~ = v^ + clip(alpha * r^) rule reproduces the filter exactly at alpha=1.
    """

    kind = 'filter'

    def __init__(self, ema_grid=(0.40, 0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08)):
        self.name = 'C_ema'
        self.ema_grid = tuple(ema_grid)
        self.alpha = None

    def fit(self, train, val=None):
        self._remember_features(train)
        tune = val if val is not None else train
        best = (None, np.inf)
        for a in self.ema_grid:
            sm = _ema_stream(tune['pred'], tune['segments'], a)
            mse = float(np.mean((tune['gt'] - sm) ** 2))
            if mse < best[1]:
                best = (a, mse)
        self.alpha = best[0]
        return self

    def predict_residual(self, data):
        smoothed = _ema_stream(data['pred'], data['segments'], self.alpha)
        return (smoothed - data['pred']).astype(np.float32)

    def size_bytes(self):
        return 4

    def params(self):
        return {'ema_alpha': float(self.alpha), 'search_grid': list(self.ema_grid)}

    def describe(self):
        return 'CausalEMA(alpha=%.3f)' % self.alpha


# --------------------------------------------------------------------------- #
# torch helpers
# --------------------------------------------------------------------------- #

def _resolve_device(device):
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# D. Small MLP
# --------------------------------------------------------------------------- #

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 64), out_dim=2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPCorrector(Stage2Corrector):
    """Per-timestep feed-forward net on the same features the RF sees."""

    kind = 'neural'

    def __init__(self, hidden=(64, 64), epochs=15, lr=1e-3, batch_size=4096,
                 weight_decay=0.0, device='auto', seed=42, verbose=True):
        self.name = 'D_mlp'
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = _resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.scaler = Standardizer()
        self.model = None

    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        self._remember_features(train)
        self.scaler.fit(train['X'])
        Xt = torch.from_numpy(self.scaler.transform(train['X']))
        yt = torch.from_numpy(train['y'])

        self.model = _MLP(Xt.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        n = Xt.shape[0]

        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xt[idx].to(self.device, non_blocking=True)
                yb = yt[idx].to(self.device, non_blocking=True)
                loss = F.mse_loss(self.model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * xb.shape[0]
            if self.verbose:
                msg = '    [%s] epoch %2d/%d  train_mse=%.6f' % (
                    self.name, ep + 1, self.epochs, total / n)
                if val is not None:
                    msg += '  val_mse=%.6f' % residual_mse(val, self.predict_residual(val))[2]
                print(msg)
        return self

    @torch.no_grad()
    def predict_residual(self, data):
        self.model.eval()
        X = torch.from_numpy(self.scaler.transform(self._X(data)))
        out = []
        for i in range(0, X.shape[0], 65536):
            out.append(self.model(X[i:i + 65536].to(self.device)).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def size_bytes(self):
        n = _count_params(self.model)
        n += self.scaler.mean.size + self.scaler.std.size
        return int(n) * 4

    def params(self):
        return {'hidden': list(self.hidden), 'epochs': self.epochs, 'lr': self.lr,
                'batch_size': self.batch_size, 'weight_decay': self.weight_decay,
                'n_parameters': _count_params(self.model)}

    def describe(self):
        return 'MLP(hidden=%s, params=%d)' % (list(self.hidden), _count_params(self.model))


# --------------------------------------------------------------------------- #
# E. Causal temporal model (TCN)
# --------------------------------------------------------------------------- #

class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each timestep independently.

    Deliberately not GroupNorm/BatchNorm: those pool statistics over the time
    axis, which would make every output depend on future samples and silently
    break causality.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):  # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _CausalTCN(nn.Module):
    """Dilated causal 1D conv stack. Input/output: (B, C, T)."""

    def __init__(self, in_dim, channels=32, kernel=3, dilations=(1, 2, 4, 8), out_dim=2):
        super().__init__()
        self.kernel = kernel
        self.dilations = tuple(dilations)
        blocks = []
        prev = in_dim
        for d in self.dilations:
            blocks.append(nn.ModuleDict({
                'conv': nn.Conv1d(prev, channels, kernel, dilation=d),
                'norm': _ChannelLayerNorm(channels),
            }))
            prev = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(prev, out_dim, 1)

    @property
    def receptive_field(self):
        return 1 + sum((self.kernel - 1) * d for d in self.dilations)

    def forward(self, x):
        for blk, d in zip(self.blocks, self.dilations):
            x = F.pad(x, ((self.kernel - 1) * d, 0))  # left pad only -> causal
            x = F.silu(blk['norm'](blk['conv'](x)))
        return self.head(x)


class TemporalCorrector(Stage2Corrector):
    """Dilated causal TCN over the per-timestep feature stream.

    Trained on random crops sampled inside trajectories. The first `warmup`
    positions of each crop are excluded from the loss so the model is never
    scored on outputs whose history was truncated by the crop boundary.
    """

    kind = 'neural'

    def __init__(self, crop=256, batch_size=32, epochs=8, lr=3e-3,
                 device='auto', seed=42, verbose=True, chunk=4096, **arch_kwargs):
        self.name = 'E_tcn'
        self.arch_kwargs = arch_kwargs
        self.crop = crop
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = _resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.chunk = chunk
        self.scaler = Standardizer()
        self.model = None

    def _build(self, in_dim):
        return _CausalTCN(in_dim, **self.arch_kwargs)

    @property
    def _warmup(self):
        return min(int(self.model.receptive_field), self.crop // 2)

    def _crop_starts(self, data):
        """Every row index at which a full crop fits without crossing a boundary."""
        spans = [np.arange(a, b - self.crop + 1) for a, b in data['segments']
                 if b - a >= self.crop]
        return np.concatenate(spans) if spans else np.empty(0, dtype=np.int64)

    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self._remember_features(train)
        self.scaler.fit(train['X'])
        Xs = self.scaler.transform(train['X'])
        y = train['y']
        self.model = self._build(Xs.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        starts = self._crop_starts(train)
        if starts.size == 0:
            raise RuntimeError('no trajectory is long enough for crop=%d' % self.crop)
        # One epoch = roughly one pass over the timesteps, not over crop offsets
        # (consecutive offsets overlap almost entirely).
        per_epoch = max(1, Xs.shape[0] // (self.batch_size * self.crop))
        warm = self._warmup

        for ep in range(self.epochs):
            self.model.train()
            total, seen = 0.0, 0
            for _ in range(per_epoch):
                picks = starts[rng.integers(0, starts.size, size=self.batch_size)]
                xb = np.stack([Xs[s:s + self.crop] for s in picks])
                yb = np.stack([y[s:s + self.crop] for s in picks])
                xb = torch.from_numpy(xb).permute(0, 2, 1).to(self.device)
                yb = torch.from_numpy(yb).permute(0, 2, 1).to(self.device)

                out = self.model(xb)
                loss = F.mse_loss(out[:, :, warm:], yb[:, :, warm:])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * xb.shape[0]
                seen += xb.shape[0]
            if self.verbose:
                msg = '    [%s] epoch %2d/%d  train_mse=%.6f' % (
                    self.name, ep + 1, self.epochs, total / seen)
                if val is not None:
                    msg += '  val_mse=%.6f' % residual_mse(val, self.predict_residual(val))[2]
                print(msg)
        return self

    @torch.no_grad()
    def predict_residual(self, data):
        """Run each trajectory independently, so no history crosses a boundary."""
        self.model.eval()
        Xs = self.scaler.transform(self._X(data))
        out = np.zeros((Xs.shape[0], 2), dtype=np.float32)
        ctx = int(self.model.receptive_field) - 1

        for a, b in data['segments']:
            seg = Xs[a:b]
            n = seg.shape[0]
            pos = 0
            # Chunk with rf-1 steps of left context, which makes the chunked
            # result identical to a single pass over the trajectory.
            while pos < n:
                stop = min(pos + self.chunk, n)
                lo = max(0, pos - ctx)
                xb = torch.from_numpy(seg[lo:stop]).T.unsqueeze(0).to(self.device)
                res = self.model(xb)[0].T.cpu().numpy()
                out[a + pos:a + stop] = res[pos - lo:]
                pos = stop
        return out

    def size_bytes(self):
        n = _count_params(self.model)
        n += self.scaler.mean.size + self.scaler.std.size
        return int(n) * 4

    def params(self):
        return {'channels': self.model.blocks[0]['conv'].out_channels,
                'kernel': self.model.kernel, 'dilations': list(self.model.dilations),
                'receptive_field': int(self.model.receptive_field),
                'crop': self.crop, 'batch_size': self.batch_size,
                'epochs': self.epochs, 'lr': self.lr,
                'n_parameters': _count_params(self.model)}

    def describe(self):
        return 'TCN(params=%d, receptive_field=%d)' % (
            _count_params(self.model), self.model.receptive_field)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def residual_mse(data, residual, alpha=1.0, clip=None):
    """MSE of the corrected velocity against ground truth."""
    corrected = apply_rf_correction(data['pred'], residual, alpha=alpha, residual_clip=clip)
    err = data['gt'] - corrected
    mse_x = float(np.mean(err[:, 0] ** 2))
    mse_y = float(np.mean(err[:, 1] ** 2))
    return mse_x, mse_y, (mse_x + mse_y) / 2.0


def raw_mse(data):
    return float(np.mean((data['gt'] - data['pred']) ** 2))


def tune_alpha_clip(data, residual, alphas=(0.25, 0.5, 0.75, 1.0, 1.25), clips=(None, 0.25, 0.5, 0.75)):
    """Pick the correction gain and clip bound on the validation split.

    Tuned per variant: reusing the RF's alpha/clip for a differently calibrated
    corrector would understate it.
    """
    best = None
    for a in alphas:
        for c in clips:
            mse = residual_mse(data, residual, alpha=a, clip=c)[2]
            if best is None or mse < best['mse']:
                best = {'alpha': a, 'clip': c, 'mse': mse}
    return best


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def build_variant(key, **overrides):
    key = key.upper()
    factories = {
        'A': lambda: RandomForestCorrector(**overrides),
        'B': lambda: RidgeCorrector(**overrides),
        'C': lambda: CausalFilterCorrector(**overrides),
        'D': lambda: MLPCorrector(**overrides),
        'E': lambda: TemporalCorrector(**overrides),
    }
    if key not in factories:
        raise ValueError('unknown variant %r; choose from %s' % (key, sorted(factories)))
    return factories[key]()


ALL_VARIANTS = ('B', 'C', 'D', 'E', 'A')
NEURAL_VARIANTS = ('D', 'E')


def fmt_size(nbytes):
    if nbytes < 1024:
        return '%d B' % nbytes
    if nbytes < 1024 ** 2:
        return '%.1f KiB' % (nbytes / 1024)
    return '%.1f MiB' % (nbytes / 1024 ** 2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Train and compare stage-2 residual-correction variants.')
    parser.add_argument('--train_npz', type=str, required=True)
    parser.add_argument('--val_npz', type=str, required=True)
    parser.add_argument('--variants', type=str, default='all',
                        help='comma list from A,B,C,D,E (default: all)')
    parser.add_argument('--drop_non_causal', action='store_true',
                        help='drop time_progress, which needs the sequence end')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override epochs for the neural variants (D, E)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--subsample', type=int, default=0,
                        help='use only the first N train rows (smoke test)')
    parser.add_argument('--no_tune', action='store_true',
                        help='skip the per-variant alpha/clip sweep')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='if set, write fitted models and a JSON summary here')
    args = parser.parse_args()

    drop = NON_CAUSAL_FEATURES if args.drop_non_causal else ()
    train = load_stage2_npz(args.train_npz, drop_features=drop)
    val = load_stage2_npz(args.val_npz, drop_features=drop)

    if args.subsample and args.subsample < train['X'].shape[0]:
        n = args.subsample
        keep = [(a, min(b, n)) for a, b in train['segments'] if a < n]
        train = dict(train)
        for k in ('X', 'y', 'pred', 'gt', 'seq_name', 'ts'):
            train[k] = train[k][:n]
        train['segments'] = keep

    print('train: X=%s  y=%s  %d trajectories' % (
        train['X'].shape, train['y'].shape, len(train['segments'])))
    print('val:   X=%s  y=%s  %d trajectories' % (
        val['X'].shape, val['y'].shape, len(val['segments'])))
    print('features (%d): %s' % (len(val['feature_names']), ', '.join(val['feature_names'])))

    base = raw_mse(val)
    print('\nval MSE with no correction: %.6f\n' % base)

    keys = ALL_VARIANTS if args.variants.lower() == 'all' else \
        [k.strip() for k in args.variants.split(',') if k.strip()]

    rows = []
    for key in keys:
        overrides = {}
        if key.upper() in NEURAL_VARIANTS:
            overrides['device'] = args.device
            if args.epochs is not None:
                overrides['epochs'] = args.epochs

        print('-' * 78)
        print('variant %s' % key)
        model = build_variant(key, **overrides)
        t0 = time.time()
        model.fit(train, val=val)
        fit_s = time.time() - t0

        t0 = time.time()
        residual = model.predict_residual(val)
        infer_s = time.time() - t0

        if residual.shape != val['y'].shape:
            raise RuntimeError('%s returned %s, expected %s' % (
                model.name, residual.shape, val['y'].shape))
        if not np.isfinite(residual).all():
            raise RuntimeError('%s produced non-finite residuals' % model.name)

        plain = residual_mse(val, residual, alpha=1.0, clip=None)
        tuned = None if args.no_tune else tune_alpha_clip(val, residual)

        print('  %s' % model.describe())
        print('  size            : %s' % fmt_size(model.size_bytes()))
        print('  fit time        : %.1f s   predict time: %.2f s' % (fit_s, infer_s))
        print('  val MSE (a=1)   : %.6f  (%+.2f%%)' % (plain[2], 100 * (plain[2] - base) / base))
        if tuned:
            print('  val MSE (tuned) : %.6f  (%+.2f%%)  alpha=%.2f clip=%s' % (
                tuned['mse'], 100 * (tuned['mse'] - base) / base, tuned['alpha'], tuned['clip']))

        # Carry the tuned gain with the model so the trajectory evaluation can
        # default to it instead of reusing the RF's alpha/clip.
        model.best_alpha = tuned['alpha'] if tuned else 1.0
        model.best_clip = tuned['clip'] if tuned else None

        rows.append({
            'variant': key,
            'name': model.name,
            'description': model.describe(),
            'size_bytes': model.size_bytes(),
            'fit_seconds': fit_s,
            'predict_seconds': infer_s,
            'val_mse_alpha1': plain[2],
            'val_mse_x': plain[0],
            'val_mse_y': plain[1],
            'val_mse_tuned': tuned['mse'] if tuned else None,
            'best_alpha': model.best_alpha,
            'best_clip': model.best_clip,
        })

        if args.out_dir:
            # Same layout as the existing RF stage 2:
            #   rf_corrector.joblib / rf_corrector_summary.json
            #   -> <name>_corrector.joblib / <name>_corrector_summary.json
            stem = osp.join(args.out_dir, '%s_corrector' % model.name)
            path = save_corrector(model, stem + '.joblib')
            variant_summary = {
                'train_npz': args.train_npz,
                'val_npz': args.val_npz,
                'variant': key.upper(),
                'name': model.name,
                'model': model.describe(),
                'output_mode': 'multi',
                'best_params': model.params(),
                # Same definition as rf_corrector_summary.json: MSE of the
                # residual left after correction, at alpha=1 with no clipping.
                'val_mse_x': plain[0],
                'val_mse_y': plain[1],
                'val_mse_avg': plain[2],
                'baseline_zero_residual_mse_avg': base,
                'val_mse_avg_tuned': tuned['mse'] if tuned else None,
                'correction_alpha': model.best_alpha,
                'correction_clip': model.best_clip,
                'size_bytes': model.size_bytes(),
                'fit_seconds': fit_s,
                'predict_seconds': infer_s,
                'feature_names': model.feature_names,
            }
            with open(stem + '_summary.json', 'w') as f:
                json.dump(variant_summary, f, indent=2, default=str)
            print('  saved           : %s' % path)
            print('  summary         : %s' % (stem + '_summary.json'))

    print('=' * 78)
    print('%-9s %-11s %12s %12s   %s' % ('variant', 'size', 'MSE(a=1)', 'MSE(tuned)', 'vs raw'))
    print('%-9s %-11s %12.6f %12s   %s' % ('raw', '-', base, '-', '-'))
    for r in sorted(rows, key=lambda z: z['val_mse_tuned'] or z['val_mse_alpha1']):
        best = r['val_mse_tuned'] or r['val_mse_alpha1']
        print('%-9s %-11s %12.6f %12s   %+.2f%%' % (
            r['variant'], fmt_size(r['size_bytes']), r['val_mse_alpha1'],
            '%.6f' % r['val_mse_tuned'] if r['val_mse_tuned'] else '-',
            100 * (best - base) / base))

    if args.out_dir:
        # Cross-variant comparison, alongside the per-variant *_corrector_summary.json.
        os.makedirs(args.out_dir, exist_ok=True)
        summary = {'raw_val_mse': base, 'drop_non_causal': bool(drop),
                   'feature_names': val['feature_names'], 'results': rows}
        path = osp.join(args.out_dir, 'stage2_comparison.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print('\nwrote %s' % path)


if __name__ == '__main__':
    # Re-enter through the module namespace before doing any work. Running this
    # file as a script puts the classes in __main__, so pickled correctors would
    # record their type as __main__.RidgeCorrector and fail to load anywhere
    # else - including ronin_yolo26_baseline_plain.py. Importing the module and
    # calling its main() makes saved artifacts portable.
    from stage2_variants import main as _module_main

    _module_main()
