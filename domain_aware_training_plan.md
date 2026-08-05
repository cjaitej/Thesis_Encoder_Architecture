# Domain-Aware Training Plan: RIDI Pretraining → RoNIN Fine-Tuning
# Fixes: BN Reset (Fix 1) + Input Normalisation (Fix 2) + Discriminative LR (Fix 3)

> **Root Cause:** RIDI BatchNorm running stats mismatch RoNIN signal variance (gyro_z: 0.685 vs 1.029).
> **Strategy:** Reset BN stats + per-channel z-score normalisation + discriminative LR across 4 layer groups.

---

## Files Overview

```
ronin/source/
├── model_yolo26_1d.py          ← MODIFY: add reset_bn_stats() utility
├── ronin_resnet.py             ← MODIFY: normalisation + BN reset + disc. LR
├── compute_ronin_stats.py      ← NEW: compute RoNIN channel mean/std once
├── compute_ridi_stats.py       ← NEW: compute RIDI channel mean/std once
├── test_smoke.py               ← UNCHANGED
└── extract_backbone.py         ← UNCHANGED

ronin/output/
├── pretrain_ridi/
│   └── ridi_pretrained_full.pt       ← existing pretrained checkpoint
├── stats/
│   ├── ronin_channel_stats.npy       ← NEW (Stage 1)
│   └── ridi_channel_stats.npy        ← NEW (Stage 1)
└── finetune_domain_aware/
    └── checkpoints/                  ← NEW (Stage 4)
```

---

## Stage 0 — Verify Existing Pretrained Checkpoint

- [ ] Confirm `ridi_pretrained_full.pt` exists:
```bash
ls -lh ../output/pretrain_ridi/ridi_pretrained_full.pt
```
- [ ] Confirm it loads correctly:
```python
import torch
ckpt = torch.load('../output/pretrain_ridi/ridi_pretrained_full.pt', map_location='cpu')
print("Keys         :", list(ckpt.keys()))
print("Epochs trained:", ckpt.get('epochs_pretrained'))
print("RIDI val loss :", ckpt.get('ridi_val_loss'))
```
- [ ] If checkpoint is missing or corrupt — re-run RIDI pretraining (20 epochs, Adam lr=0.0001, batch=64) before proceeding.

> ✅ **Success Criterion:** Checkpoint loads. `model_state_dict` is present in keys.

---

## Stage 1 — Compute Channel Statistics (Fix 2 Prerequisite)

> Run ONCE. These stats files are reused in every subsequent training run.

### 1.1 — Create `source/compute_ronin_stats.py`

```python
"""
Computes per-channel mean and std from RoNIN training set.
Channels: [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z]
Run once. Output saved to ../output/stats/ronin_channel_stats.npy
"""
import numpy as np, h5py, glob, os

DATA_DIR  = '../data/train_dataset'
OUT_PATH  = '../output/stats/ronin_channel_stats.npy'
os.makedirs('../output/stats', exist_ok=True)

ch_sum  = np.zeros(6)
ch_sq   = np.zeros(6)
total_n = 0

files = glob.glob(f'{DATA_DIR}/**/*.hdf5', recursive=True)
print(f"Found {len(files)} HDF5 files...")

for path in files:
    try:
        with h5py.File(path, 'r') as f:
            acce = np.array(f['synced/acce'][:, :3])   # (T, 3)
            gyro = np.array(f['synced/gyro'][:, :3])   # (T, 3)
            imu  = np.concatenate([acce, gyro], axis=1) # (T, 6)
            ch_sum  += imu.sum(axis=0)
            ch_sq   += (imu**2).sum(axis=0)
            total_n += imu.shape[0]
    except Exception as e:
        print(f"  Skipping {path}: {e}")

mean = ch_sum / total_n
std  = np.sqrt(np.maximum(ch_sq / total_n - mean**2, 1e-8))

np.save(OUT_PATH, {'mean': mean, 'std': std})
print(f"
RoNIN stats saved to {OUT_PATH}")
print(f"Frames processed : {total_n:,}")
print(f"Channel mean     : {np.round(mean, 4)}")
print(f"Channel std      : {np.round(std,  4)}")
print(f"Labels           : [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z]")
```

Run:
```bash
cd ronin/source && python compute_ronin_stats.py
```

### 1.2 — Create `source/compute_ridi_stats.py`

```python
"""
Computes per-channel mean and std from RIDI training set.
Run once. Output saved to ../output/stats/ridi_channel_stats.npy
"""
import numpy as np, h5py, glob, os

DATA_DIR = '../data/ridi_data'
OUT_PATH = '../output/stats/ridi_channel_stats.npy'
os.makedirs('../output/stats', exist_ok=True)

ch_sum  = np.zeros(6)
ch_sq   = np.zeros(6)
total_n = 0

files = glob.glob(f'{DATA_DIR}/**/*.hdf5', recursive=True)
print(f"Found {len(files)} HDF5 files...")

for path in files:
    try:
        with h5py.File(path, 'r') as f:
            acce = np.array(f['synced/acce'][:, :3])
            gyro = np.array(f['synced/gyro'][:, :3])
            imu  = np.concatenate([acce, gyro], axis=1)
            ch_sum  += imu.sum(axis=0)
            ch_sq   += (imu**2).sum(axis=0)
            total_n += imu.shape[0]
    except Exception as e:
        print(f"  Skipping {path}: {e}")

mean = ch_sum / total_n
std  = np.sqrt(np.maximum(ch_sq / total_n - mean**2, 1e-8))

np.save(OUT_PATH, {'mean': mean, 'std': std})
print(f"
RIDI stats saved to {OUT_PATH}")
print(f"Frames processed : {total_n:,}")
print(f"Channel mean     : {np.round(mean, 4)}")
print(f"Channel std      : {np.round(std,  4)}")
```

Run:
```bash
cd ronin/source && python compute_ridi_stats.py
```

### 1.3 — Verify KL divergence drops after normalisation

```python
# Quick check: after normalisation both datasets should have std ≈ 1.0
import numpy as np
ronin = np.load('../output/stats/ronin_channel_stats.npy', allow_pickle=True).item()
ridi  = np.load('../output/stats/ridi_channel_stats.npy',  allow_pickle=True).item()
print("RoNIN std after normalisation will be: 1.0 (by construction)")
print("RIDI  std after normalisation will be: 1.0 (by construction)")
print("Remaining distribution shift will be only in SHAPE, not scale")
print(f"RoNIN mean: {np.round(ronin['mean'], 4)}")
print(f"RIDI  mean: {np.round(ridi['mean'],  4)}")
print(f"Mean shift (should be small): {np.round(ronin['mean'] - ridi['mean'], 4)}")
```

> ✅ **Success Criterion:** Both `ronin_channel_stats.npy` and `ridi_channel_stats.npy`
> exist. Each contains `mean` and `std` arrays of shape `(6,)`.

---

## Stage 2 — Add `reset_bn_stats()` to `model_yolo26_1d.py` (Fix 1)

Open `source/model_yolo26_1d.py` and add this function at the bottom of the file,
AFTER the `get_model()` function:

```python
def reset_bn_stats(model, new_momentum=0.1):
    """
    Fix 1: Reset all BatchNorm1d running statistics.

    Called after loading RIDI pretrained checkpoint and before RoNIN fine-tuning.
    Forces BN layers to recalibrate their running mean/var to RoNIN distribution
    instead of inheriting RIDI's narrower signal statistics.

    Args:
        model:        YOLO26_1D_Regressor instance
        new_momentum: BN update speed. 0.1 = fast recalibration (default PyTorch).
                      Was 0.03 during pretraining (slow). Increase for fine-tuning.
    """
    reset_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()   # sets running_mean=0, running_var=1
            module.momentum = new_momentum
            reset_count += 1
    print(f"[BN Reset] Reset {reset_count} BatchNorm1d layers. "
          f"New momentum={new_momentum}")
    return model
```

- [ ] Confirm the function is importable:
```bash
python3 -c "from model_yolo26_1d import reset_bn_stats; print('OK')"
```

> ✅ **Success Criterion:** Import succeeds. No errors.

---

## Stage 3 — RIDI Pretraining WITH Normalisation

> If `ridi_pretrained_full.pt` from previous run is valid — **skip this stage**.
> Only redo pretraining if the checkpoint is missing or was trained WITHOUT normalisation.

### 3.1 — Add normalisation loading to `ronin_resnet.py`

Find the section where the dataloader/feature loading happens. Add:

```python
import numpy as np

# ── Load dataset-specific channel stats for normalisation (Fix 2) ────────
# Set STATS_PATH to the correct stats file for the current dataset
STATS_PATH = '../output/stats/ridi_channel_stats.npy'   # ← RIDI for pretraining
stats      = np.load(STATS_PATH, allow_pickle=True).item()
NORM_MEAN  = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 6, 1)
NORM_STD   = torch.tensor(stats['std'],  dtype=torch.float32).view(1, 6, 1)
print(f"[Normalisation] Loaded stats from: {STATS_PATH}")
print(f"[Normalisation] Channel mean: {np.round(stats['mean'], 4)}")
print(f"[Normalisation] Channel std : {np.round(stats['std'],  4)}")
# ─────────────────────────────────────────────────────────────────────────
```

### 3.2 — Apply normalisation in training loop

Find the line where `feat` (input tensor) is sent to the network inside the train loop.
Add normalisation IMMEDIATELY before the forward pass:

```python
# Inside train loop, before: pred = network(feat)
feat = (feat - NORM_MEAN.to(device)) / (NORM_STD.to(device) + 1e-8)
pred = network(feat)
```

Apply the same in the validation loop:
```python
# Inside val loop, before: pred = network(feat)
feat = (feat - NORM_MEAN.to(device)) / (NORM_STD.to(device) + 1e-8)
pred = network(feat)
```

### 3.3 — Run RIDI pretraining with normalisation

```bash
cd ronin/source
python ronin_resnet.py     --mode train     --dataset ridi     --root_dir ../data/ridi_data     --train_list ../lists/ridi_train.txt     --val_list ../lists/ridi_val.txt     --cache_path /tmp/ridi_cache     --out_dir ../output/pretrain_ridi_v2     --epochs 20     --batch_size 64
```

> **Note:** Pretraining with RIDI-normalised inputs means the model learns weights
> calibrated to unit-variance signals. When we fine-tune on RoNIN (also normalised
> to unit-variance), the BN reset just recalibrates residual mean shifts —
> the variance mismatch is already eliminated at input level.

### 3.4 — Extract backbone from new pretrained checkpoint

```bash
# Update extract_backbone.py to point to new pretrain dir:
python source/extract_backbone.py     --ckpt ../output/pretrain_ridi_v2/checkpoints/checkpoint_best.pt     --out  ../output/pretrain_ridi_v2/ridi_pretrained_full_v2.pt
```

> ✅ **Success Criterion:** `ridi_pretrained_full_v2.pt` exists.
> Verify: `python3 -c "import torch; c=torch.load('...v2.pt'); print(c.keys())"`

---

## Stage 4 — RoNIN Fine-Tuning (All 3 Fixes Combined)

### 4.1 — Update `STATS_PATH` in `ronin_resnet.py` for RoNIN

Change the one line set in Stage 3.1:

```python
# BEFORE (pretraining):
STATS_PATH = '../output/stats/ridi_channel_stats.npy'

# AFTER (fine-tuning):
STATS_PATH = '../output/stats/ronin_channel_stats.npy'   # ← RoNIN stats now
```

> This is the ONLY line that changes between pretraining and fine-tuning
> for the normalisation setup.

### 4.2 — Update checkpoint loading block in `ronin_resnet.py`

Find the RIDI checkpoint loading block added previously. Replace entirely with:

```python
# ── Load RIDI pretrained weights (domain-aware fine-tuning) ──────────
from model_yolo26_1d import reset_bn_stats

_pretrain_path = '../output/pretrain_ridi_v2/ridi_pretrained_full_v2.pt'
_state = torch.load(_pretrain_path, map_location='cpu')
network.load_state_dict(_state['model_state_dict'])
print(f"[YOLO26-1D] Loaded RIDI pretrained weights")
print(f"[YOLO26-1D] Pretrained for {_state['epochs_pretrained']} epochs")

# Fix 1: Reset BN running stats so they recalibrate to RoNIN distribution
network = reset_bn_stats(network, new_momentum=0.1)

# Sanity check
_w = list(network.parameters())[0]
assert _w.abs().mean().item() > 1e-5, "ERROR: Weights look uninitialized"
print("[YOLO26-1D] Pretrained weights verified. BN stats reset. Ready.")
# ─────────────────────────────────────────────────────────────────────
```

### 4.3 — Replace optimizer with discriminative LR groups (Fix 3)

Find the optimizer line in `ronin_resnet.py`. Replace with:

```python
# ── Fix 3: Discriminative learning rates per layer group ─────────────
# Geometric decay: each group gets 10x larger lr than the previous.
# Stem is closest to raw signal — most affected by distribution shift → slowest.
# Head is task-specific and needs to adapt fastest → fastest.

_param_groups = [
    {
        'params' : list(network.stem.parameters()),
        'lr'     : 1e-6,
        'name'   : 'stem'
    },
    {
        'params' : list(network.stage1.parameters()) +
                   list(network.stage2.parameters()),
        'lr'     : 5e-6,
        'name'   : 'early_stages'
    },
    {
        'params' : list(network.stage3.parameters()) +
                   list(network.neck.parameters()),
        'lr'     : 1e-5,
        'name'   : 'deep_stages'
    },
    {
        'params' : list(network.head.parameters()),
        'lr'     : 1e-4,
        'name'   : 'head'
    },
]

# PSA attention (if present) — treat same as deep stages
if hasattr(network, 'psa'):
    _param_groups[2]['params'] += list(network.psa.parameters())

optimizer = torch.optim.Adam(_param_groups, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

print("[Optimizer] Discriminative LR groups:")
for g in _param_groups:
    n_params = sum(p.numel() for p in g['params'])
    print(f"  {g['name']:15s}  lr={g['lr']:.0e}  params={n_params:,}")
# ─────────────────────────────────────────────────────────────────────
```

### 4.4 — Add LR group logging to validation step

After each epoch, log the current lr per group to track scheduler behaviour:

```python
# Add at end of each epoch (after scheduler.step):
current_lrs = [g['lr'] for g in optimizer.param_groups]
print(f"  LR groups: stem={current_lrs[0]:.2e} | "
      f"early={current_lrs[1]:.2e} | "
      f"deep={current_lrs[2]:.2e} | "
      f"head={current_lrs[3]:.2e}")
```

### 4.5 — Run domain-aware fine-tuning

```bash
cd ronin/source
python ronin_resnet.py     --mode train     --dataset ronin     --root_dir ../data/train_dataset     --train_list ../lists/list_train.txt     --val_list ../lists/list_val.txt     --cache_path /tmp/ronin_cache     --out_dir ../output/finetune_domain_aware     --epochs 100     --batch_size 128
```

### 4.6 — Monitor first 10 epochs for health signals

| Epoch | Healthy Signal | Problem Signal | Action |
|-------|---------------|----------------|--------|
| 0 | Val loss drops from init | Val loss rises or stays same | BN reset may not have applied — check Step 4.2 |
| 1–5 | Both losses descending | Train descends but val rises | Head lr=1e-4 too aggressive — reduce to 5e-5 |
| 5–15 | Train/val gap < 0.012 | Gap > 0.015 and growing | Overfitting — reduce head lr to 1e-5 |
| Any | LR group printout shows 4 different values | All same lr | Discriminative LR not applied — check Step 4.3 |
| Any | Loss = NaN | — | Head lr too high — reduce to 5e-5, restart |

> ✅ **Success Criterion:** Val loss at epoch 10 is lower than val loss at
> epoch 10 of previous run (target: below 0.038). LR groups show 4 different values.

---

## Stage 5 — Evaluation

### 5.1 — Find best checkpoint

```bash
# List all checkpoints by modification time
ls -lt ../output/finetune_domain_aware/checkpoints/ | head -20

# Or find the best programmatically:
python3 -c "
import os, torch, glob
ckpts = glob.glob('../output/finetune_domain_aware/checkpoints/checkpoint_*.pt')
best_loss, best_path = float('inf'), None
for p in ckpts:
    try:
        c = torch.load(p, map_location='cpu')
        loss = c.get('val_loss', float('inf'))
        if loss < best_loss:
            best_loss, best_path = loss, p
    except: pass
print(f'Best checkpoint: {best_path}')
print(f'Best val loss:   {best_loss:.6f}')
"
```

### 5.2 — IMPORTANT: Apply normalisation during test too

The test run must use the SAME normalisation as training. Confirm `STATS_PATH` in
`ronin_resnet.py` still points to `ronin_channel_stats.npy` before running test.

```python
# Verify in ronin_resnet.py before test run:
STATS_PATH = '../output/stats/ronin_channel_stats.npy'   # must be RoNIN stats
```

### 5.3 — Run evaluation on seen subjects

```bash
python ronin_resnet.py     --mode test     --model_path ../output/finetune_domain_aware/checkpoints/<best_checkpoint>.pt     --data_dir ../data/seen_subjects_test_set     --out_dir ../output/eval_domain_aware_seen
```

### 5.4 — Run evaluation on unseen subjects

```bash
python ronin_resnet.py     --mode test     --model_path ../output/finetune_domain_aware/checkpoints/<best_checkpoint>.pt     --data_dir ../data/unseen_subjects_test_set     --out_dir ../output/eval_domain_aware_unseen
```

### 5.5 — Fill results table

| Model | ATE Seen | ATE Unseen | RTE Seen | RTE Unseen |
|-------|---------|------------|---------|------------|
| RoNIN-ResNet (paper, full data) | 3.54 | 5.14 | 2.67 | 4.37 |
| YOLO26-1D v1 (no fixes) | ___ | ___ | ___ | ___ |
| YOLO26-1D v2 (BN+norm+disc.LR) | ___ | ___ | ___ | ___ |

---

## Quick Reference Card

| Parameter | RIDI Pretraining (Stage 3) | RoNIN Fine-Tuning (Stage 4) |
|-----------|--------------------------|---------------------------|
| STATS_PATH | `ridi_channel_stats.npy` | `ronin_channel_stats.npy` |
| BN reset | Not needed | YES — `reset_bn_stats()` |
| Optimizer | Adam, single lr=0.0001 | Adam, 4 discriminative LRs |
| LR stem | 0.0001 | **1e-6** |
| LR early stages | 0.0001 | **5e-6** |
| LR deep stages | 0.0001 | **1e-5** |
| LR head | 0.0001 | **1e-4** |
| Scheduler patience | 5 | 10 |
| Epochs | 20 | 100 |
| Batch size | 64 | 128 |

---

## Files Changed Summary

| File | Change | Stage |
|------|--------|-------|
| `model_yolo26_1d.py` | Add `reset_bn_stats()` at bottom | Stage 2 |
| `compute_ronin_stats.py` | NEW — run once | Stage 1 |
| `compute_ridi_stats.py` | NEW — run once | Stage 1 |
| `ronin_resnet.py` | Add STATS_PATH + NORM_MEAN/STD | Stage 3.1 |
| `ronin_resnet.py` | Apply normalisation in train/val loop | Stage 3.2 |
| `ronin_resnet.py` | Update checkpoint loader + call reset_bn_stats() | Stage 4.2 |
| `ronin_resnet.py` | Replace optimizer with discriminative LR groups | Stage 4.3 |
| `ronin_resnet.py` | STATS_PATH switch from ridi → ronin stats | Stage 4.1 |
| Everything else | UNCHANGED | — |

---

## Error Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: synced/acce` | HDF5 key path wrong for your data version | Check actual key with `h5py`: `list(f['synced'].keys())` |
| `shape mismatch` in normalisation | NORM_MEAN/STD wrong shape | Confirm `.view(1, 6, 1)` — must match `(B, 6, T)` input |
| `All LRs same in log` | Discriminative LR groups not applied | Check optimizer was replaced, not just lr changed |
| `Val loss same as v1` | STATS_PATH not switched to RoNIN stats for fine-tune | Check Stage 4.1 |
| `NaN loss epoch 0` | Head lr=1e-4 too high with BN reset | Reduce head lr to 5e-5 |
| `reset_bn_stats not found` | Function not added to model_yolo26_1d.py | Check Stage 2 |
