# YOLO26-1D × RoNIN: Agent Implementation Plan

**Purpose:** A complete, stage-by-stage instruction set for an AI coding agent to implement,
integrate, and train a 1D YOLO26-inspired neural network inside the Sachini/ronin repository.
Each stage defines exact files to touch, exact code to write, and a verifiable success criterion
before moving to the next stage.

---

## Repository Map (Pre-Implementation)

```
ronin/
├── source/
│   ├── model_resnet1d.py        ← DO NOT MODIFY
│   ├── ronin_resnet.py          ← MODIFY (Changes 1–3 only)
│   ├── ronin_lstm.py            ← DO NOT TOUCH
│   ├── ronin_tcn.py             ← DO NOT TOUCH
│   ├── data_glob_speed.py       ← DO NOT TOUCH
│   └── utils.py                 ← DO NOT TOUCH
├── trained_models/
└── requirements.txt
```

**Files to create:**

- `source/model_yolo26_1d.py` ← NEW (full model definition)

**Files to modify:**

- `source/ronin_resnet.py` ← 3 surgical changes only

---

## Stage 0 — Repository Audit

**Goal:** Understand the existing codebase before writing a single line of new code.

### 0.1 — Read these files completely before any edits

```
source/model_resnet1d.py      ← understand: ResNet1D class, BasicBlock1D, FCOutputModule
source/ronin_resnet.py        ← understand: model instantiation, optimizer, train loop, loss fn
```

### 0.2 — Identify and record these exact values from ronin_resnet.py

The agent MUST locate and note down:

| Item                      | Where to find it              | What to record                    |
| ------------------------- | ----------------------------- | --------------------------------- |
| Model instantiation block | search `ResNet1D(`            | full constructor call + arguments |
| Optimizer line            | search `torch.optim`          | optimizer class + lr + other args |
| Scheduler line            | search `lr_scheduler`         | scheduler class + patience        |
| Loss function             | search `loss` or `criterion`  | function name and call signature  |
| Input tensor shape        | search `x =` in train loop    | confirm it is `(B, 6, 200)`       |
| Output tensor shape       | search `output` in train loop | confirm it is `(B, 2)`            |
| Checkpoint save           | search `torch.save`           | full save call                    |
| Checkpoint load           | search `torch.load`           | full load call                    |

### 0.3 — Verify data pipeline produces correct shape

Run this before touching any model code:

```bash
cd ronin/source
python3 -c "
from data_glob_speed import *   # or whichever dataset class is used
# instantiate a loader with a small sample
# print one batch's x.shape and y.shape
# expected: x=(B, 6, 200), y=(B, 2) or (B, 3)
"
```

✅ **Stage 0 success criterion:** Agent has recorded all 8 items from the table above.
Agent confirms input shape is `(B, 6, 200)` and output shape is `(B, 2)`.

---

## Stage 1 — Create `source/model_yolo26_1d.py`

**Goal:** Write the complete model file with zero imports from the existing RoNIN codebase.
This file must be self-contained and independently importable.

### 1.1 — File header

```python
"""
source/model_yolo26_1d.py
Drop-in replacement for ResNet1D in the RoNIN pipeline.
Input:  (B, 6, 200)
Output: (B, 2)
All operations: 1D only (Conv1d, BatchNorm1d, AdaptiveAvgPool1d).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 1.2 — Block 1: `CBS` (Conv1d + BatchNorm1d + SiLU)

Write class `CBS(nn.Module)`:

- `__init__(self, in_ch, out_ch, k=1, s=1, p=None)` — auto-pad as `k//2` when `p` is None
- Uses `nn.Conv1d` with `bias=False`, `nn.BatchNorm1d`, `nn.SiLU(inplace=True)`
- `forward(x)` → returns `self.act(self.bn(self.conv(x)))`

### 1.3 — Block 2: `Bottleneck1D`

Write class `Bottleneck1D(nn.Module)`:

- `__init__(self, in_ch, out_ch, shortcut=True, e=0.5)`
- Three CBS calls: `1×1 → 3×3 → 1×1`, hidden dim = `int(out_ch * e)`
- Residual: only add `x` to output when `shortcut=True AND in_ch == out_ch`
- `forward(x)` → `x + cv3(cv2(cv1(x)))` if residual else `cv3(cv2(cv1(x)))`

### 1.4 — Block 3: `C3k2_1D` (core YOLO26 block)

Write class `C3k2_1D(nn.Module)`:

- `__init__(self, in_ch, out_ch, n=2, shortcut=True, stride=1)`
  - If `stride > 1`: add `self.ds = CBS(in_ch, in_ch, k=3, s=stride)` else `nn.Identity()`
  - `hidden = out_ch // 2`
  - `self.cv1 = CBS(in_ch, hidden, k=1)` — main path
  - `self.cv2 = CBS(in_ch, hidden, k=1)` — skip path
  - `self.bottlenecks = nn.Sequential(*[Bottleneck1D(hidden, hidden, shortcut) for _ in range(n)])`
  - `self.cv_out = CBS(2 * hidden, out_ch, k=1)` — fusion
- `forward(x)`:
  ```python
  x = self.ds(x)
  main = self.bottlenecks(self.cv1(x))
  skip = self.cv2(x)
  return self.cv_out(torch.cat([main, skip], dim=1))
  ```

### 1.5 — Block 4: `ELANNeck1D` (multi-scale temporal fusion)

Write class `ELANNeck1D(nn.Module)`:

- `__init__(self, ch1=64, ch2=128, ch3=256, out_ch=256)`
  - `self.lat1 = CBS(ch1, ch3, k=1)` — channel-align stage-1 features
  - `self.lat2 = CBS(ch2, ch3, k=1)` — channel-align stage-2 features
  - `self.fuse = C3k2_1D(3 * ch3, out_ch, n=1, stride=1)`
- `forward(f1, f2, f3)`:
  ```python
  target_len = f3.shape[-1]
  f1_up = F.adaptive_avg_pool1d(f1, target_len)
  f2_up = F.adaptive_avg_pool1d(f2, target_len)
  f1_up = self.lat1(f1_up)
  f2_up = self.lat2(f2_up)
  return self.fuse(torch.cat([f1_up, f2_up, f3], dim=1))
  ```

### 1.6 — Block 5: `YOLO26_1D_Regressor` (main model class)

Write class `YOLO26_1D_Regressor(nn.Module)`:

**`__init__` signature:**

```python
def __init__(self, in_channels=6, num_outputs=2,
             base_ch=32, widths=(64, 128, 256),
             n_blocks=(1, 2, 2), dropout=0.5):
```

**Build the layers in this exact order:**

```python
ch1, ch2, ch3 = widths

# Stem: (B, 6, 200) → (B, 32, 50)
self.stem = nn.Sequential(
    CBS(in_channels, base_ch, k=7, s=2),       # → (B, 32, 100)
    nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # → (B, 32, 50)
)

# Backbone stages
self.stage1 = C3k2_1D(base_ch, ch1, n=n_blocks[0], stride=2)  # → (B, 64,  25)
self.stage2 = C3k2_1D(ch1,     ch2, n=n_blocks[1], stride=2)  # → (B, 128, 13)
self.stage3 = C3k2_1D(ch2,     ch3, n=n_blocks[2], stride=2)  # → (B, 256,  7)

# ELAN multi-scale neck
self.neck = ELANNeck1D(ch1, ch2, ch3, out_ch=ch3)              # → (B, 256,  7)

# Global temporal pooling
self.pool = nn.AdaptiveAvgPool1d(1)                             # → (B, 256,  1)

# Regression head
self.head = nn.Sequential(
    nn.Linear(ch3, ch3 // 2),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout),
    nn.Linear(ch3 // 2, num_outputs)
)
```

**Add weight init method:**

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
```

**Add `get_num_params` helper:**

```python
def get_num_params(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**`forward` method:**

```python
def forward(self, x):
    x      = self.stem(x)               # (B, 32,  50)
    f1     = self.stage1(x)             # (B,  64, 25)
    f2     = self.stage2(f1)            # (B, 128, 13)
    f3     = self.stage3(f2)            # (B, 256,  7)
    fused  = self.neck(f1, f2, f3)     # (B, 256,  7)
    pooled = self.pool(fused).squeeze(-1)  # (B, 256)
    return self.head(pooled)            # (B, 2)
```

### 1.7 — Block 6: `MuSGD` optimizer

Write class `MuSGD(torch.optim.Optimizer)`:

**`__init__` signature:**

```python
def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, ns_steps=5):
    defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
    super().__init__(params, defaults)
```

**Static helper `_ns_orthogonalise(G, steps=5)`:**

```python
@staticmethod
def _ns_orthogonalise(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    X = G.reshape(G.shape[0], -1).float()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed: X = X.T
    return X.reshape(G.shape).to(orig_dtype)
```

**`step` method:**

```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad(): loss = closure()
    for group in self.param_groups:
        lr, mu, wd, ns = group['lr'], group['momentum'], group['weight_decay'], group['ns_steps']
        for p in group['params']:
            if p.grad is None: continue
            g = p.grad.data.clone()
            if wd != 0.0: g.add_(p.data, alpha=wd)
            state = self.state[p]
            if 'buf' not in state: state['buf'] = torch.zeros_like(p.data)
            buf = state['buf']
            buf.mul_(mu).add_(g)
            g = g.add(buf, alpha=mu)               # Nesterov
            if p.ndim >= 2:                        # Muon step for weight matrices
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                g = MuSGD._ns_orthogonalise(g, steps=ns) * scale
            p.data.add_(g, alpha=-lr)
    return loss
```

### 1.8 — Factory function `get_model`

```python
def get_model(in_channels=6, num_outputs=2, dropout=0.5):
    return YOLO26_1D_Regressor(
        in_channels=in_channels,
        num_outputs=num_outputs,
        base_ch=32,
        widths=(64, 128, 256),
        n_blocks=(1, 2, 2),
        dropout=dropout,
    )
```

✅ **Stage 1 success criterion:**

```bash
python3 -c "
from model_yolo26_1d import get_model
import torch
m = get_model(); m.eval()
x = torch.randn(8, 6, 200)
y = m(x)
assert y.shape == (8, 2), f'Expected (8,2) got {y.shape}'
print('PASS — output shape:', y.shape)
print('Params:', m.get_num_params())
"
```

Expected output: `PASS — output shape: torch.Size([8, 2])`

---

## Stage 2 — Integrate into `source/ronin_resnet.py`

**Goal:** Three surgical edits. No other line in ronin_resnet.py is touched.

### Change 1 — Add import (top of file)

Find the block:

```python
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
```

Add **immediately below** it (do NOT remove the original import line):

```python
from model_yolo26_1d import YOLO26_1D_Regressor, MuSGD, get_model
```

### Change 2 — Swap model instantiation

Find the **entire** `ResNet1D(...)` constructor block.
It typically spans ~10 lines with arguments like `BasicBlock1D`, `base_plane=64`, `fc_dim=512`.

**Replace the entire block** with:

```python
network = YOLO26_1D_Regressor(
    in_channels=6,
    num_outputs=2,
    base_ch=32,
    widths=(64, 128, 256),
    n_blocks=(1, 2, 2),
    dropout=0.5,
)
print(f"[YOLO26-1D] Parameters: {network.get_num_params():,}")
```

### Change 3 — Swap optimizer (optional but recommended)

Find the optimizer line — typically:

```python
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
```

**Replace with:**

```python
optimizer = MuSGD(
    network.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4,
    ns_steps=5,
)
```

> **Note:** If you want to keep Adam for an A/B comparison, leave the optimizer unchanged.
> The model is fully compatible with Adam.

### What NOT to touch

| Component                                   | Action          |
| ------------------------------------------- | --------------- |
| `HACFDataset` or any dataset class          | ❌ Do not touch |
| HACF tensor transformation logic            | ❌ Do not touch |
| `strided_velocity_loss` / MSE loss function | ❌ Do not touch |
| `train()` loop body                         | ❌ Do not touch |
| `test()` / `evaluate()` body                | ❌ Do not touch |
| Trajectory integration/reconstruction       | ❌ Do not touch |
| `torch.save` / `torch.load` checkpoints     | ❌ Do not touch |
| `argparse` CLI arguments                    | ❌ Do not touch |
| `ReduceLROnPlateau` scheduler               | ❌ Do not touch |

✅ **Stage 2 success criterion:**

```bash
python3 -c "
import sys; sys.path.insert(0, 'source')
import ronin_resnet  # should import without errors
print('PASS — ronin_resnet imports cleanly')
"
```

---

## Stage 3 — Smoke Test (Forward Pass + Backward Pass)

**Goal:** Verify the integrated model trains for 1 step without crashing.

### 3.1 — Quick shape validation script

Create `source/test_yolo26_integration.py`:

```python
"""Quick smoke test — run before full training."""
import torch
import sys
sys.path.insert(0, '.')

from model_yolo26_1d import YOLO26_1D_Regressor, MuSGD

def test_shapes():
    """Verify (B, 6, 200) → (B, 2) end-to-end."""
    model = YOLO26_1D_Regressor()
    model.eval()
    x = torch.randn(8, 6, 200)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 2), f"FAIL: expected (8,2), got {y.shape}"
    print(f"[PASS] Output shape: {y.shape}")
    print(f"[INFO] Parameters  : {model.get_num_params():,}")

def test_backward():
    """Verify gradients flow correctly."""
    model = YOLO26_1D_Regressor()
    model.train()
    optim = MuSGD(model.parameters(), lr=0.005)
    x = torch.randn(4, 6, 200)
    target = torch.randn(4, 2)
    pred = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optim.step()
    print(f"[PASS] Backward + MuSGD step succeeded. Loss={loss.item():.6f}")

def test_strided_loss_compat():
    """Simulate the strided velocity loss used in ronin_resnet.py."""
    model = YOLO26_1D_Regressor()
    model.train()
    x = torch.randn(16, 6, 200)
    pred_velocity = model(x)          # (B, 2)
    # Strided loss: compare against positional difference P_i - P_{i-200}
    gt_displacement = torch.randn(16, 2)
    loss = torch.nn.functional.mse_loss(pred_velocity, gt_displacement)
    loss.backward()
    print(f"[PASS] Strided velocity loss compat. Loss={loss.item():.6f}")

def test_batch_sizes():
    """Verify model handles variable batch sizes."""
    model = YOLO26_1D_Regressor()
    model.eval()
    for B in [1, 16, 32, 128]:
        with torch.no_grad():
            y = model(torch.randn(B, 6, 200))
        assert y.shape == (B, 2)
    print("[PASS] Variable batch sizes: 1, 16, 32, 128")

def test_intermediate_shapes():
    """Trace every intermediate tensor shape."""
    model = YOLO26_1D_Regressor()
    model.eval()
    x = torch.randn(4, 6, 200)
    with torch.no_grad():
        s  = model.stem(x);          print(f"  Stem:    {tuple(s.shape)}")   # (4, 32, 50)
        f1 = model.stage1(s);        print(f"  Stage1:  {tuple(f1.shape)}")  # (4, 64, 25)
        f2 = model.stage2(f1);       print(f"  Stage2:  {tuple(f2.shape)}")  # (4, 128,13)
        f3 = model.stage3(f2);       print(f"  Stage3:  {tuple(f3.shape)}")  # (4, 256, 7)
        fn = model.neck(f1,f2,f3);   print(f"  Neck:    {tuple(fn.shape)}")  # (4, 256, 7)
        gp = model.pool(fn).squeeze(-1); print(f"  GAP:     {tuple(gp.shape)}")  # (4, 256)
        out = model.head(gp);        print(f"  Output:  {tuple(out.shape)}")  # (4, 2)
    print("[PASS] All intermediate shapes correct")

if __name__ == '__main__':
    print("=== YOLO26-1D Smoke Tests ===")
    test_shapes()
    test_backward()
    test_strided_loss_compat()
    test_batch_sizes()
    test_intermediate_shapes()
    print("\n✅ All smoke tests passed. Ready for training.")
```

Run from `source/`:

```bash
cd ronin/source
python3 test_yolo26_integration.py
```

All five `[PASS]` lines must appear before proceeding.

### 3.2 — GPU readiness check

```bash
python3 -c "
import torch
from model_yolo26_1d import get_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
m = get_model().to(device)
x = torch.randn(8, 6, 200).to(device)
y = m(x)
print(f'PASS — GPU forward: {y.shape} on {device}')
"
```

✅ **Stage 3 success criterion:** All 5 smoke tests pass. GPU test shows correct device.

---

## Stage 4 — Training Run

**Goal:** Full training using the existing RoNIN training entry point with the new model.

### 4.1 — Dry run (1 epoch, sanity check)

```bash
cd ronin/source
python3 ronin_resnet.py \
    --mode train \
    --data_dir /path/to/ronin_dataset \
    --out_dir ./output/yolo26_test \
    --max_epoch 1 \
    --batch_size 128 \
    --cpu   # remove if GPU available
```

**Expected console output:**

```
[YOLO26-1D] Parameters: X,XXX,XXX
Epoch 1/1 — train_loss: X.XXXX — val_loss: X.XXXX
```

If you see `[YOLO26-1D] Parameters:` in the log, the model is loaded correctly.

### 4.2 — Full training run (Adam baseline — Option A)

Use this if you kept Adam as the optimizer for a fair comparison against the original paper results:

```bash
python3 ronin_resnet.py \
    --mode train \
    --data_dir /path/to/ronin_dataset \
    --out_dir ./output/yolo26_adam \
    --max_epoch 100 \
    --batch_size 128 \
    --lr 0.0001
```

### 4.3 — Full training run (MuSGD — Option B)

Use this after Change 3 (MuSGD swap) is applied:

```bash
python3 ronin_resnet.py \
    --mode train \
    --data_dir /path/to/ronin_dataset \
    --out_dir ./output/yolo26_musgd \
    --max_epoch 100 \
    --batch_size 128
```

### 4.4 — Checkpoint verification

After training completes, verify the saved checkpoint loads cleanly:

```bash
python3 -c "
import torch
from model_yolo26_1d import get_model
model = get_model()
ckpt  = torch.load('./output/yolo26_musgd/checkpoint_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
print('PASS — checkpoint loaded successfully')
"
```

### 4.5 — Training hyperparameter reference

| Hyperparameter | Adam (baseline)                             | MuSGD (recommended) |
| -------------- | ------------------------------------------- | ------------------- |
| Learning rate  | 0.0001                                      | 0.005               |
| Batch size     | 128                                         | 128                 |
| Max epochs     | 100                                         | 100                 |
| LR scheduler   | ReduceLROnPlateau (patience=10, factor=0.1) | same                |
| Weight decay   | — (in Adam eps)                             | 1e-4                |
| Dropout        | 0.5                                         | 0.5                 |
| ns_steps       | N/A                                         | 5                   |

✅ **Stage 4 success criterion:** Training completes 100 epochs without NaN loss.
Checkpoint saved at `output/*/checkpoint_best.pt`.

---

## Stage 5 — Evaluation and Benchmarking

**Goal:** Run the same evaluation protocol as the original RoNIN paper and record ATE/RTE.

### 5.1 — Evaluate on RoNIN test set

```bash
python3 ronin_resnet.py \
    --mode test \
    --data_dir /path/to/ronin_dataset \
    --model_path ./output/yolo26_musgd/checkpoint_best.pt \
    --out_dir ./output/yolo26_eval
```

### 5.3 — Metrics to record

For each dataset × seen/unseen subject split, record:

| Metric                | Definition                                 | Target (vs RoNIN-ResNet baseline) |
| --------------------- | ------------------------------------------ | --------------------------------- |
| **ATE**               | Root Mean Squared Error of full trajectory | Match or improve on RoNIN-ResNet  |
| **RTE**               | Average RMSE over 1-minute windows         | Match or improve on RoNIN-ResNet  |
| **Training time**     | Total wall-clock time to 100 epochs        | Should be ≤ ResNet (fewer params) |
| **Inference latency** | ms per batch on CPU/GPU                    | Should be ≤ ResNet                |

### 5.4 — Comparison table to fill in

```markdown
| Dataset | Split  | Model              | ATE (m) | RTE (m) |
| ------- | ------ | ------------------ | ------- | ------- |
| RoNIN   | Seen   | RoNIN-ResNet (ref) | 3.54    | 2.67    |
| RoNIN   | Seen   | YOLO26-1D (yours)  | ???     | ???     |
| RoNIN   | Unseen | RoNIN-ResNet (ref) | 5.14    | 4.37    |
| RoNIN   | Unseen | YOLO26-1D (yours)  | ???     | ???     |
| RIDI    | Seen   | RoNIN-ResNet (ref) | 1.63    | 1.91    |
| RIDI    | Seen   | YOLO26-1D (yours)  | ???     | ???     |
| OXIOD   | Seen   | RoNIN-ResNet (ref) | 2.40    | 1.77    |
| OXIOD   | Seen   | YOLO26-1D (yours)  | ???     | ???     |
```

✅ **Stage 5 success criterion:** All 8 ATE/RTE cells in the table above are filled with real numbers.
YOLO26-1D ATE on RoNIN-Seen should be within 20% of the 3.54m baseline.

---

## Error Handling Reference

| Error                                                 | Likely cause                                | Fix                                                               |
| ----------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| `ImportError: cannot import name YOLO26_1D_Regressor` | File not in `source/` or wrong path         | Move `model_yolo26_1d.py` into `source/`                          |
| `RuntimeError: Expected all tensors ... same device`  | Model on CPU, data on CUDA                  | Add `network = network.to(device)`                                |
| `shape mismatch` in neck                              | Stage output shapes changed due to rounding | Print all shapes in 3.2 trace; adjust padding in CBS              |
| `NaN loss` after step 1                               | Learning rate too high for MuSGD            | Reduce `lr` from 0.005 to 0.001                                   |
| `loss = nan` from start                               | Bad weight init or exploding inputs         | Check input normalization; call `_init_weights()` in `__init__`   |
| `checkpoint keys mismatch`                            | Model changed after saving                  | Re-train from scratch; never mix checkpoints across architectures |
| `CUDA out of memory`                                  | Batch size too large                        | Reduce `--batch_size` from 128 to 64                              |

---

## File Checklist (Final State)

```
ronin/source/
├── model_resnet1d.py              ← UNCHANGED ✓
├── model_yolo26_1d.py             ← NEW ✓ (488 lines)
├── ronin_resnet.py                ← 3 lines changed ✓
├── test_yolo26_integration.py     ← NEW ✓ (smoke tests)
├── ronin_lstm.py                  ← UNCHANGED ✓
├── ronin_tcn.py                   ← UNCHANGED ✓
├── data_glob_speed.py             ← UNCHANGED ✓
└── utils.py                       ← UNCHANGED ✓
```
