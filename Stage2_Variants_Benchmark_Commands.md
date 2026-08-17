# Raspberry Pi TFLite Benchmark — YOLOv26-1D and Efficient YOLOv26-1D

Instructions for benchmarking **two backbones and their five stage-2 correctors each** —
12 targets total — on the Raspberry Pi, using `tools/run_tflite_benchmark.py`.

| Backbone | TFLite file | Params | Size |
|---|---|---|---|
| YOLOv26-1D | `models_tflite/yolo26.tflite` | 1,174,594 | 4.55 MiB |
| Efficient YOLOv26-1D | `models_tflite/yolo26_eff.tflite` | 598,530 | 2.41 MiB |

Both take `[1, 6, 200]` and return `[1, 2]`, so the data pipeline, windowing and CSV columns are
identical for every target. The benchmark measures **latency only** — see §6.

PyTorch is **not** required on the Pi. Only the two `A_rf` targets need scikit-learn.

---

## 1. Targets

### Backbone only

| `--which` | Model file |
|---|---|
| `yolo26` | `yolo26.tflite` |
| `yolo26_eff` | `yolo26_eff.tflite` |

### YOLOv26-1D + stage 2

| `--which` | Stage 2 | Runtime |
|---|---|---|
| `yolo26_A_rf` | Random Forest | tflite + sklearn |
| `yolo26_B_ridge` | Ridge / linear | tflite + numpy |
| `yolo26_C_ema` | Causal EMA | tflite + numpy |
| `yolo26_D_mlp` | Small MLP | tflite + numpy |
| `yolo26_E_tcn` | Causal TCN | tflite + numpy |

### Efficient YOLOv26-1D + stage 2

| `--which` | Stage 2 | Runtime |
|---|---|---|
| `yolo26_eff_A_rf` | Random Forest | tflite + sklearn |
| `yolo26_eff_B_ridge` | Ridge / linear | tflite + numpy |
| `yolo26_eff_C_ema` | Causal EMA | tflite + numpy |
| `yolo26_eff_D_mlp` | Small MLP | tflite + numpy |
| `yolo26_eff_E_tcn` | Causal TCN | tflite + numpy |

## 2. Setup

```bash
source ~/ronin_env/bin/activate
cd /media/shashwat/A431-E4641/ronin_yolov26
```

## 3. Files to Copy

**Models** → `models_tflite/`

- [ ] `yolo26.tflite` (4.55 MiB)
- [ ] `yolo26_eff.tflite` (2.41 MiB)

**Correctors** → two directories at the project root. Copy from the training machine:

| Copy from (training machine) | To (Pi, project root) | Used by |
|---|---|---|
| `output/stage2_models_yolo26_ckpt35/` | `stage2_models/` | `yolo26_*` targets |
| `output/stage2_models_yolo26_eff/` | `stage2_models_yolo26_eff/` | `yolo26_eff_*` targets |

Each directory holds the same six files (sizes measured from the actual artifacts):

| File | Size | Needed for |
|---|---|---|
| `B_ridge_corrector.npz` | 3.2 KB | `*_B_ridge` |
| `C_ema_corrector.npz` | 2.2 KB | `*_C_ema` |
| `D_mlp_corrector.npz` | 25.7 KB | `*_D_mlp` |
| `E_tcn_corrector.npz` | 53.4 KB | `*_E_tcn` |
| `A_rf_corrector.npz` | 2.3 KB | `*_A_rf` |
| `A_rf_corrector_sklearn.joblib` | **~185 MB** (yolo26) / **~181 MB** (yolo26_eff) | `*_A_rf` |

The four numpy variants total roughly 84 KB per backbone; the Random Forest joblib dominates.
`A_rf_corrector.npz` is only a descriptor — it points at `A_rf_corrector_sklearn.joblib` by
filename and loads it **from its own directory**, so the two files must stay together.

Correction gain and clip bound are stored inside each `.npz` (tuned per corrector on the
validation split), so `--rf_alpha` and `--rf_clip` do not apply to these targets. The fitted
values differ per variant and per backbone — e.g. `C_ema` uses alpha 0.75 with no clip, while
`A_rf` uses alpha 1.25 with clip 0.25 — which is why the corrector sets are not interchangeable.

> **The two corrector sets are not interchangeable.** A corrector is fitted on one backbone's
> residual errors. The filenames are identical in both directories, so the only thing keeping
> them apart is `--stage2_dir`. Pointing a `yolo26_eff_*` target at `stage2_models/` will run
> and produce numbers — they will just be wrong. Match the directory to the target prefix every
> time.

**Input data** — same sequence as the TinyCNN run:

- [ ] `data/seen_subjects_test_set/a000_11/`

## 4. Verify the Setup

Run one iteration of the cheapest and most expensive corrector on each backbone. This confirms
the model files load, the corrector files are found, and the numpy path executes on ARM:

```bash
python3 tools/run_tflite_benchmark.py --which yolo26_C_ema --runs 1 \
  --root_dir data/seen_subjects_test_set --sequence a000_11 \
  --stage2_dir stage2_models

python3 tools/run_tflite_benchmark.py --which yolo26_eff_E_tcn --runs 1 \
  --root_dir data/seen_subjects_test_set --sequence a000_11 \
  --stage2_dir stage2_models_yolo26_eff
```

Each should print a line ending `neural=… rf=… total=…` and write a CSV under
`bench_results_tflite/`. If either fails, fix it before continuing — the full sweep is ~30
minutes.

Optional regression check that the existing workflow still runs:

```bash
python3 tools/run_tflite_benchmark.py --which tinycnn --runs 1 \
  --root_dir data/seen_subjects_test_set --sequence a000_11
```

## 5. Run the Benchmark

Run the three blocks in order.

### Step 1 — bare backbones

Run these first. Stage-2 overhead is isolated by subtracting these from the corrected targets,
and this is the headline backbone comparison.

```bash
for M in yolo26 yolo26_eff; do
  python3 tools/run_tflite_benchmark.py --which $M --runs 50 \
    --root_dir data/seen_subjects_test_set --sequence a000_11
done
```

### Step 2 — YOLOv26-1D + stage 2

```bash
for M in yolo26_A_rf yolo26_B_ridge yolo26_C_ema yolo26_D_mlp yolo26_E_tcn; do
  python3 tools/run_tflite_benchmark.py --which $M --runs 50 \
    --root_dir data/seen_subjects_test_set --sequence a000_11 \
    --stage2_dir stage2_models
done
```

### Step 3 — Efficient YOLOv26-1D + stage 2

Note the different `--stage2_dir`.

```bash
for M in yolo26_eff_A_rf yolo26_eff_B_ridge yolo26_eff_C_ema yolo26_eff_D_mlp yolo26_eff_E_tcn; do
  python3 tools/run_tflite_benchmark.py --which $M --runs 50 \
    --root_dir data/seen_subjects_test_set --sequence a000_11 \
    --stage2_dir stage2_models_yolo26_eff
done
```

If the shell variable is lost when pasting, run each command individually with the model name
written out.

## 6. Output Files

Written to `bench_results_tflite/`:

- `<model>_benchmark_results.csv` — per-run results
- `<model>_benchmark_summary.txt` — per-model summary
- `benchmark_summary.txt` — overall summary

CSV columns — exactly ten, verified against both the script and the existing Pi result files:

    model, run, sample_time_ms, seq_time_ms, neural_ms, rf_ms, total_ms,
    mse_x, mse_y, returncode

`sample_time_ms` is the per-sample latency — this is the number quoted in the paper's Raspberry
Pi table (the existing `yolo26` run recorded 40.688 ms here). `seq_time_ms` is the whole-sequence
time. For stage-2 targets, `rf_ms` holds the corrector time: feature construction, corrector
inference, and the correction itself.

> **This script records timing only — no power or energy.** It contains no INA219 code and
> writes no `voltage_v` / `current_a` / `power_w` / `energy_j` columns; every existing Pi result
> CSV in `bench_results_tflite_pi/` confirms the ten columns above. If power and energy figures
> are needed, they must come from a separate INA219 capture run alongside this benchmark, and
> correlated by wall-clock time.

## 7. Checking the Results

```bash
wc -l bench_results_tflite/yolo26_eff_benchmark_results.csv
cat  bench_results_tflite/yolo26_eff_benchmark_summary.txt
cat  bench_results_tflite/benchmark_summary.txt
```

Three sanity checks before quoting anything:

1. **12 result files exist**, one per target.
2. **`neural_ms` for `yolo26_eff` is below `yolo26`.** If not, the wrong `.tflite` was picked up.
3. **`neural_ms` is roughly constant across a backbone's six targets.** Stage 2 changes `rf_ms`,
   not `neural_ms`. A moving `neural_ms` means the wrong model file is bound to some target.

Expected `rf_ms` per sample, measured on x86 and scaled by 5× for the Pi (that factor comes from
the Random Forest's known 0.268 ms Pi overhead against 0.0555 ms on x86). These costs are
backbone-independent — the corrector consumes the same 18-feature vector either way — so the
same figures apply to both backbones:

| Variant | x86 (ms/sample) | Pi estimate (ms/sample) |
|---|---|---|
| `B_ridge` | 0.0044 | ~0.02 |
| `D_mlp` | 0.0048 | ~0.02 |
| `C_ema` | 0.0078 | ~0.04 |
| `E_tcn` | 0.0203 | ~0.10 |
| `A_rf` | 0.0555 | ~0.28 |

Against the 50 ms update budget, `yolo26` measured 40.42 ms, leaving ~9.3 ms — comfortably more
than any corrector needs. `yolo26_eff` Pi latency is **not yet measured**; producing that number
is the main purpose of this run. It has 49% fewer parameters, but latency does not scale with
parameter count, and the reductions were concentrated in the attention block and neck rather
than the convolutional stages that usually dominate ARM runtime. Do not assume a speedup until
step 1 reports one.

## 8. Before Quoting Results

**Latency, power and energy are the deliverable here.** They are architecture-driven and valid
for all 12 targets.

**`mse_x` / `mse_y` in these CSVs are for sanity-checking only.** They come from a single
sequence (`a000_11`). Trajectory accuracy (ATE / RTE) comes from the full seen/unseen
evaluation, not from this script — do not put these columns in a results table, and do not
compare accuracy across the two backbones from them.

**`yolo26.tflite` was exported from epoch 35**, while the finalized model is epoch 43. Timing
and power are unaffected — identical architecture, identical compute — but any accuracy figure
from this target reflects the older weights.

**The `yolo26_*` correctors were fitted on a dataset that included the unseen test set.** Their
timing is valid; their accuracy should not be published until they are refitted. The
`yolo26_eff_*` correctors were fitted on training sequences only and do not carry this issue.

`yolo26_eff.tflite` was exported from `checkpoint_59.pt` of the `yolo26_eff_adam_v1` run;
PyTorch-to-TFLite weight transfer was verified at `max_abs_diff = 1.70e-06`.
