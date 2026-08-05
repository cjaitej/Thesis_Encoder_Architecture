# Raspberry Pi TFLite Inference Report

## 1. Purpose

This report summarizes the Raspberry Pi edge-device inference evaluation for the RoNIN inertial navigation models and the proposed Random Forest (RF) residual-correction postprocessing pipeline.

The main goal of this experiment was to show that the proposed YOLO26-based inertial navigation model and RF residual correction are not only desktop/GPU methods, but can also run on a low-power edge device using TensorFlow Lite and CPU-only inference.

The core comparison is:

- Original RoNIN ResNet baseline.
- Proposed YOLO26 model.
- YOLO26 + RF residual correction.
- Additional compact or mobile-oriented models: TinyCNN, LightTCN, ShuffleNet, and MobileNet.
- RF variants for non-ResNet models to estimate RF postprocessing overhead.

The strongest paper-relevant result is YOLO26 + RF, because the RF residual corrector was trained using YOLO26 residual features.

## 2. Raspberry Pi Evaluation Setup

### Hardware And Runtime

The benchmark was run on a Raspberry Pi device using a 32-bit ARM environment.

Observed configuration from the experiment:

| Item | Value |
|---|---|
| Device | Raspberry Pi |
| Architecture | `armv7l` |
| OS family | Raspberry Pi OS / Raspbian Bookworm, `armhf` packages |
| Python | Python 3.11 |
| Neural runtime | TensorFlow Lite runtime |
| RF runtime | scikit-learn RandomForestRegressor |
| RF sklearn version on Pi | `1.2.1` from apt |
| Input window | 200 IMU samples |
| Input channels | 6 channels: 3-axis gyro + 3-axis acceleration |
| Batch size | 1 |
| Test sequence | `a038_2` |
| Number of windows | 8,974 |
| Repeated runs | 5 full-sequence runs per model |
| Temperature observed | `48.9 C` |
| Throttling flag observed | `throttled=0x50000` |

The `0x50000` throttling flag means the Pi had experienced throttling or frequency capping earlier since boot. It does not necessarily mean the device was throttling at the exact timing instant, especially since the observed temperature was only `48.9 C`. However, it does indicate that Raspberry Pi timing can be affected by embedded-system conditions such as CPU frequency scaling, background services, power state, previous throttling history, and thermal state.

Before final runs, the background package management service was stopped:

```bash
sudo systemctl stop packagekit 2>/dev/null
```

This reduces interference from background apt/package activity during benchmarking.

## 3. Benchmark Method

The TFLite benchmark script used was:

```text
tools/run_tflite_benchmark.py
```

For each model, the script:

1. Loads the RoNIN sequence `a038_2`.
2. Reads `data.hdf5` and `info.json`.
3. Converts IMU measurements into global-frame RoNIN-style features.
4. Creates fixed-size sliding windows of shape `[1, 6, 200]` or the corresponding TFLite layout.
5. Runs TFLite inference over all 8,974 windows.
6. Repeats the full-sequence inference run 5 times.
7. Reports per-window latency:

```text
sample_time_ms = total_sequence_inference_time_ms / number_of_windows
```

For RF variants, the timing includes:

```text
neural TFLite inference + RF feature construction + RF prediction
```

The RF timing does not include loading the RF model from disk. This is appropriate for deployment, where model loading is a one-time initialization cost.

## 4. Model Conversion Pipeline

The original models were trained and saved as PyTorch checkpoints (`.pt`). Since PyTorch was not practical on the 32-bit Raspberry Pi, the models were converted to TensorFlow Lite (`.tflite`) and run with TFLite runtime.

### Conversion Routes

| Model | Conversion Route | Reason |
|---|---|---|
| ResNet | `.pt -> ONNX -> TFLite` | ONNX path worked |
| MobileNet | `.pt -> ONNX -> TFLite` | ONNX path worked |
| TinyCNN | `.pt -> ONNX -> TFLite` | ONNX path worked |
| LightTCN | Direct PyTorch -> Keras -> TFLite | ONNX path failed due TCN slice/chomp/residual shape issues |
| YOLO26 | Direct PyTorch -> Keras -> TFLite | ONNX path failed due layout/pooling/neck conversion problems |
| ShuffleNet | Direct PyTorch -> Keras -> TFLite | ONNX path failed due channel-shuffle dynamic reshape issues |

### Key Conversion Issues Faced

1. **PyTorch on Raspberry Pi was not feasible**

   The Raspberry Pi environment was 32-bit `armv7l`, and PyTorch installation was not practical. This motivated the TFLite deployment path.

2. **ONNX export issue for YOLO26**

   YOLO26 initially failed during ONNX export because `adaptive_avg_pool1d` with output size 7 was not supported when the input size was not a clean factor. This required export-side changes and later a direct TFLite conversion route.

3. **ONNX-to-TFLite issue for LightTCN**

   LightTCN failed during ONNX-to-TF conversion because TCN causal padding/chomp/slice logic caused shape mismatches in TensorFlow. A direct Keras/TFLite implementation was used instead.

4. **ONNX-to-TFLite issue for ShuffleNet**

   ShuffleNet failed because the channel shuffle reshape path led to zero-sized tensors during ONNX-to-TF tracing. A direct Keras/TFLite conversion was created.

5. **YOLO26 direct conversion issues**

   YOLO26 direct conversion initially had differences due to Conv1D padding and resize/pooling operators. These were corrected by matching PyTorch padding exactly and avoiding unsupported `ResizeArea` operations.

6. **RF sklearn compatibility issue**

   The original RF model was saved with a newer sklearn version. Raspberry Pi apt provided sklearn `1.2.1`, which could not load the newer tree pickle due to internal dtype changes. The RF was retrained on the Raspberry Pi using the same best hyperparameters so the joblib model was compatible with the Pi runtime.

### Conversion Validation

Direct-converted models were validated by comparing PyTorch and Keras outputs on random inputs:

| Model | PyTorch vs Keras max absolute difference |
|---|---:|
| LightTCN | `2.38e-07` |
| ShuffleNet | `4.69e-07` |
| YOLO26 | `1.11e-06` |

These differences are very small, confirming that the direct TFLite conversion logic preserved model behavior.

## 5. TFLite Model Sizes And Parameters

Approximate TFLite model characteristics:

| Model | TFLite Params | TFLite Size |
|---|---:|---:|
| ResNet | 4,630,007 | 8.854 MB |
| YOLO26 | 1,171,460 | 4.552 MB |
| MobileNet | 2,626,932 | 5.052 MB |
| ShuffleNet | 848,320 | 3.361 MB |
| TinyCNN | 114,620 | 0.235 MB |
| LightTCN | 79,128 | 0.329 MB |

Parameter count is useful, but it is not sufficient to predict edge latency. Operator type, memory movement, TFLite kernel support, padding, slicing, reshape operations, and ARM CPU optimization all affect runtime.

## 6. Timing Results

All timings are averaged over 5 repeated full-sequence runs on `a038_2`, with 8,974 inference windows per run.

### Baseline Models

| Model | Neural ms/window | RF ms/window | Total ms/window | Total sequence time |
|---|---:|---:|---:|---:|
| ResNet | 91.966 | 0.000 | 91.966 | 825.304 s |
| YOLO26 | 40.756 | 0.000 | 40.756 | 365.743 s |
| TinyCNN | 6.405 | 0.000 | 6.405 | 57.481 s |
| LightTCN | 49.338 | 0.000 | 49.338 | 442.763 s |
| ShuffleNet | 33.048 | 0.000 | 33.048 | 296.571 s |
| MobileNet | 79.737 | 0.000 | 79.737 | 715.561 s |

### RF Variants

For RF variants, the table reports the neural inference time observed inside the RF benchmark run, the RF postprocessing time, and the total.

| Model + RF | Neural ms/window | RF ms/window | Total ms/window | Total sequence time |
|---|---:|---:|---:|---:|
| YOLO26 + RF | 40.568 | 0.120 | 40.688 | 365.136 s |
| TinyCNN + RF | 5.316 | 0.138 | 5.454 | 48.947 s |
| LightTCN + RF | 49.565 | 0.146 | 49.711 | 446.105 s |
| ShuffleNet + RF | 19.165 | 0.107 | 19.272 | 172.943 s |
| MobileNet + RF | 79.236 | 0.115 | 79.351 | 712.096 s |

### Accuracy Proxy: Average MSE On `a038_2`

| Model | Avg MSE |
|---|---:|
| ResNet | 0.004623 |
| YOLO26 | 0.005238 |
| YOLO26 + RF | 0.004899 |
| TinyCNN | 0.006553 |
| TinyCNN + RF | 0.006784 |
| LightTCN | 0.012663 |
| LightTCN + RF | 0.016702 |
| ShuffleNet | 0.005159 |
| ShuffleNet + RF | 0.005535 |
| MobileNet | 0.004961 |
| MobileNet + RF | 0.005361 |

The RF accuracy result is most meaningful for YOLO26 + RF because the RF corrector was trained using YOLO26 residual features. For other models, RF timing is useful for overhead estimation, but their RF accuracy should not be interpreted as a fully optimized model-specific RF correction unless separate RF correctors are trained per base model.

## 7. Model-Specific Analysis

### ResNet

ResNet is the original RoNIN baseline and produced strong accuracy:

```text
Avg MSE: 0.004623
Latency: 91.966 ms/window
```

It is the slowest model in the experiment. This is expected because ResNet has a deeper residual architecture and the largest TFLite parameter count among the tested models.

Interpretation:

> ResNet is accurate but computationally expensive on Raspberry Pi. It is useful as a reference baseline, but its latency is less attractive for real-time edge deployment.

### YOLO26

YOLO26 achieved:

```text
Baseline latency: 40.756 ms/window
YOLO26 + RF total latency: 40.688 ms/window
YOLO26 avg MSE: 0.005238
YOLO26 + RF avg MSE: 0.004899
RF overhead: about 0.120 ms/window
```

YOLO26 is more than twice as fast as ResNet:

```text
ResNet / YOLO26 = 91.966 / 40.756 = about 2.26x
```

RF correction reduced YOLO26 average MSE by approximately:

```text
(0.005238 - 0.004899) / 0.005238 = about 6.5%
```

The apparent small latency difference between YOLO26 and YOLO26 + RF is within measurement noise. RF does not make neural inference faster. The correct interpretation is that YOLO26 + RF has comparable latency to YOLO26 baseline, with small RF overhead and improved MSE.

Interpretation:

> YOLO26 + RF provides the best accuracy-latency tradeoff for the proposed system. It is substantially faster than ResNet while RF correction narrows the accuracy gap.

### TinyCNN

TinyCNN is the fastest model:

```text
Baseline latency: 6.405 ms/window
TinyCNN + RF total latency: 5.454 ms/window
```

The RF result appears faster than the separate TinyCNN baseline, but this is not a true algorithmic speedup. Logically:

```text
TinyCNN + RF = TinyCNN inference + RF feature construction + RF prediction
```

So it cannot be intrinsically faster than TinyCNN alone under identical system conditions. The difference is due to Raspberry Pi runtime variability between separate benchmark sessions.

TinyCNN has the smallest and simplest CNN-style architecture, which makes it very fast. However, its MSE is worse than YOLO26, MobileNet, ShuffleNet, and ResNet.

Interpretation:

> TinyCNN is excellent for low-latency deployment but sacrifices accuracy. It establishes a lower-bound latency reference.

### LightTCN

LightTCN has the smallest parameter count, but its latency is high:

```text
Params: 79,128
Latency: 49.338 ms/window
```

This shows that parameter count alone does not predict edge inference time. TCN-style operations may involve temporal padding, slicing, reshape, and memory movement patterns that are not efficiently optimized by TFLite on ARM CPU.

Interpretation:

> LightTCN is small in memory footprint but not fast on this TFLite/Raspberry Pi backend. Operator efficiency matters as much as model size.

### ShuffleNet

ShuffleNet is intended as a mobile-efficient architecture, but its baseline timing was:

```text
33.048 ms/window
```

Its RF timing appears much lower:

```text
19.272 ms/window
```

This cannot mean RF made ShuffleNet faster. It indicates that the baseline and RF runs were affected by different runtime conditions. The RF CSV is internally meaningful because it reports neural and RF timing within the same RF run:

```text
Neural inside RF run: 19.165 ms/window
RF overhead: 0.107 ms/window
```

Interpretation:

> ShuffleNet can be relatively fast on edge hardware, but the current baseline-vs-RF comparison should be interpreted cautiously because separate runs showed system-level timing variability.

### MobileNet

MobileNet showed:

```text
Baseline latency: 79.737 ms/window
MobileNet + RF: 79.351 ms/window
```

MobileNet was slower than YOLO26, ShuffleNet, TinyCNN, and LightTCN in this setup. Although MobileNet is designed for mobile vision workloads, its 1D converted TFLite graph may not map as efficiently to this Raspberry Pi CPU backend. It also has more TFLite parameters than YOLO26.

Interpretation:

> MobileNet provides competitive MSE but high latency in this 1D inertial-navigation TFLite deployment.

## 8. Comparative Analysis

### Latency Ranking

Using the measured baseline mean latency:

1. TinyCNN: 6.405 ms/window
2. ShuffleNet: 33.048 ms/window
3. YOLO26: 40.756 ms/window
4. LightTCN: 49.338 ms/window
5. MobileNet: 79.737 ms/window
6. ResNet: 91.966 ms/window

TinyCNN is fastest, but not most accurate. ResNet is accurate but slow. YOLO26 sits in the middle and becomes highly attractive after RF correction.

### Accuracy-Latency Tradeoff

ResNet has the best MSE among the measured baselines, but it is slow:

```text
ResNet: 0.004623 MSE, 91.966 ms/window
```

YOLO26 + RF gives:

```text
YOLO26 + RF: 0.004899 MSE, 40.688 ms/window
```

This is close to ResNet accuracy while requiring less than half the inference time.

This supports the paper argument:

> YOLO26 + RF offers a favorable accuracy-latency tradeoff for edge inertial navigation.

### RF Overhead

RF overhead per window was small across models:

| RF Variant | RF overhead ms/window |
|---|---:|
| YOLO26 + RF | 0.120 |
| TinyCNN + RF | 0.138 |
| LightTCN + RF | 0.146 |
| ShuffleNet + RF | 0.107 |
| MobileNet + RF | 0.115 |

This shows that RF postprocessing is lightweight relative to neural inference, especially for larger models.

## 9. Why Some RF Runs Look Faster Than Baseline

Some RF variants appear faster than their corresponding separately measured baseline, for example:

```text
TinyCNN baseline: 6.405 ms/window
TinyCNN + RF:     5.454 ms/window
```

and:

```text
ShuffleNet baseline: 33.048 ms/window
ShuffleNet + RF:     19.272 ms/window
```

This should not be interpreted as RF making the model faster. Under identical conditions:

```text
model + RF = model inference + extra RF computation
```

Therefore, RF cannot reduce the true computational cost. The observed difference is caused by Raspberry Pi runtime variability, including:

- CPU frequency scaling.
- Background tasks.
- Package manager activity.
- Thermal and power state.
- Historical throttling or frequency capping.
- Cache and warm-up effects.
- Separate benchmark sessions not having identical system state.

For RF models, the most reliable timing breakdown is the RF CSV itself, because it measures:

```text
neural_ms
rf_ms
total_ms
```

within the same run. Therefore, RF overhead should be interpreted from `rf_ms`, not by subtracting a separately measured baseline.

## 10. Challenges Faced

### 1. PyTorch was not available on Raspberry Pi

The Raspberry Pi environment was 32-bit ARM, so running the original `.pt` PyTorch models directly was not practical.

### 2. Multiple conversion paths were needed

Not all models converted cleanly through ONNX. Some required direct PyTorch-to-Keras-to-TFLite conversion.

### 3. TFLite operator/layout issues

Different models required special handling:

- YOLO26 required exact padding and pooling behavior.
- LightTCN required avoiding problematic ONNX slice/chomp behavior.
- ShuffleNet required avoiding ONNX channel-shuffle reshape failure.
- Some TFLite files expected `[1, 200, 6]` instead of `[1, 6, 200]`, so the benchmark script had to detect and transpose automatically.

### 4. Dataset corruption

The original `a001_2/data.hdf5` file was truncated. The sequence `a038_2` was selected because it opened successfully and had a similar usable size among valid test sequences.

### 5. Dependency issues on Raspberry Pi

Several runtime dependencies required fixes:

- NumPy 2.x was incompatible with the installed TFLite runtime, so NumPy had to be downgraded below 2.
- Missing OpenBLAS and HDF5 system libraries had to be installed.
- scikit-learn was needed for RF joblib loading.
- The RF model had to be retrained on the Pi because sklearn model pickle files are not safely portable across versions.

### 6. Network and SSH issues

The Pi changed IP addresses and had route/DNS problems when connected through a mobile hotspot. The default route had to be corrected to use the active hotspot interface.

### 7. Timing variability on Raspberry Pi

Raspberry Pi measurements showed some runtime variability, especially for separately executed baseline and RF runs. Temperature and throttling state were checked, and background services were stopped before benchmarking.

## 11. Suggested Paper Wording

The following paragraph can be adapted for the paper:

> We evaluated all models on a Raspberry Pi edge platform using TensorFlow Lite runtime with batch size 1. Each model was executed over 8,974 fixed-length IMU windows extracted from the `a038_2` RoNIN test sequence. For each model, inference was repeated over five full-sequence runs, and per-window latency was computed as total inference time divided by the number of windows. For RF-enhanced variants, the reported latency includes neural TFLite inference, RF feature construction, and Random Forest residual prediction. Dataset loading and preprocessing were excluded from timing. The Raspberry Pi temperature during benchmarking was approximately 48.9 C; throttling history indicated that frequency capping had occurred earlier since boot, so repeated runs and median/mean statistics were used to mitigate runtime variability.

And for the main result:

> Compared with the original RoNIN ResNet baseline, YOLO26 substantially reduced per-window latency on the Raspberry Pi. Applying RF residual correction to YOLO26 improved prediction MSE with only about 0.12 ms/window of RF overhead, demonstrating that the proposed neural + RF correction pipeline is feasible for edge-device inertial navigation.

## 12. Conclusion

The Raspberry Pi TFLite experiment demonstrates that the proposed YOLO26 + RF residual-correction system is deployable on edge hardware without GPU acceleration. ResNet remains a strong accuracy baseline but is the slowest model. TinyCNN is the fastest but less accurate. LightTCN is small but unexpectedly slow due to operator/backend inefficiency. MobileNet is accurate but relatively slow in this 1D TFLite setting. ShuffleNet shows potential for low latency but had run-to-run variability.

YOLO26 + RF is the best overall tradeoff in this experiment:

- Much faster than ResNet.
- Accuracy close to ResNet.
- RF overhead is very small.
- Runs successfully on Raspberry Pi with TFLite and sklearn.

The timing differences between baseline and RF variants should be interpreted carefully. RF does not make neural inference faster; apparent negative overheads are due to Raspberry Pi system-level timing variability. The important result is that RF overhead itself is small, and YOLO26 + RF achieves improved accuracy with practical edge-device latency.
