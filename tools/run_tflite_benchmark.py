#!/usr/bin/env python3
"""
Run RoNIN TFLite inference benchmarks on Raspberry Pi without importing torch.

Expected Pi layout:

    ~/sohel/
      models_tflite/
        resnet.tflite
        yolo26.tflite
        mobilenet.tflite
        shufflenet.tflite
        tinycnn.tflite
        lighttcn.tflite
      rf_yolo26/rf_corrector.joblib
      seen_subjects_test_set/a001_2/
      tools/run_tflite_benchmark.py
"""
import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import quaternion

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


MODELS = {
    "resnet": {"path": "models_tflite/resnet.tflite", "rf": False},
    "yolo26": {"path": "models_tflite/yolo26.tflite", "rf": False},
    "mobilenet": {"path": "models_tflite/mobilenet.tflite", "rf": False},
    "shufflenet": {"path": "models_tflite/shufflenet.tflite", "rf": False},
    "tinycnn": {"path": "models_tflite/tinycnn.tflite", "rf": False},
    "lighttcn": {"path": "models_tflite/lighttcn.tflite", "rf": False},
    "yolo26_rf": {"path": "models_tflite/yolo26.tflite", "rf": True},
    "mobilenet_rf": {"path": "models_tflite/mobilenet.tflite", "rf": True},
    "shufflenet_rf": {"path": "models_tflite/shufflenet.tflite", "rf": True},
    "tinycnn_rf": {"path": "models_tflite/tinycnn.tflite", "rf": True},
    "lighttcn_rf": {"path": "models_tflite/lighttcn.tflite", "rf": True},
}


def select_game_rv(data_path):
    with h5py.File(data_path / "data.hdf5") as f:
        return np.copy(f["synced/game_rv"])


def load_ronin_sequence(data_path, interval=200):
    with (data_path / "info.json").open() as f:
        info = json.load(f)

    ori = select_game_rv(data_path)

    with h5py.File(data_path / "data.hdf5") as f:
        gyro_uncalib = np.copy(f["synced/gyro_uncalib"])
        acce_uncalib = np.copy(f["synced/acce"])
        gyro = gyro_uncalib - np.array(info["imu_init_gyro_bias"])
        acce = np.array(info["imu_acce_scale"]) * (acce_uncalib - np.array(info["imu_acce_bias"]))
        ts = np.copy(f["synced/time"])
        tango_pos = np.copy(f["pose/tango_pos"])
        init_tango_ori = quaternion.quaternion(*f["pose/tango_ori"][0])

    ori_q = quaternion.from_float_array(ori)
    rot_imu_to_tango = quaternion.quaternion(*info["start_calibration"])
    init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
    ori_q = init_rotor * ori_q

    dt = (ts[interval:] - ts[:-interval])[:, None]
    glob_v = (tango_pos[interval:] - tango_pos[:-interval]) / dt

    gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
    glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
    glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

    start_frame = info.get("start_frame", 0)
    features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:].astype(np.float32)
    targets = glob_v[start_frame:, :2].astype(np.float32)
    ts = ts[start_frame:].astype(np.float64)
    gt_pos = tango_pos[start_frame:].astype(np.float32)
    return features, targets, ts, gt_pos


def make_windows(features, targets, window_size, step_size):
    indices = np.arange(0, targets.shape[0], step_size, dtype=np.int64)
    windows = np.stack([features[i:i + window_size].T for i in indices], axis=0).astype(np.float32)
    return windows, targets[indices], indices


def run_tflite(model_path, windows, batch_size):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    def refresh_details():
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        return input_details, output_details

    input_details, output_details = refresh_details()
    native_input_shape = tuple(int(x) for x in input_details["shape"])

    def prepare_batch(batch):
        if tuple(batch.shape) == native_input_shape:
            return batch
        if batch.ndim == 3 and len(native_input_shape) == 3:
            if tuple(batch.transpose(0, 2, 1).shape) == native_input_shape:
                return batch.transpose(0, 2, 1)
        return batch

    def invoke_allocated(batch):
        batch = prepare_batch(batch)
        input_details, output_details = refresh_details()
        interpreter.set_tensor(input_details["index"], batch.astype(input_details["dtype"]))
        interpreter.invoke()
        return interpreter.get_tensor(output_details["index"])

    def invoke_batch(batch):
        batch = prepare_batch(batch)
        if tuple(batch.shape) == native_input_shape:
            return invoke_allocated(batch)

        input_details, _ = refresh_details()
        interpreter.resize_tensor_input(input_details["index"], batch.shape, strict=False)
        interpreter.allocate_tensors()
        return invoke_allocated(batch)

    preds = []
    start = time.perf_counter()
    for start_idx in range(0, windows.shape[0], batch_size):
        batch = windows[start_idx:start_idx + batch_size]
        try:
            preds.append(invoke_batch(batch))
        except Exception:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            for sample in batch:
                preds.append(invoke_allocated(sample[None, ...])[0])

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.concatenate(preds, axis=0).astype(np.float32), elapsed_ms


def causal_mean_std_1d(values, window):
    values = values.astype(np.float64)
    n = values.shape[0]
    mean = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    csum = np.cumsum(values)
    csum2 = np.cumsum(values * values)
    for i in range(n):
        left = max(0, i - window + 1)
        count = i - left + 1
        s = csum[i] - (csum[left - 1] if left > 0 else 0.0)
        s2 = csum2[i] - (csum2[left - 1] if left > 0 else 0.0)
        m = s / count
        mean[i] = m
        std[i] = np.sqrt(max(0.0, s2 / count - m * m))
    return mean.astype(np.float32), std.astype(np.float32)


def build_sequence_features(preds, feat_sparse, ts, hist_window):
    gyro = feat_sparse[:, :3]
    acce = feat_sparse[:, 3:6]
    pred_speed = np.linalg.norm(preds, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acce_norm = np.linalg.norm(acce, axis=1)
    gyro_m, gyro_s = causal_mean_std_1d(gyro_norm, hist_window)
    acce_m, acce_s = causal_mean_std_1d(acce_norm, hist_window)
    d_pred = np.zeros_like(preds)
    d_pred[1:] = preds[1:] - preds[:-1]
    t0 = ts[0]
    t1 = ts[-1] if ts[-1] > t0 else t0 + 1.0
    t_norm = ((ts - t0) / (t1 - t0)).astype(np.float32)
    return np.stack(
        [
            preds[:, 0], preds[:, 1], pred_speed,
            gyro[:, 0], gyro[:, 1], gyro[:, 2],
            acce[:, 0], acce[:, 1], acce[:, 2],
            gyro_norm, acce_norm, gyro_m, gyro_s, acce_m, acce_s,
            d_pred[:, 0], d_pred[:, 1], t_norm,
        ],
        axis=1,
    ).astype(np.float32)


def apply_rf(preds, rf_model_path, feat_sparse, ts_sparse, hist_window, alpha, clip):
    import joblib
    rf_model = joblib.load(rf_model_path)
    start = time.perf_counter()
    rf_x = build_sequence_features(preds, feat_sparse, ts_sparse, hist_window)
    residual_hat = rf_model.predict(rf_x).astype(np.float32)
    rf_elapsed_ms = (time.perf_counter() - start) * 1000.0
    corrected = preds + alpha * residual_hat
    if clip is not None and clip > 0:
        corrected = preds + np.clip(corrected - preds, -clip, clip)
    return corrected.astype(np.float32), rf_elapsed_ms


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "run", "sample_time_ms", "seq_time_ms",
                "neural_ms", "rf_ms", "total_ms", "mse_x", "mse_y", "returncode",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows):
    ok = [row for row in rows if row["returncode"] == 0]
    if not ok:
        return None
    sample = [row["sample_time_ms"] for row in ok]
    return {
        "count": len(sample),
        "mean_sample_time_ms": float(np.mean(sample)),
        "median_sample_time_ms": float(np.median(sample)),
        "std_sample_time_ms": float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0,
    }


def write_model_summary(path, name, summary):
    with path.open("w") as f:
        f.write(f"== {name} ==\n")
        f.write(str(summary) + "\n")


def benchmark_model(name, spec, args, windows, targets, features, ts, indices):
    model_path = args.base_dir / spec["path"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    rows = []
    for run_idx in range(1, args.runs + 1):
        preds, neural_ms = run_tflite(model_path, windows, args.batch_size)
        rf_ms = 0.0
        total_ms = neural_ms
        if spec["rf"]:
            preds, rf_ms = apply_rf(
                preds,
                args.base_dir / args.rf_model_path,
                features[indices],
                ts[indices],
                args.rf_hist_window,
                args.rf_alpha,
                args.rf_clip,
            )
            total_ms = neural_ms + rf_ms

        mse = np.mean((targets - preds) ** 2, axis=0)
        sample_time_ms = total_ms / max(1, windows.shape[0])
        row = {
            "model": name,
            "run": run_idx,
            "sample_time_ms": sample_time_ms,
            "seq_time_ms": total_ms,
            "neural_ms": neural_ms,
            "rf_ms": rf_ms,
            "total_ms": total_ms,
            "mse_x": float(mse[0]),
            "mse_y": float(mse[1]),
            "returncode": 0,
        }
        rows.append(row)
        print(
            f"Run {run_idx}/{args.runs}: "
            f"time_per_sample={sample_time_ms:.4f}ms, "
            f"neural={neural_ms:.2f}ms, rf={rf_ms:.2f}ms, total={total_ms:.2f}ms"
        )
        if args.delay and run_idx < args.runs:
            time.sleep(args.delay)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument("--which", default="all", help="all or one model name")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--sequence", default="a001_2")
    parser.add_argument("--root_dir", default="seen_subjects_test_set")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rf_model_path", default="rf_yolo26/rf_corrector.joblib")
    parser.add_argument("--rf_alpha", type=float, default=1.0)
    parser.add_argument("--rf_clip", type=float, default=0.75)
    parser.add_argument("--rf_hist_window", type=int, default=50)
    parser.add_argument("--out_dir", default="bench_results_tflite")
    args = parser.parse_args()

    args.base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(MODELS) if args.which == "all" else [args.which]
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")

    seq_path = args.base_dir / args.root_dir / args.sequence
    features, targets_all, ts, _ = load_ronin_sequence(seq_path, interval=args.window_size)
    windows, targets, indices = make_windows(features, targets_all, args.window_size, args.step_size)
    print(f"Loaded {args.sequence}: windows={windows.shape}, targets={targets.shape}")

    summary_lines = []
    for name in selected:
        print(f"\n=== Running TFLite benchmark for: {name} ===")
        rows = benchmark_model(name, MODELS[name], args, windows, targets, features, ts, indices)
        csv_path = out_dir / f"{name}_benchmark_results.csv"
        write_csv(csv_path, rows)
        summary = summarize(rows)
        summary_lines.append((name, summary))
        model_summary_path = out_dir / f"{name}_benchmark_summary.txt"
        write_model_summary(model_summary_path, name, summary)
        print(f"Wrote {csv_path}")
        print(f"Wrote {model_summary_path}")
        print("Summary:", summary)

    summary_path = out_dir / "benchmark_summary.txt"
    with summary_path.open("w") as f:
        for name, summary in summary_lines:
            f.write(f"== {name} ==\n")
            f.write(str(summary) + "\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
