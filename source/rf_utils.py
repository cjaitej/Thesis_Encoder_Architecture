import numpy as np


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
        v = max(0.0, s2 / count - m * m)
        mean[i] = m
        std[i] = np.sqrt(v)

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

    feat_cols = [
        preds[:, 0],
        preds[:, 1],
        pred_speed,
        gyro[:, 0],
        gyro[:, 1],
        gyro[:, 2],
        acce[:, 0],
        acce[:, 1],
        acce[:, 2],
        gyro_norm,
        acce_norm,
        gyro_m,
        gyro_s,
        acce_m,
        acce_s,
        d_pred[:, 0],
        d_pred[:, 1],
        t_norm,
    ]
    X = np.stack(feat_cols, axis=1).astype(np.float32)

    feature_names = [
        "pred_vx",
        "pred_vy",
        "pred_speed",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "acce_x",
        "acce_y",
        "acce_z",
        "gyro_norm",
        "acce_norm",
        "gyro_norm_mean",
        "gyro_norm_std",
        "acce_norm_mean",
        "acce_norm_std",
        "d_pred_vx",
        "d_pred_vy",
        "time_progress",
    ]

    return X, feature_names


def apply_rf_correction(preds, residual_hat, alpha=1.0, residual_clip=None):
    corrected = preds + alpha * residual_hat
    if residual_clip is not None and residual_clip > 0:
        delta = corrected - preds
        delta = np.clip(delta, -residual_clip, residual_clip)
        corrected = preds + delta
    return corrected.astype(np.float32)
