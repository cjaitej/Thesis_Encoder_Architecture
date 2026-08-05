"""
Computes per-channel mean and std from RIDI training set.
Channels: [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z]
Output: ../output/stats/ridi_channel_stats.npy
"""

import os
import numpy as np
import pandas as pd

RIDI_ROOT = "../data/ridi_dataset"
TRAIN_LIST = "../lists/list_ridi_train.txt"
OUT_PATH = "../output/stats/ridi_channel_stats.npy"


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    seqs = read_list(TRAIN_LIST)
    ch_sum = np.zeros(6, dtype=np.float64)
    ch_sq = np.zeros(6, dtype=np.float64)
    total_n = 0
    used = 0
    missing = []

    for rel in seqs:
        seq_path = os.path.join(RIDI_ROOT, rel)
        csv_path = os.path.join(seq_path, "processed", "data.csv")
        pkl_path = os.path.join(seq_path, "processed", "data.pkl")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.exists(pkl_path):
            df = pd.read_pickle(pkl_path)
        else:
            missing.append(rel)
            continue

        acce = df[["acce_x", "acce_y", "acce_z"]].to_numpy(dtype=np.float64)
        gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float64)
        imu = np.concatenate([acce, gyro], axis=1)

        ch_sum += imu.sum(axis=0)
        ch_sq += (imu ** 2).sum(axis=0)
        total_n += imu.shape[0]
        used += 1

    if total_n == 0:
        raise RuntimeError("No valid RIDI frames found to compute stats.")

    mean = ch_sum / total_n
    std = np.sqrt(np.maximum(ch_sq / total_n - mean ** 2, 1e-8))

    payload = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "labels": ["acce_x", "acce_y", "acce_z", "gyro_x", "gyro_y", "gyro_z"],
        "frames_processed": int(total_n),
        "sequences_used": int(used),
        "missing_sequences": missing,
    }
    np.save(OUT_PATH, payload)

    print(f"RIDI stats saved to {OUT_PATH}")
    print(f"Sequences used  : {used}/{len(seqs)}")
    print(f"Frames processed: {total_n:,}")
    print(f"Channel mean    : {np.round(mean, 4)}")
    print(f"Channel std     : {np.round(std, 4)}")
    if missing:
        print(f"Missing sequences ({len(missing)}): {missing[:10]}")


if __name__ == "__main__":
    main()
