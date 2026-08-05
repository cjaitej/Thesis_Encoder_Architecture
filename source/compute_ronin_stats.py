"""
Computes per-channel mean and std from RoNIN training set.
Channels: [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z]
Output: ../output/stats/ronin_channel_stats.npy
"""

import os
import json
import h5py
import numpy as np

ROOT_CANDIDATES = [
    "../data/train_dataset_1",
    "../data/train_dataset_2",
]
TRAIN_LIST = "../lists/list_train.txt"
OUT_PATH = "../output/stats/ronin_channel_stats.npy"


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def resolve_seq_path(seq_name):
    for root in ROOT_CANDIDATES:
        seq_path = os.path.join(root, seq_name)
        if os.path.isdir(seq_path):
            return seq_path
    return None


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    seqs = read_list(TRAIN_LIST)
    ch_sum = np.zeros(6, dtype=np.float64)
    ch_sq = np.zeros(6, dtype=np.float64)
    total_n = 0
    used = 0
    missing = []

    for s in seqs:
        seq_path = resolve_seq_path(s)
        if seq_path is None:
            missing.append(s)
            continue

        h5_path = os.path.join(seq_path, "data.hdf5")
        if not os.path.exists(h5_path):
            missing.append(s)
            continue

        with h5py.File(h5_path, "r") as f:
            acce = np.array(f["synced/acce"][:, :3], dtype=np.float64)
            if "synced/gyro" in f:
                gyro = np.array(f["synced/gyro"][:, :3], dtype=np.float64)
            else:
                gyro = np.array(f["synced/gyro_uncalib"][:, :3], dtype=np.float64)
            imu = np.concatenate([acce, gyro], axis=1)

        ch_sum += imu.sum(axis=0)
        ch_sq += (imu ** 2).sum(axis=0)
        total_n += imu.shape[0]
        used += 1

    if total_n == 0:
        raise RuntimeError("No valid RoNIN frames found to compute stats.")

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

    print(f"RoNIN stats saved to {OUT_PATH}")
    print(f"Sequences used  : {used}/{len(seqs)}")
    print(f"Frames processed: {total_n:,}")
    print(f"Channel mean    : {np.round(mean, 4)}")
    print(f"Channel std     : {np.round(std, 4)}")
    if missing:
        print(f"Missing sequences ({len(missing)}): {missing[:10]}")


if __name__ == "__main__":
    main()
