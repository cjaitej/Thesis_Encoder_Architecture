import argparse
from os import path as osp

import numpy as np

from data_glob_speed import GlobSpeedSequence
from data_ridi import RIDIGlobSpeedSequence


def read_list(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#" and s.strip()
        ]


def resolve_multi_root_list(roots, data_list):
    resolved = []
    missing = []
    for name in data_list:
        found = False
        for root in roots:
            p = osp.join(root, name)
            if osp.isdir(p):
                resolved.append(p)
                found = True
                break
        if not found:
            missing.append(name)
    return resolved, missing


def sample_features_from_paths(seq_type, seq_paths, max_samples, rng):
    feats = []
    for i, p in enumerate(seq_paths):
        seq = seq_type(p, interval=1)
        feats.append(seq.get_feature())
        if (i + 1) % 20 == 0 or (i + 1) == len(seq_paths):
            print(f"Loaded {i + 1}/{len(seq_paths)} sequences")

    feat = np.concatenate(feats, axis=0).astype(np.float64)
    if feat.shape[0] > max_samples:
        idx = rng.choice(feat.shape[0], size=max_samples, replace=False)
        feat = feat[idx]
    return feat


def kl_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def main():
    parser = argparse.ArgumentParser(description="Compute RIDI vs RoNIN distribution divergence")
    parser.add_argument("--ronin_root", type=str, required=True)
    parser.add_argument("--ronin_list", type=str, required=True)
    parser.add_argument("--ridi_root", type=str, required=True)
    parser.add_argument("--ridi_list", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for p in [args.ronin_list, args.ridi_root, args.ridi_list]:
        if not osp.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")

    rng = np.random.default_rng(args.seed)

    ronin_data = read_list(args.ronin_list)
    ridi_data = read_list(args.ridi_list)

    ronin_roots = [s.strip() for s in args.ronin_root.split(",") if s.strip()]
    for r in ronin_roots:
        if not osp.exists(r):
            raise FileNotFoundError(f"Path not found: {r}")
    ronin_paths, missing = resolve_multi_root_list(ronin_roots, ronin_data)
    if missing:
        print(f"Warning: {len(missing)} RoNIN sequences were not found across provided roots")
        print("First missing:", missing[:10])

    ridi_paths = [osp.join(args.ridi_root, s) for s in ridi_data if osp.isdir(osp.join(args.ridi_root, s))]
    ridi_missing = [s for s in ridi_data if not osp.isdir(osp.join(args.ridi_root, s))]
    if ridi_missing:
        print(f"Warning: {len(ridi_missing)} RIDI sequences were not found")
        print("First missing:", ridi_missing[:10])

    print(f"Loading RoNIN features from {len(ronin_paths)} sequences...")
    x_ronin = sample_features_from_paths(GlobSpeedSequence, ronin_paths, args.max_samples, rng)

    print(f"Loading RIDI features from {len(ridi_paths)} sequences...")
    x_ridi = sample_features_from_paths(RIDIGlobSpeedSequence, ridi_paths, args.max_samples, rng)

    print(f"RoNIN sample shape: {x_ronin.shape}")
    print(f"RIDI sample shape: {x_ridi.shape}")

    channel_names = ["gyro_x", "gyro_y", "gyro_z", "acce_x", "acce_y", "acce_z"]

    print("\nChannel-wise divergence (histogram-based):")
    print("channel, KL(RIDI||RoNIN), KL(RoNIN||RIDI), JS")

    kl_pr_all = []
    kl_rp_all = []
    js_all = []

    for c, name in enumerate(channel_names):
        a = x_ridi[:, c]
        b = x_ronin[:, c]

        lo = float(min(np.min(a), np.min(b)))
        hi = float(max(np.max(a), np.max(b)))
        if hi <= lo:
            hi = lo + 1e-6

        pa, edges = np.histogram(a, bins=args.bins, range=(lo, hi), density=False)
        pb, _ = np.histogram(b, bins=edges, density=False)

        pa = pa.astype(np.float64)
        pb = pb.astype(np.float64)

        kl_pr = kl_divergence(pa, pb)
        kl_rp = kl_divergence(pb, pa)
        js = js_divergence(pa, pb)

        kl_pr_all.append(kl_pr)
        kl_rp_all.append(kl_rp)
        js_all.append(js)

        print(f"{name},{kl_pr:.6f},{kl_rp:.6f},{js:.6f}")

    print("\nSummary:")
    print(f"Mean KL(RIDI||RoNIN): {np.mean(kl_pr_all):.6f}")
    print(f"Mean KL(RoNIN||RIDI): {np.mean(kl_rp_all):.6f}")
    print(f"Mean JS: {np.mean(js_all):.6f}")

    print("\nPer-dataset moments:")
    print("channel, ronin_mean, ronin_std, ridi_mean, ridi_std")
    for c, name in enumerate(channel_names):
        print(
            f"{name},"
            f"{x_ronin[:, c].mean():.6f},{x_ronin[:, c].std():.6f},"
            f"{x_ridi[:, c].mean():.6f},{x_ridi[:, c].std():.6f}"
        )


if __name__ == "__main__":
    main()
