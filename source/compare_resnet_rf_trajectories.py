import argparse
import csv
import os
from os import path as osp

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import ronin_resnet as rr
from metric import compute_ate_rte
from rf_utils import apply_rf_correction, build_sequence_features


def parse_list(list_path):
    with open(list_path) as f:
        return [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']


def ensure_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare RoNIN ResNet vs ResNet+RF trajectories on seen/unseen sets."
    )
    parser.add_argument('--seen_list', type=str, required=True)
    parser.add_argument('--unseen_list', type=str, required=True)
    parser.add_argument('--seen_root', type=str, required=True)
    parser.add_argument('--unseen_root', type=str, required=True)

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--rf_model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--cache_path', type=str, default=None)

    parser.add_argument('--rf_alpha', type=float, default=1.0)
    parser.add_argument('--rf_clip', type=float, default=0.75)
    parser.add_argument('--rf_hist_window', type=int, default=50)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def build_network(args, device):
    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path)

    # Trigger dataset feature/target shape initialization inside ronin_resnet.
    _ = rr.get_dataset(args.seen_root, [parse_list(args.seen_list)[0]], args)
    rr._fc_config['in_dim'] = args.window_size // 32 + 1

    network = rr.get_model(args.arch)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    return network


def evaluate_split(split_name, root_dir, sequence_names, args, network, rf_model, device):
    split_dir = osp.join(args.out_dir, split_name)
    ensure_dir(split_dir)

    pred_per_min = 200 * 60
    rows = []

    for i, seq_name in enumerate(sequence_names):
        seq_dataset = rr.get_dataset(root_dir, [seq_name], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=False)
        ind = np.array([idx[1] for idx in seq_dataset.index_map if idx[0] == 0], dtype=int)

        targets, preds_base = rr.run_test(network, seq_loader, device, eval_mode=True)

        feat_sparse = seq_dataset.features[0][ind]
        ts_sparse = seq_dataset.ts[0][ind]
        rf_X, _ = build_sequence_features(preds_base, feat_sparse, ts_sparse, args.rf_hist_window)
        residual_hat = rf_model.predict(rf_X).astype(np.float32)
        preds_rf = apply_rf_correction(
            preds_base,
            residual_hat,
            alpha=args.rf_alpha,
            residual_clip=args.rf_clip,
        )

        pos_gt = seq_dataset.gt_pos[0][:, :2]
        pos_base = rr.recon_traj_with_preds(seq_dataset, preds_base)[:, :2]
        pos_rf = rr.recon_traj_with_preds(seq_dataset, preds_rf)[:, :2]

        ate_base, rte_base = compute_ate_rte(pos_base, pos_gt, pred_per_min)
        ate_rf, rte_rf = compute_ate_rte(pos_rf, pos_gt, pred_per_min)

        mse_base = np.mean((targets - preds_base) ** 2, axis=0)
        mse_rf = np.mean((targets - preds_rf) ** 2, axis=0)

        rows.append({
            'split': split_name,
            'sequence': seq_name,
            'ate_base': float(ate_base),
            'rte_base': float(rte_base),
            'ate_rf': float(ate_rf),
            'rte_rf': float(rte_rf),
            'ate_improvement': float(ate_base - ate_rf),
            'rte_improvement': float(rte_base - rte_rf),
            'mse_vx_base': float(mse_base[0]),
            'mse_vy_base': float(mse_base[1]),
            'mse_vx_rf': float(mse_rf[0]),
            'mse_vy_rf': float(mse_rf[1]),
        })

        plt.figure(figsize=(11, 8))
        plt.plot(pos_gt[:, 0], pos_gt[:, 1], linewidth=2.2, color='black', label='Ground Truth')
        plt.plot(pos_base[:, 0], pos_base[:, 1], linewidth=1.8, color='#1f77b4', label='ResNet')
        plt.plot(pos_rf[:, 0], pos_rf[:, 1], linewidth=1.8, color='#d62728', label='ResNet + RF')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(
            f"{seq_name} ({split_name})\\n"
            f"ATE base={ate_base:.3f}, RF={ate_rf:.3f} | RTE base={rte_base:.3f}, RF={rte_rf:.3f}"
        )
        plt.legend(loc='best')
        plt.tight_layout()

        fig_path = osp.join(split_dir, f"{seq_name}_traj_compare.png")
        npy_path = osp.join(split_dir, f"{seq_name}_traj_compare.npy")
        plt.savefig(fig_path, dpi=180)
        plt.close('all')

        np.save(
            npy_path,
            np.concatenate([pos_gt, pos_base, pos_rf], axis=1),
        )

        print(
            f"[{split_name}] [{i + 1}/{len(sequence_names)}] {seq_name}: "
            f"ATE base={ate_base:.4f}, RF={ate_rf:.4f}, "
            f"RTE base={rte_base:.4f}, RF={rte_rf:.4f}"
        )

    return rows


def write_summary_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'split',
                'sequence',
                'ate_base',
                'rte_base',
                'ate_rf',
                'rte_rf',
                'ate_improvement',
                'rte_improvement',
                'mse_vx_base',
                'mse_vy_base',
                'mse_vx_rf',
                'mse_vy_rf',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_aggregate(rows, split_name):
    split_rows = [r for r in rows if r['split'] == split_name]
    if not split_rows:
        return
    ate_base = np.mean([r['ate_base'] for r in split_rows])
    ate_rf = np.mean([r['ate_rf'] for r in split_rows])
    rte_base = np.mean([r['rte_base'] for r in split_rows])
    rte_rf = np.mean([r['rte_rf'] for r in split_rows])
    print(
        f"[{split_name}] avg ATE base={ate_base:.4f}, RF={ate_rf:.4f} | "
        f"avg RTE base={rte_base:.4f}, RF={rte_rf:.4f}"
    )


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print(f"Running on device: {device}")

    seen_sequences = parse_list(args.seen_list)
    unseen_sequences = parse_list(args.unseen_list)

    network = build_network(args, device)
    rf_model = joblib.load(args.rf_model_path)
    print(f"Loaded model: {args.model_path}")
    print(f"Loaded RF: {args.rf_model_path}")

    all_rows = []
    all_rows.extend(evaluate_split('seen', args.seen_root, seen_sequences, args, network, rf_model, device))
    all_rows.extend(evaluate_split('unseen', args.unseen_root, unseen_sequences, args, network, rf_model, device))

    summary_csv = osp.join(args.out_dir, 'comparison_metrics.csv')
    write_summary_csv(summary_csv, all_rows)
    print_aggregate(all_rows, 'seen')
    print_aggregate(all_rows, 'unseen')
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == '__main__':
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    main()
