import argparse
import json
import os
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

from ronin_yolo26_baseline_plain import get_dataset, get_model, run_test
from rf_utils import build_sequence_features


def parse_folder_args(values):
    if isinstance(values, str):
        values = [values]

    folders = []
    for token in values:
        for part in token.split(','):
            part = part.strip()
            if part:
                folders.append(part)
    return folders


def parse_sequence_list(list_path):
    with open(list_path) as f:
        return [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']


def collect_sequences_from_folders(data_root, folder_names):
    sequences = []
    resolved_folders = []

    alias_map = {
        'seen_subject_test_set': 'seen_subjects_test_set',
        'seen_subjects_test_set': 'seen_subjects_test_set',
    }

    for raw_folder in folder_names:
        folder = alias_map.get(raw_folder, raw_folder)
        folder_path = osp.join(data_root, folder)
        if not osp.isdir(folder_path):
            raise FileNotFoundError(f'Data folder not found: {folder_path}')

        resolved_folders.append(folder)

        for item in sorted(os.listdir(folder_path)):
            item_path = osp.join(folder_path, item)
            if osp.isdir(item_path):
                sequences.append(f'{folder}/{item}')

    if not sequences:
        raise ValueError('No trajectories found in provided folders')

    return sequences, resolved_folders


def split_sequences(sequences, train_ratio, seed):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError('train_ratio must be between 0 and 1')

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sequences))
    train_count = int(np.floor(len(sequences) * train_ratio))
    train_count = max(1, min(train_count, len(sequences) - 1))

    train_idx = perm[:train_count]
    val_idx = perm[train_count:]

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]

    return train_sequences, val_sequences


def export_split(sequence_names, split_name, out_path, network, device, args, feature_names):
    X_all, y_all = [], []
    pred_all, gt_all = [], []
    seq_name_all, frame_all, ts_all = [], [], []

    for i, seq_name in enumerate(sequence_names):
        seq_dataset = get_dataset(args.root_dir, [seq_name], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=False)

        targets, preds = run_test(network, seq_loader, device, eval_mode=True)

        sparse_ind = np.array([idx[1] for idx in seq_dataset.index_map if idx[0] == 0], dtype=int)
        feat_sparse = seq_dataset.features[0][sparse_ind]
        ts_sparse = seq_dataset.ts[0][sparse_ind]

        X_seq, _ = build_sequence_features(
            preds=preds,
            feat_sparse=feat_sparse,
            ts=ts_sparse,
            hist_window=args.hist_window,
        )
        residual = (targets - preds).astype(np.float32)

        X_all.append(X_seq)
        y_all.append(residual)
        pred_all.append(preds.astype(np.float32))
        gt_all.append(targets.astype(np.float32))

        seq_name_all.extend([seq_name] * X_seq.shape[0])
        frame_all.append(sparse_ind.astype(np.int32))
        ts_all.append(ts_sparse.astype(np.float64))

        print(f'[{split_name}] [{i + 1}/{len(sequence_names)}] {seq_name}: samples={X_seq.shape[0]}')

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)
    gt = np.concatenate(gt_all, axis=0)
    frame = np.concatenate(frame_all, axis=0)
    ts = np.concatenate(ts_all, axis=0)
    seq_name_arr = np.array(seq_name_all, dtype=object)

    out_dir = osp.dirname(out_path)
    if out_dir and not osp.isdir(out_dir):
        os.makedirs(out_dir)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        pred=pred,
        gt=gt,
        frame=frame,
        ts=ts,
        seq_name=seq_name_arr,
        feature_names=np.array(feature_names, dtype=object),
        meta=np.array([
            json.dumps(
                {
                    'model': 'yolo26',
                    'arch': args.arch,
                    'use_attention': args.use_attention,
                    'model_dropout': args.model_dropout,
                    'root_dir': args.root_dir,
                    'window_size': args.window_size,
                    'step_size': args.step_size,
                    'hist_window': args.hist_window,
                    'sparse_alignment': True,
                    'interpolated': False,
                    'split': split_name,
                    'num_trajectories': len(sequence_names),
                }
            )
        ], dtype=object),
    )

    print(f'Saved {split_name} RF dataset:')
    print(f'  file: {out_path}')
    print(f'  trajectories: {len(sequence_names)}')
    print(f'  samples: {X.shape[0]}')
    print(f'  features: {X.shape[1]}')


def main():
    parser = argparse.ArgumentParser(description='Prepare RF residual dataset from YOLO26 predictions.')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--list_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--train_out_path', type=str, default=None)
    parser.add_argument('--val_out_path', type=str, default=None)
    parser.add_argument(
        '--data_folders',
        nargs='+',
        default=['train_dataset_1,seen_subjects_test_set'],
        help='Folders under root_dir. Accepts comma-separated and/or space-separated values.',
    )
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--split_seed', type=int, default=42)

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hist_window', type=int, default=50)
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model_dropout', type=float, default=0.2)
    parser.add_argument('--use_attention', action='store_true')

    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path)

    network = get_model(args.arch, model_dropout=args.model_dropout, use_attention=args.use_attention)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)

    _, feature_names = build_sequence_features(
        preds=np.zeros((2, 2), dtype=np.float32),
        feat_sparse=np.zeros((2, 6), dtype=np.float32),
        ts=np.array([0.0, 1.0], dtype=np.float32),
        hist_window=args.hist_window,
    )

    if args.list_path is not None and args.out_path is not None:
        sequence_names = parse_sequence_list(args.list_path)
        export_split(
            sequence_names=sequence_names,
            split_name='all',
            out_path=args.out_path,
            network=network,
            device=device,
            args=args,
            feature_names=feature_names,
        )
        return

    if args.train_out_path is None or args.val_out_path is None:
        raise ValueError('Provide --train_out_path and --val_out_path for split mode')

    folder_names = parse_folder_args(args.data_folders)
    all_sequences, resolved_folders = collect_sequences_from_folders(args.root_dir, folder_names)
    train_sequences, val_sequences = split_sequences(all_sequences, args.split_ratio, args.split_seed)

    print('RF trajectory split summary:')
    print(f'  root_dir: {args.root_dir}')
    print(f"  folders: {', '.join(resolved_folders)}")
    print(f'  total trajectories: {len(all_sequences)}')
    print(f'  train trajectories: {len(train_sequences)}')
    print(f'  val trajectories: {len(val_sequences)}')
    print(f'  split_ratio: {args.split_ratio}, seed: {args.split_seed}')

    export_split(
        sequence_names=train_sequences,
        split_name='train',
        out_path=args.train_out_path,
        network=network,
        device=device,
        args=args,
        feature_names=feature_names,
    )
    export_split(
        sequence_names=val_sequences,
        split_name='val',
        out_path=args.val_out_path,
        network=network,
        device=device,
        args=args,
        feature_names=feature_names,
    )


if __name__ == '__main__':
    main()
