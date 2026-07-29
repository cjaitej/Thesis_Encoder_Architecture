"""
Train / evaluate the neural residual-refinement stage (model_refinement.RefinementNet)
on top of a frozen window-regression backbone (e.g. --arch yolo26_eff).

Two-stage design (matches ResidualNav's structure so it is directly comparable):
  1. Train the backbone as usual with ronin_resnet.py.
  2. This script freezes that backbone, runs it over each trajectory to get the
     per-window velocity-prediction sequence, and trains RefinementNet to correct
     those sequences (residual learning). At test time it applies backbone ->
     refinement -> integrate -> ATE/RTE, and prints baseline vs refined side by side.

Train:
  python refinement.py --mode train --arch yolo26_eff \
    --backbone_path output/yolo26_eff_v1/checkpoints/checkpoint_XX.pt \
    --root_dir data/train_dataset --train_list lists/list_train_fixed.txt \
    --cache_path /tmp/ronin_cache_resnet --out_dir output/refine_v1 --epochs 100

Test:
  python refinement.py --mode test --arch yolo26_eff \
    --backbone_path output/yolo26_eff_v1/checkpoints/checkpoint_XX.pt \
    --model_path output/refine_v1/refine_best.pt \
    --root_dir data/seen_subjects_test_set --test_list lists/list_test_seen.txt \
    --cache_path /tmp/ronin_cache_resnet
"""
import os
import argparse
import random
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

# reuse the backbone pipeline exactly (dataset, inference, trajectory reconstruction)
from ronin_resnet import get_dataset, run_test, recon_traj_with_preds, get_model
from metric import compute_ate_rte
from model_refinement import RefinementNet

_PRED_PER_MIN = 200 * 60


def _device(args):
    if torch.cuda.is_available() and not args.cpu:
        return torch.device('cuda:0')
    return torch.device('cpu')


def _read_list(path):
    with open(path) as f:
        return [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']


def load_backbone(args, device):
    net = get_model(args.arch)
    ckpt = torch.load(args.backbone_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval().to(device)
    print('Backbone {} loaded ({} params).'.format(
        args.arch, sum(p.numel() for p in net.parameters())))
    return net


def collect_sequences(backbone, data_list, root_dir, args, device):
    """For each trajectory: run the frozen backbone to get (preds, targets, dataset)."""
    seqs = []
    for data in data_list:
        ds = get_dataset(root_dir, [data], args, mode='test')
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        targets, preds = run_test(backbone, loader, device, True)   # each [N, 2]
        seqs.append((data, ds, preds.astype(np.float32), targets.astype(np.float32)))
    return seqs


def _refine_loss(corrected, targets, drift_weight):
    mse = torch.mean((corrected - targets) ** 2)
    if drift_weight > 0:
        # integrated-position (drift) term -- targets ATE, not just per-window error
        n = corrected.shape[1]
        pred_pos = torch.cumsum(corrected, dim=1)
        gt_pos = torch.cumsum(targets, dim=1)
        mse = mse + drift_weight * torch.mean((pred_pos - gt_pos) ** 2) / n
    return mse


def eval_loss(refine, seqs, device, drift_weight):
    """Average refinement loss over a set of (frozen-backbone) trajectories, no grad."""
    refine.eval()
    total = 0.0
    with torch.no_grad():
        for _, _, preds, targets in seqs:
            p = torch.from_numpy(preds).unsqueeze(0).to(device)
            t = torch.from_numpy(targets).unsqueeze(0).to(device)
            total += _refine_loss(refine(p), t, drift_weight).item()
    return total / len(seqs)


def train(args):
    device = _device(args)
    backbone = load_backbone(args, device)

    seqs = collect_sequences(backbone, _read_list(args.train_list), args.root_dir, args, device)
    print('Collected {} training trajectories.'.format(len(seqs)))
    val_seqs = None
    if args.val_list:
        val_seqs = collect_sequences(backbone, _read_list(args.val_list), args.root_dir, args, device)
        print('Collected {} validation trajectories.'.format(len(val_seqs)))

    refine = RefinementNet(hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    print('RefinementNet: {} params'.format(refine.get_num_params()))
    optimizer = torch.optim.Adam(refine.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=15, threshold=1e-2, verbose=True)

    if args.out_dir and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    best = np.inf
    for epoch in range(args.epochs):
        refine.train()
        random.shuffle(seqs)
        total = 0.0
        for _, _, preds, targets in seqs:
            p = torch.from_numpy(preds).unsqueeze(0).to(device)      # [1, N, 2]
            t = torch.from_numpy(targets).unsqueeze(0).to(device)
            optimizer.zero_grad()
            loss = _refine_loss(refine(p), t, args.drift_weight)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_avg = total / len(seqs)
        val_avg = eval_loss(refine, val_seqs, device, args.drift_weight) if val_seqs else train_avg
        scheduler.step(val_avg)                       # schedule/select on val when available
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print('epoch {:3d}  train {:.6f}  val {:.6f}  lr {:.2e}'.format(
                epoch, train_avg, val_avg, optimizer.param_groups[0]['lr']))
        if val_avg < best and args.out_dir:
            best = val_avg
            torch.save({'model_state_dict': refine.state_dict(), 'epoch': epoch,
                        'hidden': args.hidden, 'layers': args.layers},
                       osp.join(args.out_dir, 'refine_best.pt'))
    tag = 'val' if val_seqs else 'train'
    print('Done. Best {} loss {:.6f}. Saved to {}/refine_best.pt'.format(tag, best, args.out_dir))


def test(args):
    device = _device(args)
    backbone = load_backbone(args, device)

    ckpt = torch.load(args.model_path, map_location=device)
    refine = RefinementNet(hidden=ckpt.get('hidden', args.hidden),
                           num_layers=ckpt.get('layers', args.layers)).to(device)
    refine.load_state_dict(ckpt['model_state_dict'])
    refine.eval()
    print('RefinementNet loaded from {}.'.format(args.model_path))

    seqs = collect_sequences(backbone, _read_list(args.test_list), args.root_dir, args, device)

    print('\n{:12s} {:>8s} {:>8s}   {:>8s} {:>8s}'.format('seq', 'ATE', 'RTE', 'ATE_ref', 'RTE_ref'))
    base_ate, base_rte, ref_ate, ref_rte = [], [], [], []
    for data, ds, preds, targets in seqs:
        with torch.no_grad():
            refined = refine(torch.from_numpy(preds).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        pos_gt = ds.gt_pos[0][:, :2]
        pos_base = recon_traj_with_preds(ds, preds)[:, :2]
        pos_ref = recon_traj_with_preds(ds, refined)[:, :2]
        a_b, r_b = compute_ate_rte(pos_base, pos_gt, _PRED_PER_MIN)
        a_r, r_r = compute_ate_rte(pos_ref, pos_gt, _PRED_PER_MIN)
        base_ate.append(a_b); base_rte.append(r_b); ref_ate.append(a_r); ref_rte.append(r_r)
        print('{:12s} {:8.3f} {:8.3f}   {:8.3f} {:8.3f}'.format(data, a_b, r_b, a_r, r_r))

    print('\n{:12s} {:>8s} {:>8s}   {:>8s} {:>8s}'.format('MEAN', 'ATE', 'RTE', 'ATE_ref', 'RTE_ref'))
    print('{:12s} {:8.3f} {:8.3f}   {:8.3f} {:8.3f}'.format(
        '', np.mean(base_ate), np.mean(base_rte), np.mean(ref_ate), np.mean(ref_rte)))
    print('delta: ATE {:+.3f}  RTE {:+.3f}'.format(
        np.mean(ref_ate) - np.mean(base_ate), np.mean(ref_rte) - np.mean(base_rte)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'test'], required=True)
    p.add_argument('--arch', type=str, required=True)
    p.add_argument('--backbone_path', type=str, required=True)
    p.add_argument('--model_path', type=str, default=None, help='RefinementNet checkpoint (test)')
    p.add_argument('--root_dir', type=str, required=True)
    p.add_argument('--train_list', type=str, default=None)
    p.add_argument('--val_list', type=str, default=None)
    p.add_argument('--test_list', type=str, default=None)
    p.add_argument('--cache_path', type=str, default=None)
    p.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    p.add_argument('--step_size', type=int, default=10)
    p.add_argument('--window_size', type=int, default=200)
    p.add_argument('--max_ori_error', type=float, default=20.0)
    p.add_argument('--out_dir', type=str, default=None)
    # refinement hyper-params
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--drift_weight', type=float, default=0.0,
                   help='weight of the integrated-position (drift/ATE) loss term; 0 = velocity MSE only')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        if not args.model_path:
            raise ValueError('--model_path (RefinementNet checkpoint) required for test')
        test(args)
