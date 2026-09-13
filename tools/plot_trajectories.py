"""Paper trajectory figures: ground truth vs RoNIN-ResNet vs YOLOv26-1D-Eff + RF.

Reads the <seq>_gsn.npy files written by the test scripts
([pred_x, pred_y, gt_x, gt_y] per frame) and the per-sequence ATE/RTE printed
in each run's test.log, so the numbers in the figure match the paper tables.

Step 1 - list every sequence with both models' errors, then choose:
    python tools/plot_trajectories.py --list \
        --resnet_dir output/test_resnet/seen \
        --ours_dir   output/test_yolo26_eff/seen_A_rf

Step 2 - plot the chosen sequences (one PNG + PDF each, plus a 2x2 overview):
    python tools/plot_trajectories.py \
        --resnet_dir output/test_resnet/seen \
        --ours_dir   output/test_yolo26_eff/seen_A_rf \
        --seqs a001_2 a003_3 a005_1 a006_2 --out_dir paper
    -> paper/traj1_large_text.png ... traj4_large_text.png (names used by paper.tex)
"""
import argparse
import os
import re
import sys
from os import path as osp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito blue / bluish green / vermilion: passes colorblind (protan, deutan,
# tritan) separation. ResNet is also dashed so the figure survives grayscale print.
STYLE = {
    'gt':     dict(color='#0072B2', lw=1.8, ls='-',  zorder=2),
    'resnet': dict(color='#009E73', lw=1.5, ls='--', zorder=3),
    'ours':   dict(color='#D55E00', lw=1.5, ls='-',  zorder=4),
}
INK = '#1a1a1a'
GRID = '#d9d9d9'

SEQ_RE = re.compile(r'^Sequence\s+(\S+),.*?\bate\s+([\d.eE+-]+),\s*rte\s+([\d.eE+-]+)')


def read_metrics(run_dir):
    """{seq: (ate, rte)} from test.log; the last line per sequence wins."""
    log = osp.join(run_dir, 'test.log')
    if not osp.isfile(log):
        sys.exit('ERROR: no test.log in %s' % run_dir)
    out = {}
    with open(log, errors='replace') as f:
        for line in f:
            m = SEQ_RE.match(line)
            if m:
                out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def load_traj(run_dir, seq):
    p = osp.join(run_dir, seq + '_gsn.npy')
    if not osp.isfile(p):
        return None
    a = np.load(p)
    return a[:, 0:2], a[:, 2:4]   # predicted, ground truth


def list_sequences(args, res_m, our_m):
    seqs = sorted(set(res_m) & set(our_m))
    print('%-10s %17s %17s %9s %9s  npy' % ('sequence', 'ResNet ATE/RTE', 'Ours ATE/RTE',
                                            'dATE', 'dRTE'))
    print('-' * 72)
    both_better = 0
    for s in seqs:
        (ra, rr), (oa, orr) = res_m[s], our_m[s]
        has = (osp.isfile(osp.join(args.resnet_dir, s + '_gsn.npy')) and
               osp.isfile(osp.join(args.ours_dir, s + '_gsn.npy')))
        both_better += (oa < ra and orr < rr)
        print('%-10s %8.2f / %6.2f %8.2f / %6.2f %+9.2f %+9.2f  %s' %
              (s, ra, rr, oa, orr, oa - ra, orr - rr, 'yes' if has else 'MISSING'))
    print('-' * 72)
    print('%d sequences; ours lower on both ATE and RTE in %d.' % (len(seqs), both_better))
    print('Negative dATE/dRTE = YOLOv26-1D-Eff + RF is better.')


def draw(ax, seq, res, ours, res_m, our_m, labels, fs):
    (res_pred, res_gt), (our_pred, our_gt) = res, ours
    gt = our_gt
    n = min(len(res_gt), len(our_gt))
    if len(res_gt) != len(our_gt) or np.max(np.abs(res_gt[:n] - our_gt[:n])) > 1e-3:
        print('WARNING: %s ground truth differs between the two runs '
              '(lengths %d vs %d); plotting the YOLO run\'s ground truth.'
              % (seq, len(res_gt), len(our_gt)))

    # ATE/RTE live in the legend entries, next to each line sample.
    ra, rr = res_m[seq]
    oa, orr = our_m[seq]
    ax.plot(gt[:, 0], gt[:, 1], label=labels[0], **STYLE['gt'])
    ax.plot(res_pred[:, 0], res_pred[:, 1],
            label='%s  (A %.2f, R %.2f)' % (labels[1], ra, rr), **STYLE['resnet'])
    ax.plot(our_pred[:, 0], our_pred[:, 1],
            label='%s  (A %.2f, R %.2f)' % (labels[2], oa, orr), **STYLE['ours'])

    # Equal axes with headroom at the top, so the legend sits in empty space
    # instead of on the trajectories. A = ATE (m), R = RTE (m).
    pts = np.vstack([gt, res_pred, our_pred])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    span = max(x1 - x0, y1 - y0)
    pad = 0.06 * span
    cx = 0.5 * (x0 + x1)
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(y0 - pad, y0 - pad + 2 * half + 0.24 * span)
    ax.set_aspect('equal', adjustable='box')

    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color('#8c8c8c')
        sp.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=fs - 1, width=0.8)
    ax.set_xlabel('x (m)', fontsize=fs, color=INK)
    ax.set_ylabel('y (m)', fontsize=fs, color=INK)

    leg = ax.legend(loc='upper left', fontsize=fs - 2, frameon=True, framealpha=0.92,
                    edgecolor='#bfbfbf', handlelength=2.2, borderpad=0.4, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resnet_dir', default='output/test_resnet/seen')
    # Adam-trained YOLOv26-1D-Eff (checkpoint_59) + Random Forest, as in the paper tables.
    ap.add_argument('--ours_dir', default='output/test_yolo26_eff/seen_A_rf')
    ap.add_argument('--seqs', nargs='+', help='sequences to plot, in figure order')
    ap.add_argument('--list', action='store_true', help='print per-sequence ATE/RTE and exit')
    ap.add_argument('--out_dir', default='paper')
    ap.add_argument('--prefix', default='traj', help='files: <prefix><i>_large_text.png')
    ap.add_argument('--labels', nargs=3, default=['Ground Truth', 'RoNIN-ResNet', 'YOLOv26-1D-Eff + RF'])
    ap.add_argument('--font_size', type=float, default=11)
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    res_m, our_m = read_metrics(args.resnet_dir), read_metrics(args.ours_dir)
    if args.list or not args.seqs:
        list_sequences(args, res_m, our_m)
        if not args.seqs:
            print('\nPass --seqs to plot.')
        return

    loaded = []
    for s in args.seqs:
        if s not in res_m or s not in our_m:
            sys.exit('ERROR: %s has no ATE/RTE line in one of the test.log files' % s)
        res, ours = load_traj(args.resnet_dir, s), load_traj(args.ours_dir, s)
        if res is None or ours is None:
            sys.exit('ERROR: %s_gsn.npy missing in %s' %
                     (s, args.resnet_dir if res is None else args.ours_dir))
        loaded.append((s, res, ours))

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})

    for i, (s, res, ours) in enumerate(loaded, 1):
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        draw(ax, s, res, ours, res_m, our_m, args.labels, args.font_size)
        base = osp.join(args.out_dir, '%s%d_large_text' % (args.prefix, i))
        fig.savefig(base + '.png', dpi=args.dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(base + '.pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('saved %s.png/.pdf  (%s)' % (base, s))

    cols = 2 if len(loaded) > 1 else 1
    rows = int(np.ceil(len(loaded) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for ax, (s, res, ours) in zip(axes.flat, loaded):
        draw(ax, s, res, ours, res_m, our_m, args.labels, args.font_size)
    for ax in list(axes.flat)[len(loaded):]:
        ax.axis('off')
    fig.tight_layout()
    grid = osp.join(args.out_dir, '%s_overview.png' % args.prefix)
    fig.savefig(grid, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved %s  (preview only; paper.tex includes the individual files)' % grid)


if __name__ == '__main__':
    main()
