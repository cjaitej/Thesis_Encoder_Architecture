import argparse
import os

import torch

from model_yolo26_1d import YOLO26_1D_Regressor


def main():
    parser = argparse.ArgumentParser(description='Extract and save full pretrained model state')
    parser.add_argument('--ckpt', type=str,
                        default='./output/pretrain_ridi_gpu/checkpoints/checkpoint_latest.pt',
                        help='Path to source training checkpoint (.pt)')
    parser.add_argument('--out', type=str,
                        default='./output/pretrain_ridi/ridi_pretrained_full.pt',
                        help='Path to output extracted checkpoint (.pt)')
    args = parser.parse_args()

    model = YOLO26_1D_Regressor()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'pretrained_on': 'ridi',
        'epochs_pretrained': ckpt.get('epoch', 20),
        'ridi_val_loss': ckpt.get('val_loss', None),
    }, args.out)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
