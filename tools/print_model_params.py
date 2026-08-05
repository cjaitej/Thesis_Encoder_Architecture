#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

from model_yolo26_1d import YOLO26_1D_Regressor
from model_mobilenet1d import MobileNetV2_1D
from model_shufflenet1d import ShuffleNetV2_1D
from model_efficientnet_lite1d import EfficientNetLite0_1D
from model_tinycnn1d import TinyCNN1D
from model_lighttcn1d import LightTCN1D


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='all',
                        choices=['all', 'yolo26', 'mobilenetv2', 'shufflenetv2',
                                 'efficientnet_lite0', 'tinycnn', 'lighttcn'])
    args = parser.parse_args()

    models = {
        'yolo26': YOLO26_1D_Regressor(
            in_channels=6,
            num_outputs=2,
            base_ch=32,
            widths=(64, 128, 256),
            n_blocks=(1, 2, 2),
            dropout=0.2,
            use_attention=False,
        ),
        'mobilenetv2': MobileNetV2_1D(in_channels=6, num_outputs=2, width_mult=0.9, dropout=0.2),
        'shufflenetv2': ShuffleNetV2_1D(in_channels=6, num_outputs=2, dropout=0.2),
        'efficientnet_lite0': EfficientNetLite0_1D(in_channels=6, num_outputs=2, width_mult=1.0, dropout=0.2),
        'tinycnn': TinyCNN1D(in_channels=6, num_outputs=2, dropout=0.2),
        'lighttcn': LightTCN1D(in_channels=6, num_outputs=2, dropout=0.2),
    }

    if args.backbone == 'all':
        for name, model in models.items():
            print(f"{name}: {count_params(model):,}")
    else:
        model = models[args.backbone]
        print(f"{args.backbone}: {count_params(model):,}")


if __name__ == '__main__':
    main()
