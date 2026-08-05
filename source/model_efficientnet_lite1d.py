import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MBConvLite1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, kernel_size):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1D(in_ch, hidden, k=1, s=1))
        # Depthwise conv
        layers.append(ConvBNAct1D(hidden, hidden, k=kernel_size, s=stride, groups=hidden))
        layers.append(nn.Conv1d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            return x + out
        return out


class EfficientNetLite0_1D(nn.Module):
    def __init__(self, in_channels=6, num_outputs=2, width_mult=0.6, dropout=0.2):
        super().__init__()
        cfg = [
            # (expand, c, n, s, k)
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        stem_ch = _make_divisible(32 * width_mult)
        self.stem = ConvBNAct1D(in_channels, stem_ch, k=3, s=2)

        layers = []
        in_ch = stem_ch
        for expand, c, n, s, k in cfg:
            out_ch = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConvLite1D(in_ch, out_ch, stride, expand, k))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        head_ch = _make_divisible(1280 * width_mult)
        self.head_conv = ConvBNAct1D(in_ch, head_ch, k=1, s=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(head_ch, head_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_ch // 2, num_outputs),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head_conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2, width_mult=0.6):
    return EfficientNetLite0_1D(
        in_channels=in_channels,
        num_outputs=num_outputs,
        width_mult=width_mult,
        dropout=dropout,
    )
