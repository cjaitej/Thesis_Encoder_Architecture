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
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True, groups=1):
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
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1D(in_ch, hidden, k=1, s=1))
        # Depthwise conv
        layers.append(ConvBNAct1D(hidden, hidden, k=3, s=stride, p=1, groups=hidden))
        layers.append(ConvBNAct1D(hidden, out_ch, k=1, s=1, act=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            return x + out
        return out


class MobileNetV2_1D(nn.Module):
    def __init__(self, in_channels=6, num_outputs=2, width_mult=0.9, dropout=0.2):
        super().__init__()
        cfg = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))

        self.stem = ConvBNAct1D(in_channels, input_channel, k=3, s=2)

        layers = []
        for t, c, n, s in cfg:
            out_ch = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual1D(input_channel, out_ch, stride, t))
                input_channel = out_ch
        self.features = nn.Sequential(*layers)

        self.tail = ConvBNAct1D(input_channel, last_channel, k=1, s=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(last_channel, last_channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(last_channel // 2, num_outputs),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.tail(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2, width_mult=0.9):
    return MobileNetV2_1D(
        in_channels=in_channels,
        num_outputs=num_outputs,
        width_mult=width_mult,
        dropout=dropout,
    )
