import torch
import torch.nn as nn


class DWSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class TinyCNN1D(nn.Module):
    def __init__(self, in_channels=6, num_outputs=2, dropout=0.2):
        super().__init__()
        channels = [32, 48, 64, 96, 128, 128, 160, 192]
        strides = [1, 2, 1, 2, 1, 2, 1, 2]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = channels[0]
        for out_ch, s in zip(channels, strides):
            blocks.append(DWSeparableConv1D(in_ch, out_ch, k=3, s=s))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1] // 2, num_outputs),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2):
    return TinyCNN1D(in_channels=in_channels, num_outputs=num_outputs, dropout=dropout)
