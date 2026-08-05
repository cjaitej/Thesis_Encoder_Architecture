import torch
import torch.nn as nn

from tcn import TemporalConvNet


class LightTCN1D(nn.Module):
    def __init__(self, in_channels=6, num_outputs=2, channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if channels is None:
            channels = [48, 48, 48, 48, 48, 48]
        self.tcn = TemporalConvNet(in_channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1] // 2, num_outputs),
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2):
    return LightTCN1D(in_channels=in_channels, num_outputs=num_outputs, dropout=dropout)
