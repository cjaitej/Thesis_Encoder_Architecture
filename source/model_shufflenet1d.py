import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    b, c, t = x.size()
    x = x.view(b, groups, c // groups, t)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, t)


class ConvBNAct1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConvBN1D(nn.Module):
    def __init__(self, in_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm1d(in_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class ShuffleV2Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.stride = stride
        branch_ch = out_ch // 2
        if stride == 1:
            self.branch2 = nn.Sequential(
                ConvBNAct1D(branch_ch, branch_ch, k=1, s=1),
                DWConvBN1D(branch_ch, k=3, s=1),
                ConvBNAct1D(branch_ch, branch_ch, k=1, s=1),
            )
        else:
            self.branch1 = nn.Sequential(
                DWConvBN1D(in_ch, k=3, s=2),
                ConvBNAct1D(in_ch, branch_ch, k=1, s=1),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct1D(in_ch, branch_ch, k=1, s=1),
                DWConvBN1D(branch_ch, k=3, s=2),
                ConvBNAct1D(branch_ch, branch_ch, k=1, s=1),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2_1D(nn.Module):
    def __init__(self, in_channels=6, num_outputs=2, stage_repeats=(4, 8, 4),
                 stage_out_channels=(24, 96, 192, 384, 512), dropout=0.2):
        super().__init__()
        self.stem = ConvBNAct1D(in_channels, stage_out_channels[0], k=3, s=2)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        input_ch = stage_out_channels[0]
        stages = []
        for repeats, out_ch in zip(stage_repeats, stage_out_channels[1:-1]):
            blocks = [ShuffleV2Block1D(input_ch, out_ch, stride=2)]
            for _ in range(repeats - 1):
                blocks.append(ShuffleV2Block1D(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            input_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.tail = ConvBNAct1D(input_ch, stage_out_channels[-1], k=1, s=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(stage_out_channels[-1], stage_out_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(stage_out_channels[-1] // 2, num_outputs),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.tail(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(in_channels=6, num_outputs=2, dropout=0.2):
    return ShuffleNetV2_1D(
        in_channels=in_channels,
        num_outputs=num_outputs,
        stage_repeats=(4, 8, 4),
        stage_out_channels=(24, 96, 192, 384, 512),
        dropout=dropout,
    )
