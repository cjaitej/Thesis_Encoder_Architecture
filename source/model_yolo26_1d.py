"""
source/model_yolo26_1d.py
Drop-in replacement for ResNet1D in the RoNIN pipeline.
Input:  (B, 6, 200)
Output: (B, 2)
All operations: 1D only (Conv1d, BatchNorm1d, AdaptiveAvgPool1d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBS(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck1D(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True, e=0.5):
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1 = CBS(in_ch, hidden, k=1)
        self.cv2 = CBS(hidden, hidden, k=3)
        self.cv3 = CBS(hidden, out_ch, k=1)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.cv3(self.cv2(self.cv1(x)))
        if self.use_shortcut:
            return x + out
        return out


class C3k2_1D(nn.Module):
    def __init__(self, in_ch, out_ch, n=2, shortcut=True, stride=1):
        super().__init__()
        self.ds = CBS(in_ch, in_ch, k=3, s=stride) if stride > 1 else nn.Identity()
        hidden = out_ch // 2
        self.cv1 = CBS(in_ch, hidden, k=1)
        self.cv2 = CBS(in_ch, hidden, k=1)
        self.bottlenecks = nn.Sequential(*[Bottleneck1D(hidden, hidden, shortcut) for _ in range(n)])
        self.cv_out = CBS(2 * hidden, out_ch, k=1)

    def forward(self, x):
        x = self.ds(x)
        main = self.bottlenecks(self.cv1(x))
        skip = self.cv2(x)
        return self.cv_out(torch.cat([main, skip], dim=1))


class ELANNeck1D(nn.Module):
    def __init__(self, ch1=64, ch2=128, ch3=256, out_ch=256):
        super().__init__()
        self.lat1 = CBS(ch1, ch3, k=1)
        self.lat2 = CBS(ch2, ch3, k=1)
        self.fuse = C3k2_1D(3 * ch3, out_ch, n=1, stride=1)

    def forward(self, f1, f2, f3):
        target_len = f3.shape[-1]
        f1_up = F.adaptive_avg_pool1d(f1, target_len)
        f2_up = F.adaptive_avg_pool1d(f2, target_len)
        f1_up = self.lat1(f1_up)
        f2_up = self.lat2(f2_up)
        return self.fuse(torch.cat([f1_up, f2_up, f3], dim=1))


class PSA1D(nn.Module):
    """1D position-sensitive attention style block for deep temporal features."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(
            nn.Linear(ch, ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, ch),
        )

    def forward(self, x):
        # x: (B, C, T) -> attn over temporal tokens (B, T, C)
        q = x.permute(0, 2, 1)
        a, _ = self.attn(q, q, q)
        x = self.norm(q + a)
        x = x + self.ffn(x)
        return x.permute(0, 2, 1)


class YOLO26_1D_Regressor(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_outputs=2,
        base_ch=32,
        widths=(64, 128, 256),
        n_blocks=(1, 2, 2),
        dropout=0.5,
        use_attention=True,
        attn_heads=4,
    ):
        super().__init__()

        ch1, ch2, ch3 = widths

        self.stem = nn.Sequential(
            CBS(in_channels, base_ch, k=7, s=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = C3k2_1D(base_ch, ch1, n=n_blocks[0], stride=2)
        self.stage2 = C3k2_1D(ch1, ch2, n=n_blocks[1], stride=2)
        self.stage3 = C3k2_1D(ch2, ch3, n=n_blocks[2], stride=2)

        self.psa = PSA1D(ch3, num_heads=attn_heads) if use_attention else nn.Identity()

        self.neck = ELANNeck1D(ch1, ch2, ch3, out_ch=ch3)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(ch3, ch3 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ch3 // 2, num_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f3 = self.psa(f3)
        fused = self.neck(f1, f2, f3)
        pooled = self.pool(fused).squeeze(-1)
        return self.head(pooled)


class MuSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _ns_orthogonalise(G, steps=5):
        a, b, c = 3.4445, -4.7750, 2.0315
        orig_dtype = G.dtype
        X = G.reshape(G.shape[0], -1).float()
        X = X / (X.norm() + 1e-7)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
        if transposed:
            X = X.T
        return X.reshape(G.shape).to(orig_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data.clone()
                if wd != 0.0:
                    g.add_(p.data, alpha=wd)
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.data)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu)
                if p.ndim >= 2:
                    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    g = MuSGD._ns_orthogonalise(g, steps=ns) * scale
                p.data.add_(g, alpha=-lr)
        return loss


def get_model(in_channels=6, num_outputs=2, dropout=0.5, use_attention=True, attn_heads=4):
    return YOLO26_1D_Regressor(
        in_channels=in_channels,
        num_outputs=num_outputs,
        base_ch=32,
        widths=(64, 128, 256),
        n_blocks=(1, 2, 2),
        dropout=dropout,
        use_attention=use_attention,
        attn_heads=attn_heads,
    )


def reset_bn_stats(model, new_momentum=0.1):
    """
    Reset all BatchNorm1d running statistics.

    Called after loading RIDI pretrained checkpoint and before RoNIN fine-tuning
    so BN running stats can recalibrate to RoNIN distribution.
    """
    reset_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = new_momentum
            reset_count += 1
    print(f"[BN Reset] Reset {reset_count} BatchNorm1d layers. New momentum={new_momentum}")
    return model
