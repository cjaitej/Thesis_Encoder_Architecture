"""
source/model_yolo26_1d.py

YOLOv26-1D window->velocity regressor for RoNIN (input (B, 6, 200) -> (B, 2)),
plus a parameter-efficient variant (YOLO26_1D_Efficient).

Why an efficient variant: a per-module breakdown of the original 1,174,594-param
model shows it is top-heavy -- the temporal self-attention block (PSA1D) is 44.8%
of the parameters and the multi-scale neck (ELANNeck1D) another 29.2%, i.e. 74%
of the model sits in just two modules, while the whole convolutional feature
backbone (stem + 3 stages) is only ~24%. Crucially, PSA runs a full 256-d
MultiheadAttention + 2x FFN over only ~7 temporal tokens (200 -> /4 stem -> /8
stages), which is heavily over-parameterised.

The efficient variant replaces those two modules:
  * EfficientPSA1D : a local depthwise conv (cheap local mixing) + a REDUCED-
    dimension attention (project 256 -> attn_dim, attend, project back) + a slim
    FFN. Keeps local+global temporal mixing at a fraction of the params.
  * LiteNeck1D     : projects all three scales to a small fuse dim before fusion
    (instead of concatenating three full 256-d maps), and fuses to a smaller
    output width.
This roughly halves the parameter count (~1.17M -> ~0.6M) and, because the cut
capacity was the over-parameterised part, is expected to REGULARISE (the models
in this repo overfit) rather than hurt. The original model is kept in the same
file so the efficient variant can be ablated against it under identical training.

All operations are 1D (Conv1d, BatchNorm1d, AdaptiveAvgPool1d) + a small
reduced-dim attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks (original YOLOv26-1D)
# ---------------------------------------------------------------------------
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
    """Original neck: lateral 1x1s to 256 then fuse three full-width maps."""

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
    """Original attention block: full-dim MultiheadAttention + 2x FFN."""

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
        q = x.permute(0, 2, 1)
        a, _ = self.attn(q, q, q)
        x = self.norm(q + a)
        x = x + self.ffn(x)
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Efficient replacements
# ---------------------------------------------------------------------------
class EfficientPSA1D(nn.Module):
    """
    Parameter-efficient replacement for PSA1D.

    Three cheap ingredients instead of one heavy attention block:
      1. depthwise local conv over time  -- local temporal mixing, ~ch*k params
      2. REDUCED-dimension attention      -- project ch -> attn_dim (<< ch),
         attend, project back; the over-7-token attention no longer needs the
         full 256-d projections
      3. a slim FFN (ffn_ratio < 1)       -- the original 2x FFN was the single
         biggest sub-cost

    ch=256, attn_dim=96, ffn_ratio=0.5 -> ~154K params vs PSA1D's ~527K.
    """

    def __init__(self, ch, num_heads=4, attn_dim=96, ffn_ratio=0.5, dropout=0.0):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.local = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)

        self.norm1 = nn.LayerNorm(ch)
        self.down = nn.Linear(ch, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True, dropout=dropout)
        self.up = nn.Linear(attn_dim, ch)

        self.norm2 = nn.LayerNorm(ch)
        hidden = max(1, int(ch * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, ch),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x + self.local(x)                       # local temporal mixing
        q = x.permute(0, 2, 1)                       # (B, T, C)

        z = self.down(self.norm1(q))                 # -> (B, T, attn_dim)
        a, _ = self.attn(z, z, z)
        q = q + self.up(a)                           # global temporal mixing

        q = q + self.ffn(self.norm2(q))
        return q.permute(0, 2, 1)


class LiteNeck1D(nn.Module):
    """
    Parameter-efficient replacement for ELANNeck1D.

    The original concatenates three full 256-d maps (768 channels) before a
    C3k2 fuse -- expensive. Here every scale is first projected to a small
    fuse_dim, and the fused output width (out_ch) is reduced too.

    fuse_dim=96, out_ch=192 -> ~152K params vs ELANNeck1D's ~343K.
    """

    def __init__(self, ch1=64, ch2=128, ch3=256, out_ch=192, fuse_dim=96):
        super().__init__()
        self.out_ch = out_ch
        self.lat1 = CBS(ch1, fuse_dim, k=1)
        self.lat2 = CBS(ch2, fuse_dim, k=1)
        self.lat3 = CBS(ch3, fuse_dim, k=1)
        self.fuse = C3k2_1D(3 * fuse_dim, out_ch, n=1, stride=1)

    def forward(self, f1, f2, f3):
        target_len = f3.shape[-1]
        f1_up = self.lat1(F.adaptive_avg_pool1d(f1, target_len))
        f2_up = self.lat2(F.adaptive_avg_pool1d(f2, target_len))
        f3_p = self.lat3(f3)
        return self.fuse(torch.cat([f1_up, f2_up, f3_p], dim=1))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class YOLO26_1D_Regressor(nn.Module):
    """Original YOLOv26-1D (1,174,594 params). Kept as the ablation baseline."""

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


class YOLO26_1D_Efficient(nn.Module):
    """
    Parameter-efficient YOLOv26-1D: same convolutional backbone, but the two
    bloated modules (attention + neck) are replaced by EfficientPSA1D and
    LiteNeck1D. ~0.6M params (about half the original) at the same input/output.

    Input:  (B, in_channels, T)
    Output: (B, num_outputs)
    """

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
        attn_dim=96,
        ffn_ratio=0.5,
        neck_fuse_dim=96,
        neck_out=192,
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

        self.psa = (
            EfficientPSA1D(ch3, num_heads=attn_heads, attn_dim=attn_dim, ffn_ratio=ffn_ratio)
            if use_attention else nn.Identity()
        )
        self.neck = LiteNeck1D(ch1, ch2, ch3, out_ch=neck_out, fuse_dim=neck_fuse_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(neck_out, neck_out // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(neck_out // 2, num_outputs),
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


# ---------------------------------------------------------------------------
# Optimizer + helpers (unchanged)
# ---------------------------------------------------------------------------
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


def get_efficient_model(in_channels=6, num_outputs=2, dropout=0.5, use_attention=True, attn_heads=4):
    return YOLO26_1D_Efficient(
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
    """Reset all BatchNorm1d running statistics (for RIDI->RoNIN fine-tuning)."""
    reset_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = new_momentum
            reset_count += 1
    print(f"[BN Reset] Reset {reset_count} BatchNorm1d layers. New momentum={new_momentum}")
    return model


if __name__ == "__main__":
    def breakdown(model, name):
        total = model.get_num_params()
        print(f"\n=== {name}: {total:,} params ===")
        for cname, child in model.named_children():
            n = sum(p.numel() for p in child.parameters())
            if n:
                print(f"  {cname:8s} {n:>9,}  ({100 * n / total:4.1f}%)")
        x = torch.randn(2, 6, 200)
        y = model(x)
        assert y.shape == (2, 2), y.shape
        print(f"  forward OK: (2,6,200) -> {tuple(y.shape)}")
        return total

    orig = breakdown(YOLO26_1D_Regressor(), "ORIGINAL YOLO26_1D")
    eff = breakdown(YOLO26_1D_Efficient(), "EFFICIENT YOLO26_1D")
    print(f"\nreduction: {orig:,} -> {eff:,}  "
          f"({100 * (orig - eff) / orig:.1f}% fewer params, {orig / eff:.2f}x smaller)")
