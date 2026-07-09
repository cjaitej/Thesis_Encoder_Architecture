"""
New novel architectures for RoNIN seq2seq inertial odometry go in this file
from now on (previously everything accumulated in model_temporal.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) recurrent cell (Hasani et al., 2022,
    "Closed-form Continuous-time Neural Networks"), simplified for uniformly
    sampled input: RoNIN windows are fixed at 200 Hz, so the elapsed-time term
    of the original ODE closed-form solution collapses into a single learned,
    input-dependent sigmoid gate instead of an explicit dt input. Two feed-forward
    candidate states are blended by that gate, so the cell's effective memory
    decay is a function of the current sample rather than a fixed time constant
    (unlike an LSTM's fixed-shape forget gate).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        gate_in = input_size + hidden_size
        self.ff1 = nn.Linear(gate_in, hidden_size)
        self.ff2 = nn.Linear(gate_in, hidden_size)
        self.time_gate = nn.Linear(gate_in, hidden_size)

    def forward(self, x, h):
        z = torch.cat([x, h], dim=-1)
        f = torch.tanh(self.ff1(z))
        g = torch.tanh(self.ff2(z))
        tau = torch.sigmoid(self.time_gate(z))
        return f * tau + g * (1.0 - tau)


class LiquidRNN(nn.Module):
    """Stack of bidirectional CfC layers, unrolled over time with a plain loop
    (T=200 in this task, short enough that an unfused loop is not a bottleneck)."""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * 2
            self.fwd_cells.append(CfCCell(in_size, hidden_size))
            self.bwd_cells.append(CfCCell(in_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def _run_direction(self, cell, x, reverse):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        time_steps = reversed(range(T)) if reverse else range(T)
        outputs = [None] * T
        for t in time_steps:
            h = cell(x[:, t, :], h)
            outputs[t] = h
        return torch.stack(outputs, dim=1)  # [B, T, H]

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            fwd = self._run_direction(self.fwd_cells[i], out, reverse=False)
            bwd = self._run_direction(self.bwd_cells[i], out, reverse=True)
            out = torch.cat([fwd, bwd], dim=-1)  # [B, T, 2H]
            if i < self.num_layers - 1:
                out = self.dropout(out)
        return out


class ChannelGraphAttention(nn.Module):
    """
    Multi-head self-attention over the (small, fixed) channel-node axis,
    implemented with plain matmuls instead of nn.MultiheadAttention /
    scaled_dot_product_attention. See ChannelGraphEncoder for why: those
    route through fused CUDA kernels whose launch grid is sized off the
    batch dimension, and here "batch" is B*T (every timestep attends over
    its own 6-node graph independently), which routinely exceeds the 65535
    grid-dimension limit those kernels enforce. Plain batched matmul has no
    such limit.
    """

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [N, num_nodes, dim]"""
        N, num_nodes, dim = x.shape
        qkv = self.qkv(x).reshape(N, num_nodes, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                  # each [N, nhead, num_nodes, head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                           # [N, nhead, num_nodes, head_dim]
        out = out.transpose(1, 2).reshape(N, num_nodes, dim)
        return self.out_proj(out)


class ChannelGraphEncoderLayer(nn.Module):
    """Pre-norm attention + FFN block built on ChannelGraphAttention (the
    channel-graph analogue of nn.TransformerEncoderLayer)."""

    def __init__(self, dim, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ChannelGraphAttention(dim, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class ChannelGraphEncoder(nn.Module):
    """
    Treats the 6 IMU channels (ax, ay, az, gx, gy, gz) as nodes of a small
    fully-connected graph and learns their pairwise relations with
    self-attention, applied independently at every timestep (weight-shared
    across time, so cost is O(T), not O(T^2)).

    This targets a coupling that the other models in this repo do not
    represent directly: TCN/MBConv use depthwise (per-channel) convolutions
    that never mix channels, and TCNTransformer/MBConvTransformer/PMRNet only
    apply attention across time, after the channels have already been
    collapsed into a single feature vector by an ordinary (channel-mixing but
    fixed-weight) convolution. Gravity leaking across accelerometer axes as
    the phone/body reorients, and gyro/accel cross-talk during turns
    (centripetal and Coriolis terms), are both cross-channel effects at a
    given instant, not cross-time ones -- a per-timestep learned channel
    graph is a more direct inductive bias for them than a fixed 1x1 conv.
    """

    def __init__(self, input_channel, node_dim, num_graph_layers, kernel_size,
                 nhead=4, dropout=0.1):
        super().__init__()
        self.input_channel = input_channel
        self.node_dim = node_dim
        padding = kernel_size // 2

        # Depthwise: each channel keeps its own local temporal receptive field
        # before its node embedding enters the graph.
        self.stem = nn.Conv1d(input_channel, input_channel * node_dim, kernel_size,
                               padding=padding, groups=input_channel, bias=False)
        self.stem_bn = nn.BatchNorm1d(input_channel * node_dim)
        self.stem_act = nn.SiLU()

        # Not nn.TransformerEncoderLayer: this graph attention is applied
        # with "batch" = B*T (every timestep gets its own independent 6-node
        # attention), which for a normal training batch (e.g. 512 x 200) is
        # 102,400 -- past the 65535 CUDA grid-dimension limit that
        # scaled_dot_product_attention's fused kernels size off the batch
        # axis, causing "CUDA error: invalid configuration argument". Plain
        # batched matmul (ChannelGraphAttention below) has no such limit.
        self.graph_layers = nn.ModuleList([
            ChannelGraphEncoderLayer(node_dim, nhead, node_dim * 2, dropout=dropout)
            for _ in range(num_graph_layers)
        ])

    def forward(self, x):
        """x: [B, T, input_channel] -> [B, T, input_channel * node_dim]"""
        B, T, C = x.shape
        feat = self.stem_act(self.stem_bn(self.stem(x.transpose(1, 2))))   # [B, C*node_dim, T]
        feat = feat.view(B, C, self.node_dim, T).permute(0, 3, 1, 2)       # [B, T, C, node_dim]
        feat = feat.reshape(B * T, C, self.node_dim)                       # graph of C nodes per (B, T)
        for layer in self.graph_layers:
            feat = layer(feat)                                            # channel self-attention
        feat = feat.reshape(B, T, C * self.node_dim)
        return feat


class GraphLiquidNet(nn.Module):
    """
    GraphLiquidNet: channel-graph attention + liquid (CfC) temporal encoder
    for seq2seq inertial odometry.

    Two stages, each targeting a different axis of the input that the rest of
    this repo's models handle implicitly at best:

    1. ChannelGraphEncoder models cross-channel (cross-axis) structure at
       each instant via attention over the 6 IMU channels as graph nodes.
    2. LiquidRNN models the temporal axis with closed-form continuous-time
       (CfC) cells instead of dilated convolution (TCN/MBConv), fixed-length
       self-attention (TCNTransformer/MBConvTransformer/PMRNet), or a
       selective state-space scan (Mamba). A CfC cell's effective memory
       decay is input-dependent rather than fixed, which is a better match
       for IMU data alternating between near-stationary stretches and sharp
       transients (steps, turns) than a fixed receptive field.

    Both ingredients are original at least in inertial-odometry: prior work
    that models cross-channel IMU structure (iMoT's "Adaptive Spatial Sync",
    EqNIO's equivariant frame transform) does so with fixed/symmetry-derived
    or heavier encoder-decoder machinery rather than free-form per-timestep
    channel-graph attention, and no published inertial-odometry network uses
    CfC/liquid time-constant cells for the temporal encoder (KAN-based
    ResKACNNet/LKAN and Mamba-based MambaIO occupy adjacent but distinct
    territory: spline basis functions and selective-SSM scanning,
    respectively, not gated continuous-time recurrence).

    Input:  [B, T, input_channel]
    Output: [B, T, output_channel]
    """

    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        node_dim=48,
        num_graph_layers=2,
        graph_nhead=4,
        d_model=160,
        liquid_hidden=128,
        num_liquid_layers=2,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()

        self.channel_graph = ChannelGraphEncoder(
            input_channel, node_dim, num_graph_layers, kernel_size,
            nhead=graph_nhead, dropout=dropout,
        )
        self.graph_proj = nn.Linear(input_channel * node_dim, d_model)

        self.liquid = LiquidRNN(d_model, liquid_hidden, num_layers=num_liquid_layers, dropout=dropout)
        self.liquid_proj = nn.Linear(liquid_hidden * 2, d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, output_channel),
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def forward(self, x):
        """x: [B, T, input_channel]"""
        graph_feat = self.graph_proj(self.channel_graph(x))          # [B, T, d_model]
        temporal_feat = self.liquid_proj(self.liquid(graph_feat))    # [B, T, d_model]
        features = graph_feat + temporal_feat
        return self.head(features)


class LiquidConvBlock(nn.Module):
    """
    Parallel, convolutional analogue of CfCCell's gated blend. GraphLiquidNet's
    LiquidRNN unrolls that blend as a *recurrence* (an unfused Python loop over
    T=200, run twice per layer for the two directions -- its speed bottleneck).
    Here the same "two candidate branches blended by an input-dependent
    sigmoid gate" computation is produced by dilated depthwise convolutions
    instead: f, g, and tau are all local (dilated-receptive-field) functions
    of x, computed in one parallel conv call each, so the whole block has no
    sequential dependency across time. Dilation doubles every layer (as in
    TemporalConvNet) so a stack still reaches a long effective receptive
    field without ever looping.

    Trade-off vs. LiquidRNN: the time-gate now sees only a local (dilated)
    window instead of the network's full running state, so this gives up
    genuine unbounded-horizon continuous-time memory in exchange for full
    parallelism -- a real trade-off, not a strict improvement.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.f_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                 dilation=dilation, groups=channels, bias=False)
        self.g_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                 dilation=dilation, groups=channels, bias=False)
        self.tau_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                   dilation=dilation, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        """x: [B, C, T]"""
        f = torch.tanh(self.f_conv(x))
        g = torch.tanh(self.g_conv(x))
        tau = torch.sigmoid(self.tau_conv(x))
        gated = f * tau + g * (1.0 - tau)
        out = self.dropout(self.mix(gated))
        return self.act(x + out)


class LiquidConvStack(nn.Module):
    """Stack of LiquidConvBlocks with doubling dilation, mirroring TemporalConvNet."""

    def __init__(self, channels, num_layers, kernel_size, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            LiquidConvBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_receptive_field(self):
        return 1 + (self.kernel_size - 1) * (2 ** self.num_layers - 1)


class GraphLiquidConvNet(nn.Module):
    """
    Fast variant of GraphLiquidNet: identical ChannelGraphEncoder front end
    (per-timestep attention over the 6 IMU channels as graph nodes), but the
    temporal encoder is LiquidConvStack -- a fully parallel stack of dilated
    "liquid-gated" convolutions -- instead of LiquidRNN's sequential
    bidirectional CfC recurrence.

    Why this exists: LiquidRNN unrolls 200 timesteps x 2 directions x
    num_layers as unfused sequential ops, which is much slower per epoch than
    every other model in this repo (TCN/MBConv/PMR/Transformer are all fully
    parallel over time). GraphLiquidConvNet keeps the novel channel-graph
    front end (verified against current literature to have no prior art in
    inertial odometry) and the CfC-style gated-blend computation, but drops
    the recurrence so training throughput is back in line with the other
    models. See LiquidConvBlock's docstring for the memory-horizon trade-off
    this makes to get there.

    Input:  [B, T, input_channel]
    Output: [B, T, output_channel]
    """

    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        node_dim=64,
        num_graph_layers=2,
        graph_nhead=4,
        d_model=200,
        num_conv_layers=6,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()

        self.channel_graph = ChannelGraphEncoder(
            input_channel, node_dim, num_graph_layers, kernel_size,
            nhead=graph_nhead, dropout=dropout,
        )
        self.graph_proj = nn.Linear(input_channel * node_dim, d_model)

        self.temporal = LiquidConvStack(d_model, num_conv_layers, kernel_size, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, output_channel),
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def forward(self, x):
        """x: [B, T, input_channel]"""
        graph_feat = self.graph_proj(self.channel_graph(x))                    # [B, T, d_model]
        temporal_feat = self.temporal(graph_feat.transpose(1, 2)).transpose(1, 2)  # [B, T, d_model]
        features = graph_feat + temporal_feat
        return self.head(features)

    def get_receptive_field(self):
        return self.temporal.get_receptive_field()
