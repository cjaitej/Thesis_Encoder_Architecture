import sys
import os.path as osp
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


from tcn import TemporalConvNet



class LSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0):
        """
        Simple LSTM network
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(LSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(self.input_size, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size*5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input, hidden=None):
        output, self.hidden = self.lstm(input, self.init_weights())
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


class BilinearLSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0):
        """
        LSTM network with Bilinear layer
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(BilinearLSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.bilinear = torch.nn.Bilinear(self.input_size, self.input_size, self.input_size * 4)
        self.lstm = torch.nn.LSTM(self.input_size * 5, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size + self.input_size * 5, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size * 5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input):
        input_mix = self.bilinear(input, input)
        input_mix = torch.cat([input, input_mix], dim=2)
        output, self.hidden = self.lstm(input_mix, self.init_weights())
        output = torch.cat([input_mix, output], dim=2)
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


class TCNSeqNetwork(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, layer_channels, dropout=0.2):
        """
        Temporal Convolution Network with PReLU activations
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param kernel_size: size of convolution kernel (must be odd)
        :param layer_channels: array specifying num. of channels in each layer
        :param dropout: dropout probability
        """

        super(TCNSeqNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.num_layers = len(layer_channels)

        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.output_layer = torch.nn.Conv1d(layer_channels[-1], output_channel, 1)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.net = torch.nn.Sequential(self.tcn, self.output_dropout, self.output_layer)
        self.init_weights()

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.net(out)
        return out.transpose(1, 2)

    def init_weights(self):
        self.output_layer.weight.data.normal_(0, 0.01)
        self.output_layer.bias.data.normal_(0, 0.001)

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)



class TCNTransformerNetwork(nn.Module):
    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        tcn_channels=(64, 128),
        kernel_size=3,
        d_model=128,
        nhead=8,
        num_transformer_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        assert tcn_channels[-1] == d_model
        assert d_model % nhead == 0

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.num_layers = len(tcn_channels)

        # -----------------------------
        # TCN Backbone
        # -----------------------------
        self.tcn = TemporalConvNet(
            input_channel,
            list(tcn_channels),
            kernel_size,
            dropout
        )

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # -----------------------------
        # Prediction Head
        # -----------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, output_channel)
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def _positional_encoding(self, seq_len, device, dtype):
        """Sinusoidal positional encoding, computed for the actual sequence length (no fixed length cap)."""
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(dtype)

    def forward(self, x):
        """
        x: [B, T, 6]
        """

        B, T, _ = x.shape

        # TCN expects [B, C, T]
        tcn_features = self.tcn(x.transpose(1, 2))

        # [B, T, d_model]
        tcn_features = tcn_features.transpose(1, 2)

        # positional encoding
        x = tcn_features + self._positional_encoding(T, x.device, tcn_features.dtype)

        # transformer
        transformer_features = self.transformer(x)

        # residual fusion
        features = tcn_features + transformer_features

        # velocity prediction
        out = self.head(features)

        return out

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation channel gating over the temporal axis."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv1d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class InvertedResidualBlock1D(nn.Module):
    """
    MobileNetV2-style inverted residual block adapted to dilated 1D sequences:
    1x1 expand -> dilated depthwise conv -> SE gate -> 1x1 project (linear bottleneck),
    with a projection shortcut when channel counts change (mirrors TemporalBlock/BasicBlock1D
    elsewhere in this file). Padding is symmetric ("same"), not causal: the network is always
    followed by bidirectional self-attention, so nothing is gained by restricting the conv to the past.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, expand_ratio=4,
                 se_reduction=4, dropout=0.1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        padding = (kernel_size - 1) * dilation // 2

        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
            )
        else:
            hidden_dim = in_channels
            self.expand = nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.se = SEBlock1D(hidden_dim, reduction=se_reduction)
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        out = self.dropout(out)
        residual = x if self.shortcut is None else self.shortcut(x)
        return out + residual


class InvertedResidualTCN1D(nn.Module):
    """Stack of dilated MBConv1D blocks; dilation doubles every stage like TemporalConvNet."""

    def __init__(self, input_channel, channels, kernel_size, expand_ratio=4, se_reduction=4, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(channels):
            in_channels = input_channel if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(InvertedResidualBlock1D(in_channels, out_channels, kernel_size, dilation,
                                                   expand_ratio=expand_ratio, se_reduction=se_reduction,
                                                   dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MBConvTransformerNetwork(nn.Module):
    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        mbconv_channels=(32, 64, 96, 128),
        kernel_size=3,
        expand_ratio=4,
        se_reduction=4,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        assert mbconv_channels[-1] == d_model
        assert d_model % nhead == 0

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.num_layers = len(mbconv_channels)

        # -----------------------------
        # MBConv (inverted-residual + SE) Backbone
        # -----------------------------
        self.backbone = InvertedResidualTCN1D(
            input_channel,
            list(mbconv_channels),
            kernel_size,
            expand_ratio=expand_ratio,
            se_reduction=se_reduction,
            dropout=dropout,
        )

        # -----------------------------
        # Lightweight Transformer Encoder
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # -----------------------------
        # Prediction Head
        # -----------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, output_channel)
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def _positional_encoding(self, seq_len, device, dtype):
        """Sinusoidal positional encoding, computed for the actual sequence length (no fixed length cap)."""
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(dtype)

    def forward(self, x):
        """
        x: [B, T, 6]
        """

        B, T, _ = x.shape

        # backbone expects [B, C, T]
        backbone_features = self.backbone(x.transpose(1, 2))

        # [B, T, d_model]
        backbone_features = backbone_features.transpose(1, 2)

        # positional encoding
        x = backbone_features + self._positional_encoding(T, x.device, backbone_features.dtype)

        # transformer
        transformer_features = self.transformer(x)

        # residual fusion
        features = backbone_features + transformer_features

        # velocity prediction
        out = self.head(features)

        return out

    def get_receptive_field(self):
        # Each MBConv block contributes one dilated depthwise conv (vs. two dilated convs
        # per TemporalBlock in TemporalConvNet), so the growth factor is not doubled here.
        return 1 + (self.kernel_size - 1) * (2 ** self.num_layers - 1)


def _sinusoidal_pe(seq_len, d_model, device, dtype):
    """Sinusoidal positional encoding [1, seq_len, d_model] for the actual sequence length."""
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0).to(dtype)


class MultiResFusion1D(nn.Module):
    """
    HRNet-style exchange unit for 1D streams: every stream receives the sum of all
    other streams, resampled to its own temporal resolution.
    Downsampling: average pool (ratio) + 1x1 conv + BN. Upsampling: linear
    interpolation to the exact target length + 1x1 conv + BN, so streams of any
    length (odd windows included) stay aligned.
    """

    def __init__(self, channels):
        super().__init__()
        self.num_streams = len(channels)
        self.projections = nn.ModuleList()
        for i in range(self.num_streams):          # target stream
            row = nn.ModuleList()
            for j in range(self.num_streams):      # source stream
                if i == j:
                    row.append(nn.Identity())
                else:
                    row.append(nn.Sequential(
                        nn.Conv1d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm1d(channels[i]),
                    ))
            self.projections.append(row)
        self.act = nn.SiLU()

    def forward(self, streams):
        out = []
        for i in range(self.num_streams):
            target_len = streams[i].shape[-1]
            fused = streams[i]
            for j in range(self.num_streams):
                if i == j:
                    continue
                x = streams[j]
                if x.shape[-1] > target_len:                       # downsample
                    ratio = max(1, x.shape[-1] // target_len)
                    x = F.avg_pool1d(x, kernel_size=ratio, stride=ratio)
                if x.shape[-1] != target_len:                      # exact-length align / upsample
                    x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
                fused = fused + self.projections[i][j](x)
            out.append(self.act(fused))
        return out


class PMRNet(nn.Module):
    """
    PMR-Net: Parallel Multi-Resolution network with coarse-scale attention for
    seq2seq inertial odometry.

    Four parallel streams run at temporal resolutions T, T/2, T/4, T/8 and exchange
    information after every stage (HRNet-style repeated bidirectional fusion, here
    applied to 1D IMU sequences). Stream blocks are dilation-free MBConv+SE units
    (InvertedResidualBlock1D). Global self-attention is applied only on the coarsest
    (T/8) stream, where it is ~64x cheaper than at full rate, then propagated to the
    finer streams by the following fusion. The full-rate stream is preserved
    end-to-end, matching the per-frame velocity output of the seq2seq task.

    Input:  [B, T, input_channel]
    Output: [B, T, output_channel]
    """

    def __init__(
        self,
        input_channel=6,
        output_channel=2,
        stream_channels=(24, 48, 72, 96),
        num_stages=3,
        kernel_size=5,
        expand_ratio=2,
        se_reduction=4,
        nhead=4,
        dropout=0.1,
    ):
        super().__init__()

        self.stream_channels = list(stream_channels)
        self.num_streams = len(stream_channels)
        self.coarse_dim = stream_channels[-1]
        assert self.coarse_dim % nhead == 0

        # Full-rate stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channel, stream_channels[0], kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(stream_channels[0]),
            nn.SiLU(),
        )

        # Stream initializers: full-rate stem features -> pooled + projected per stream
        self.stream_init = nn.ModuleList()
        for i in range(1, self.num_streams):
            self.stream_init.append(nn.Sequential(
                nn.Conv1d(stream_channels[0], stream_channels[i], 1, bias=False),
                nn.BatchNorm1d(stream_channels[i]),
                nn.SiLU(),
            ))

        # Stages: per-stream MBConv+SE block, then all-to-all fusion
        self.stage_blocks = nn.ModuleList()
        self.stage_fusions = nn.ModuleList()
        for _ in range(num_stages):
            self.stage_blocks.append(nn.ModuleList([
                InvertedResidualBlock1D(c, c, kernel_size, dilation=1,
                                        expand_ratio=expand_ratio,
                                        se_reduction=se_reduction, dropout=dropout)
                for c in stream_channels
            ]))
            self.stage_fusions.append(MultiResFusion1D(stream_channels))

        # Coarse-scale attention: one encoder layer after each stage except the first
        coarse_layer = lambda: nn.TransformerEncoderLayer(
            d_model=self.coarse_dim,
            nhead=nhead,
            dim_feedforward=self.coarse_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.coarse_attention = nn.ModuleList([coarse_layer() for _ in range(max(1, num_stages - 1))])

        # Per-frame prediction head over concatenated full-rate features
        fuse_dim = sum(stream_channels)
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, output_channel)
        )
        self.init_weights()

    def init_weights(self):
        final_layer = self.head[-1]
        final_layer.weight.data.normal_(0, 0.01)
        final_layer.bias.data.normal_(0, 0.001)

    def forward(self, x):
        """
        x: [B, T, input_channel]
        """
        B, T, _ = x.shape

        full = self.stem(x.transpose(1, 2))                    # [B, C0, T]

        streams = [full]
        for i in range(1, self.num_streams):
            ratio = 2 ** i
            pooled = F.avg_pool1d(full, kernel_size=ratio, stride=ratio, ceil_mode=True)
            streams.append(self.stream_init[i - 1](pooled))

        attn_idx = 0
        for stage, (blocks, fusion) in enumerate(zip(self.stage_blocks, self.stage_fusions)):
            streams = [block(s) for block, s in zip(blocks, streams)]

            if stage > 0 and attn_idx < len(self.coarse_attention):
                coarse = streams[-1].transpose(1, 2)           # [B, T/8, C]
                coarse = coarse + _sinusoidal_pe(coarse.shape[1], self.coarse_dim,
                                                 coarse.device, coarse.dtype)
                coarse = self.coarse_attention[attn_idx](coarse)
                streams[-1] = coarse.transpose(1, 2)
                attn_idx += 1

            streams = fusion(streams)

        # Upsample every stream to full rate and concatenate
        features = [streams[0]]
        for s in streams[1:]:
            features.append(F.interpolate(s, size=T, mode='linear', align_corners=False))
        features = torch.cat(features, dim=1).transpose(1, 2)  # [B, T, sum(C)]

        return self.head(features)
