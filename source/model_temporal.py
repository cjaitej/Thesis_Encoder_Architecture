import sys
import os.path as osp
import math

import torch
from torch.autograd import Variable
import torch.nn as nn


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
