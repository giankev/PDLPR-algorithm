import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import CNNBlock
from attention import SelfAttention
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.pe = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pe, std=0.02)  # inizializzazione

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class EncoderUnit(nn.Module):
    def __init__(self, in_channels: int, d_embed: int, n_heads: int):
        super().__init__()
        # NOTE: [!] se metto padding = 1 le dimensioni di x aumentano di 2 sia di height che di width
        self.cnn1 = CNNBlock(in_channels=in_channels, out_channels=d_embed, kernel_size=1, stride=1, padding=0)
        self.attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        # NOTE: [!] se metto padding = 1 le dimensioni di x aumentano
        self.cnn2 = CNNBlock(in_channels=d_embed, out_channels=in_channels, kernel_size=1, stride=1, padding=0) 
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        residual = x
        # (B, in_channels, H, W) -> (B, d_embed, H, W)
        x = self.cnn1(x)
        b, c, h, w = x.shape

        # (B, d_embed, H, W) -> (B, d_embed, H * W)
        x = x.view((b, c, h * w))
        # (B, d_embed, H * W) -> (B, H * W, d_embed)
        x = x.transpose(-1, -2)
        # Self-attention WITHOUT mask
        # (B, H * W, d_embed) -> (B, H * W, d_embed)
        x = self.attention(x)
        # (B, H * W, d_embed) ->  (B, C, H * d_embed) 
        x = x.transpose(-1, -2)
        # (B, d_embed, H * W)  -> (B, d_embed, H , W) 
        x = x.view((b, c, h, w))
        #  (B, d_embed, H , W) -> (B, in_channels, H , W) 
        x = self.cnn2(x)
        # (B, in_channels, H , W) -> (B, in_channels, H , W)
        x = x + residual
        # (B, in_channels, H , W) -> # (B, H , W, in_channels)
        x = x.permute(0, 2, 3, 1)
        # (B, H , W, in_channels) -> (B, H , W, in_channels)
        x = self.norm(x) 
        # (B, H , W, in_channels) -> (B, in_channels, H , W)
        x = x.permute(0, 3, 1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=512,
                 height = 6,
                 width = 18,
                 out_channels=1024,
                 enc_unit=3,
                 n_heads=8):
        super().__init__()

        self.height = height
        self.width = width
        self.seq_len = height * width
        self.d_model = in_channels

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              seq_len=self.seq_len)
        self.layers = nn.ModuleList([
            EncoderUnit(in_channels, out_channels, n_heads)
            for _ in range(enc_unit)
        ])

    def forward(self, x):
        # x: (B, in_channels, H, W)
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        # (B, H*W, C)
        x = self.pos_encoder(x)
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        for layer in self.layers:
            x = layer(x)
        return x
