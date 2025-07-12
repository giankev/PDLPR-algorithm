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
            if x.dim() == 3:
                # (B, seq_len, d_model)
                return x + self.pe[:, :x.size(1), :]
    
            elif x.dim() == 4:
                # (B, C, H, W)  →  flatten HW
                B, C, H, W = x.shape
                x_flat = x.flatten(2).permute(0, 2, 1)       # (B, H*W, C)
                x_flat = x_flat + self.pe[:, :x_flat.size(1), :]
                x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
                return x
    
            else:
                raise ValueError("Input must be 3D or 4D tensor")


class AddAndNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels) # NOTE: e se facessimo Group Norm?

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H , W) -> (B, in_channels, H , W)
        x = x + residual
        # (B, in_channels, H , W) -> # (B, H , W, in_channels)
        x = x.permute(0, 2, 3, 1)
        # (B, H , W, in_channels) -> (B, H , W, in_channels)
        x = self.norm(x) 
        # (B, H , W, in_channels) -> (B, in_channels, H , W)
        x = x.permute(0, 3, 1, 2)
        return x

class EncoderUnit(nn.Module):
    def __init__(self, in_channels: int, d_embed: int, n_heads: int):
        super().__init__()
        # NOTE: [!] se metto padding = 1 (come nel paper) le dimensioni di x aumentano di 2 sia di height che di width 
        self.cnn1 = CNNBlock(in_channels=in_channels, out_channels=d_embed, kernel_size=1, stride=1, padding=0)
        self.attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        # NOTE: [!] se metto padding = 1 ((come nel paper)) le dimensioni di x aumentano di 2 sia di height che di width
        self.cnn2 = CNNBlock(in_channels=d_embed, out_channels=in_channels, kernel_size=1, stride=1, padding=0) 
        self.addnorm = AddAndNorm(in_channels)  

    def forward(self, x):
        # x: (B, in_channels, H, W)
        residual = x
        # (B, in_channels, H, W) -> (B, d_embed, H, W)
        x = self.cnn1(x)
        # Self-attention WITHOUT mask
        # (B, d_embed, H, W) -> (B, d_embed, H , W) 
        x = self.attention(x)
        #  (B, d_embed, H , W) -> (B, in_channels, H , W) 
        x = self.cnn2(x)
        #(B, in_channels, H , W) -> (B, in_channels, H , W)
        x = self.addnorm(x, residual)

        return x # (B, in_channels, H , W)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        x = self.pos_encoder(x)
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        for layer in self.layers:
            x = layer(x)
        return x # # (B, in_channels, H, W)
