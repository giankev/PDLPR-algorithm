import torch
import torch.nn as nn
from feature_extractor import IGFE, CNNBlock
from encoder import Encoder
from decoder import Decoder

# output Encoder -> (B, 512, 6, 18) is the input of CNN BLOCK3

# NOTE: CNN BLOCK3 ha stride=3, kernelsize (2,1), padding=1, out_dim=512
# (B, 512, 6, 18) -> (B, 512, 3, 7) 

# NOTE: CNN BLOCK4 ha stride=3, kernelsize 1, padding=(0,1), out_dim=512
# (B, 512, 1, 3)

class PDLPR(nn.Module):
    def __init__(self, d_embed: int=512, d_cross: int=18, units: int=3,
                 n_heads: int=8, height: int = 6, width: int=18, num_classes: int=68):
        super().__init__()  

        self.igfe = IGFE(in_channels=12, out_channels=d_embed)

        self.encoder = Encoder(
            in_channels=d_embed,
            height = height,
            width = width,
            out_channels=d_embed*2,
            enc_unit=units,
            n_heads=n_heads)

        # NOTE:  CNN BLOCK3 + CNN BLOCK4 danno un outout diverso da quello del paper
        self.cnn3 = CNNBlock(d_embed, d_embed, kernel_size=(2, 1), stride=(3, 1), padding=(1, 0))
        self.cnn4 = CNNBlock(d_embed, d_embed, kernel_size=(1, 2), stride=(1, 3), padding=(0,0))

        self.decoder = Decoder(
            height = height,
            width = width,
            d_embed=d_embed,
            d_cross=d_cross,
            dec_unit=units,
            n_heads=n_heads)
        
        self.classifier = nn.Linear(d_embed, num_classes)
    
    def forward(self, x):
        # (B, 3, 48, 144) -> (B, 512, 6, 18)
        x = self.igfe(x)
        # print("IGFE output:", x.shape)

        # (B, 512, 6, 18) -> (B, 512, 6, 18) 
        x = self.encoder(x)
        # print("Enc output:", x.shape)

        # (B, 512, 6, 18) -> (B, 512, 3, 6) 
        conv_out = self.cnn4(self.cnn3(x)) # (B, 512, 1, 3)

        # (B, 512, 6, 18) -> (B, 512, 6, 18)
        x = self.decoder(x, conv_out)
        # print("Dec output: ", x.shape)

        B, C, H, W = x.shape
        # (B, 512, 6, 18) -> (B, 6, 18, 512) -> (B, 108, 512) 
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        # (B, 108, 512)  ->  (B, 108, num_classes)
        logits = self.classifier(x)
        # print("Logits shape: ", logits.shape)
        return logits
