import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import logging
from logging_utils import setup_logging
import torch.nn.functional as F
setup_logging()
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out

        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out

        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(base*4, base*8)
        self.attn = AttentionBlock(dim=base*8)

        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)

        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        x4 = self.bottleneck(self.pool3(x3))
        x4 = self.attn(x4)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        out = self.out_conv(x)
        return torch.tanh(out)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_filters) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, bn=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            discriminator_block(512, 1024),
            nn.Conv2d(1024, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)
