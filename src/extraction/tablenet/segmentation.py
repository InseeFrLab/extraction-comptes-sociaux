"""
Segmentation classes.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LargeColumnDecoder(nn.Module):
    """
    Column Decoder.
    """

    def __init__(self, num_classes: int):
        """
        Initialize Column Decoder.

        Args:
            num_classes (int): Number of classes per point.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer = nn.ConvTranspose2d(
            64, num_classes, kernel_size=2, stride=2, dilation=1
        )

    def forward(self, x, pools):
        """
        Forward pass.

        Args:
            x (tensor): Batch of images to perform forward-pass.
            pools (Tuple[tensor, tensor]): The 3 and 4 pooling layer
                from VGG-19.

        Returns (tensor): Forward-pass result tensor.

        """
        pool_3, pool_4 = pools
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_4], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_1(x)
        x = torch.cat([x, pool_3], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_2(x)
        return self.layer(x)
