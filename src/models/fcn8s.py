"""FCN-8s for concrete crack segmentation.

VGG16 backbone with FCN-8s skip architecture from:
  Long, Shelhamer, Darrell. "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.

pool3 (256ch, stride 8), pool4 (512ch, stride 16), conv7 (512ch, stride 32)
2-ch intermediate score layers for skip fusion, 1-ch final output (raw logits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FCN8s(nn.Module):
    """FCN-8s with VGG16 backbone.

    Args:
        num_classes: Output channels (default 1 for raw logits).
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # Load pretrained VGG16 feature extractor
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # Slice into three stages at pool3, pool4, conv7 boundaries
        self.pool3_net = nn.Sequential(*list(vgg.children())[:17])   # indices 0-16
        self.pool4_net = nn.Sequential(*list(vgg.children())[17:24]) # indices 17-23
        self.conv7_net = nn.Sequential(*list(vgg.children())[24:])   # indices 24-30

        # Score layers (1x1 convs) — intermediate 2 channels for skip fusion
        self.score_pool3 = nn.Conv2d(256, 2, 1)
        self.score_pool4 = nn.Conv2d(512, 2, 1)
        self.score_conv7 = nn.Conv2d(512, 2, 1)

        # Final score layer: 2ch -> num_classes (raw logits)
        self.final_conv = nn.Conv2d(2, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Forward through VGG16 stages, capturing skip features
        pool3 = self.pool3_net(x)           # (B, 256, H/8,  W/8)
        pool4 = self.pool4_net(pool3)       # (B, 512, H/16, W/16)
        conv7 = self.conv7_net(pool4)       # (B, 512, H/32, W/32)

        # Score each stage
        s3 = self.score_pool3(pool3)        # (B, 2, H/8,  W/8)
        s4 = self.score_pool4(pool4)        # (B, 2, H/16, W/16)
        s7 = self.score_conv7(conv7)        # (B, 2, H/32, W/32)

        # FCN-8s skip fusion: upsample 2x and add
        up_s7 = F.interpolate(s7, scale_factor=2, mode='bilinear', align_corners=True)
        fuse1 = up_s7 + s4                  # (B, 2, H/16, W/16)

        up_fuse1 = F.interpolate(fuse1, scale_factor=2, mode='bilinear', align_corners=True)
        fuse2 = up_fuse1 + s3               # (B, 2, H/8, W/8)

        # Final 8x upsample to input resolution, then 1x1 -> raw logits
        up_final = F.interpolate(fuse2, size=input_size, mode='bilinear', align_corners=True)
        return self.final_conv(up_final)


if __name__ == "__main__":
    model = FCN8s(num_classes=1)
    x = torch.randn(1, 3, 384, 544)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (1, 1, 384, 544), f"Shape mismatch: {out.shape}"
    print("Shape check PASSED")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
