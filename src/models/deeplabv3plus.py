"""DeepLabV3+ for concrete crack segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with rates [6, 12, 18]."""
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [self.conv_1x1(x)]
        for atrous in self.atrous:
            feats.append(atrous(x))
        gp = self.global_pool(x)
        feats.append(F.interpolate(gp, size=size, mode='bilinear', align_corners=True))
        return self.out_conv(torch.cat(feats, dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, 256)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.conv1(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        aspp_out = self.aspp(x)
        low_level = self.low_level_conv(low_level)
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([aspp_up, low_level], dim=1)
        x = self.decoder(x)
        x = self.final_conv(x)
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)



if __name__ == '__main__':
    model = DeepLabV3Plus(num_classes=1)
    model.eval()
    x = torch.randn(1, 3, 384, 544)
    with torch.no_grad():
        out = model(x)
    print(f'Input: {x.shape} -> Output: {out.shape}')
    assert out.shape == (1, 1, 384, 544), f'Shape mismatch: {out.shape}'
    print('Shape check PASSED')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params/1e6:.2f}M')
