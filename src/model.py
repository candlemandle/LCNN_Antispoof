import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    """Max-Feature-Map: split channels in two and take elementwise max."""
    def __init__(self, in_channels):
        super().__init__()
        # in_channels must be 2*C
        self.out_channels = in_channels // 2

    def forward(self, x):
        # x: (B, 2*C, H, W) → (B, C, H, W)
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)

def conv_mfm(in_c, out_c, kernel_size, pool=True):
    layers = [
        nn.Conv2d(in_c, out_c*2, kernel_size=kernel_size,
                  stride=1, padding=kernel_size//2),
        MFM(out_c*2),
    ]
    if pool:
        layers += [ nn.MaxPool2d(2,2) ]
    layers += [ nn.Dropout(0.5) ]
    return nn.Sequential(*layers)

class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 conv+MFM+pool+dropout blocks
        self.layer1 = conv_mfm(1,   16, 5, pool=True)
        self.layer2 = conv_mfm(16,  32, 3, pool=True)
        self.layer3 = conv_mfm(32,  64, 3, pool=True)
        self.layer4 = conv_mfm(64, 128, 3, pool=True)
        self.layer5 = conv_mfm(128,256, 3, pool=True)

        # collapse H×W→1×1
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # two-stage FC: (256 → 128) → (128 → 2)
        # use MFM in the first layer
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128*2),
            MFM(128*2),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x: (B,1,freq_bins,frames)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)              # (B,256,1,1)
        x = x.view(x.size(0), -1)    # (B,256)
        x = self.fc1(x)              # (B,128)
        x = self.fc2(x)              # (B,2)
        return x
