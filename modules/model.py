import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SPPF, C2f, Conv


class DensityModel(nn.Module):
    def __init__(self):
        super(DensityModel, self).__init__()

        d = 0.33
        r = 2
        self.p1 = Conv(3, 16, k=3, s=2, p=1)
        self.p2 = nn.Sequential(
            Conv(16, 32, k=3, s=2, p=1),
            C2f(32, 32, n = int(3*d), shortcut=True),
        )
        self.p3 = nn.Sequential(
            Conv(32, 64, k=3, s=2, p=1),
            C2f(64, 64, n = int(6*d), shortcut=True),
        )
        self.p4 = nn.Sequential(
            Conv(64, 128, k=3, s=2, p=1),
            C2f(128, 128, n = int(6*d), shortcut=True),
        )
        self.p5 = nn.Sequential(
            Conv(128, 128*r, k=3, s=2, p=1),
            C2f(128*r, 128*r, n = int(3*d), shortcut=True),
            SPPF(128*r, 128*r)
        )

        self.b12 = C2f(128*(r+1), 128, n = int(3*d), shortcut=False)
        self.b15 = C2f(192, 64, n = int(3*d), shortcut=False)
        self.b18 = C2f(96, 32, n = int(3*d), shortcut=False)
        self.map320 = Conv(64, 1, k=1, s=1, act=nn.ReLU())

        self.b21 = C2f(48, 16, n = int(3*d), shortcut=False)
        self.map640 = Conv(16, 1, k=1, s=1, act=nn.ReLU())

        self.b22 = Conv(16, 16, k=3, s=2, p=1)
        self.b24 = C2f(48, 64, n = int(3*d), shortcut=False)
        
        self.b25 = Conv(64, 64, k=3, s=2, p=1)
        self.b27 = C2f(128, 196, n = int(3*d), shortcut=False)
        self.map160 = Conv(196, 1, k=1, s=1, act=nn.ReLU())

        self.alignment = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x):
        # Down
        x1 = self.p1(x)
        x2 = self.p2(x1)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)

        # Up
        b10 = nn.Upsample(scale_factor=2, mode='nearest')(x5)
        b11 = torch.cat([x4, b10], dim = 1)
        b12 = self.b12(b11)
        b13 = nn.Upsample(scale_factor=2, mode='nearest')(b12)
        b14 = torch.cat([b13, x3], dim = 1)
        b15 = self.b15(b14)

        b16 = nn.Upsample(scale_factor=2, mode='nearest')(b15)
        b17 = torch.cat([b16, x2], dim = 1)
        b18 = self.b18(b17)
        # map320 = self.map320(b18)

        b19 = nn.Upsample(scale_factor=2, mode='nearest')(b18)
        b20 = torch.cat([b19, x1], dim = 1)
        b21 = self.b21(b20)
        map640 = self.map640(b21)

        # Down
        b22 = self.b22(b21)
        b23 = torch.cat([b22, b18], dim = 1)
        b24 = self.b24(b23)
        map320 = self.map320(b24)

        b25 = self.b25(b24)
        b26 = torch.cat([b25, b15], dim = 1)
        b27 = self.b27(b26)
        map160 = self.map160(b27)

        heatmap_160_resized = F.interpolate(map160, size=(640, 640), mode='bilinear', align_corners=False)
        heatmap_320_resized = F.interpolate(map320, size=(640, 640), mode='bilinear', align_corners=False)
        combine_heatmap = torch.concat([map640, heatmap_320_resized, heatmap_160_resized], dim = 1)
        refine_heatmap = self.alignment(combine_heatmap)
        
        return refine_heatmap

if __name__ == '__main__':
    model = DensityModel().cuda()
    ckpt = torch.load('ckpt/02012025.pth')
    model.load_state_dict(state_dict=ckpt)
    
    x = torch.rand((1, 3, 1280, 1280)).cuda()

    y = model(x)
