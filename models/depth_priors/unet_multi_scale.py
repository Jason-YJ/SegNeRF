# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet_multi_scale(nn.Module):
    @staticmethod
    def weight_init(m):
        # 1. 根据网络层的不同定义不同的初始化方式
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, n_channels, n_classes, mode='maxpooling'):
        super(UNet_multi_scale, self).__init__()
        self.inc = inconv(n_channels, 4)
        self.down1 = down_multi_scale(4, 8, 2)
        self.down2 = down_multi_scale(8, 16, 4)
        self.down3 = down_multi_scale(16, 32, 8)
        self.down4 = down_multi_scale(32, 32, 16)

        self.up1 = up(64, 16)
        self.up2 = up(32, 8)
        self.up3 = up(16, 4)
        self.up4 = up(8, 4)
        self.outc = outconv(4, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1, x1)
        x3 = self.down2(x2, x1)
        x4 = self.down3(x3, x1)
        x5 = self.down4(x4, x1)

        dx1 = self.up1(x5, x4)
        dx2 = self.up2(dx1, x3)
        dx3 = self.up3(dx2, x2)
        dx4 = self.up4(dx3, x1)
        dx5 = self.outc(dx4)
        return dx5


