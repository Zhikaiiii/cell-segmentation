import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torchvision import models

# 初始化
def init_weights(m, std=0.02):
    # def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.normal_(m.weight.data, 0.0, std)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)
    # net.apply(init_func)

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
## 每一层的卷积和激活函数子模块
class ConvCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cell = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv_cell(x)
        return out

class ResNetCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cell = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            # nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential()
        if self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels)
            )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv_cell(x)
        x = self.downsample(x)
        out = self.relu(x + out)
        return out

## Contracting Path: Down-Sample
class Contracting(nn.Module):
    def __init__(self, in_channels, out_channels, model):
        super().__init__()
        if model == 'RUet':
            self.contracting = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                ResNetCell(in_channels, out_channels)
            )
        else:
            self.contracting = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                ConvCell(in_channels, out_channels)
            )

    def forward(self, x):
        return self.contracting(x)

## Expansive Path: Up-Sample
class Expansive(nn.Module):
    def __init__(self,in_channels, out_channels, model):
        super().__init__()
        # self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if model == 'RUet':
            self.cell_conv = ResNetCell(in_channels, out_channels)
        else:
            self.cell_conv = ConvCell(in_channels + out_channels, out_channels)

    # x1表示上一层的输入，x2是contracting层的输出
    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(x1, [1,0,1,0])
        x = torch.cat([x1,x2],dim=1)
        return self.cell_conv(x)

## U-Net网络结构
class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels, model):
        super(U_Net, self).__init__()
        self.encoder = models.resnet34(pretrained=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = model
        self.init_conv = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.down1 = Contracting(64, 128, self.model)
        self.down2 = Contracting(128, 256, self.model)
        self.down3 = Contracting(256, 512, self.model)
        self.down4 = Contracting(512, 1024, self.model)

        self.up1 = Expansive(512, 256, self.model)
        self.up2 = Expansive(256, 128, self.model)
        self.up3 = Expansive(128, 64, self.model)
        self.up4 = Expansive(64,  64, self.model)
        self.out_conv = nn.Conv2d(64, self.out_channels, kernel_size=1)
        # self.out_conv2 = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        # x = self.encoder(x)
        x = self.init_conv(x)
        x_1 = self.encoder.layer1(self.encoder.maxpool(x))
        x_2 = self.encoder.layer2(x_1)
        x_3 = self.encoder.layer3(x_2)
        x_4 = self.encoder.layer4(x_3)

        # x_1 = self.down1(x)
        # x_2 = self.down2(x_1)
        # x_3 = self.down3(x_2)
        # x_4 = self.down4(x_3)
        out = self.up1(x_4, x_3)
        out = self.up2(out, x_2)
        out = self.up3(out, x_1)
        out = self.up4(out, x)
        out = self.out_conv(out)
        # out2 = self.out_conv2(out)
        return out
