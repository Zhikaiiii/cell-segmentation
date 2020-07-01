import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from UNet import ConvCell, ResNetCell
from torchvision import models

# https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels, out_channels, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.input_channels = input_channels
        self.out_channels = out_channels

        self.max_pool = nn.MaxPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.padding = nn.ReplicationPad2d([1,0,1,0])
        self.channels = [64, 64, 128, 256, 512]
        # self.channels = [64, 128, 256, 512, 1024]
        self.encoder = models.resnet34(pretrained=True)
        # encoder_dict = self.encoder.state_dict()
        self.conv_00 = ConvCell(self.input_channels,self.channels[0])
        self.conv_10 = ConvCell(self.channels[0], self.channels[1])
        self.conv_20 = ConvCell(self.channels[1], self.channels[2])
        self.conv_30 = ConvCell(self.channels[2], self.channels[3])
        self.conv_40 = ConvCell(self.channels[3], self.channels[4])

        self.conv_01 = ConvCell(self.channels[1] + self.channels[0], self.channels[0])
        self.conv_11 = ConvCell(self.channels[2] + self.channels[1], self.channels[1])
        self.conv_21 = ConvCell(self.channels[3] + self.channels[2], self.channels[2])
        self.conv_31 = ConvCell(self.channels[4] + self.channels[3], self.channels[3])

        self.conv_02 = ConvCell(self.channels[1] + self.channels[0]*2, self.channels[0])
        self.conv_12 = ConvCell(self.channels[2] + self.channels[1]*2, self.channels[1])
        self.conv_22 = ConvCell(self.channels[3] + self.channels[2]*2, self.channels[2])

        self.conv_03 = ConvCell(self.channels[1] + self.channels[0]*3, self.channels[0])
        self.conv_13 = ConvCell(self.channels[2] + self.channels[1]*3, self.channels[1])

        self.conv_04 = ConvCell(self.channels[1] + self.channels[0]*4, self.channels[0])
        if self.deep_supervision:
            self.out_conv1 = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)
            self.out_conv2 = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)
            self.out_conv3 = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)
            self.out_conv4 = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)

    def forward(self, x):

        x_00 = self.conv_00(x)
        x_10 = self.encoder.layer1(self.encoder.maxpool(x_00))
        x_20 = self.encoder.layer2(x_10)
        x_30 = self.encoder.layer3(x_20)
        x_40 = self.encoder.layer4(x_30)

        # x_10 = self.conv_10(self.max_pool(x_00))
        x_01 = self.conv_01(torch.cat([x_00, self.padding_ornot(x_00, x_10)], dim=1))

        # x_20 = self.conv_20(self.max_pool(x_10))
        x_11 = self.conv_11(torch.cat([x_10, self.padding_ornot(x_10, x_20)], dim=1))
        x_02 = self.conv_02(torch.cat([x_00, x_01, self.padding_ornot(x_01, x_11)], dim=1))

        # x_30 = self.conv_30(self.max_pool(x_20))
        x_21 = self.conv_21(torch.cat([x_20, self.padding_ornot(x_20, x_30)], dim=1))
        x_12 = self.conv_12(torch.cat([x_10, x_11, self.padding_ornot(x_11, x_21)], dim=1))
        x_03 = self.conv_03(torch.cat([x_00, x_01, x_02, self.padding_ornot(x_02, x_12)], dim=1))

        # x_40 = self.conv_40(self.max_pool(x_30))
        x_31 = self.conv_31(torch.cat([x_30, self.padding_ornot(x_30, x_40)], dim=1))
        x_22 = self.conv_22(torch.cat([x_20, x_21, self.padding_ornot(x_21, x_31)], dim=1))
        x_13 = self.conv_13(torch.cat([x_10, x_11, x_12, self.padding_ornot(x_12, x_22)], dim=1))
        x_04 = self.conv_04(torch.cat([x_00, x_01, x_02, x_03, self.padding_ornot(x_03, x_13)], dim=1))
        if self.deep_supervision:
            output1 = self.out_conv1(x_01)
            output2 = self.out_conv2(x_02)
            output3 = self.out_conv3(x_03)
            output4 = self.out_conv4(x_04)
            return [output1, output2, output3, output4]
        else:
            output = self.out_conv(x_04)
            return output

    def padding_ornot(self, x1, x2):
        out = self.up_sample(x2)
        if x2.shape[-1]*2 != x1.shape[-1]:
            out = self.padding(out)
        return out