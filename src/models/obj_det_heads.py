import math
import torch.nn as nn
from models.DCN.dcn_v2 import DCN


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, out_padding):
        super(DeconvBlock, self).__init__()
        # Modulated deformable convolution with offset.
        self.dcn = DCN(in_channels, out_channels,
                       dilation=1, deformable_groups=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, padding=padding,
            output_padding=out_padding,
            bias=False,
        )
        self._deconv_init()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


    def _deconv_init(self):
        '''
        weight initialization of upsampling layer
        '''
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class CenternetDeconv(nn.Module):
    def __init__(self, channels, kernel_size=4,
                 stride=2, padding=1, out_padding=0):
        super(CenternetDeconv, self).__init__()
        kernel_size = kernel_size
        stride = stride
        padding = padding
        out_padding = out_padding

        blocks = []

        for i in range(3):
            blocks.append(
                DeconvBlock(channels[i], channels[i+1],
                            kernel_size=kernel_size,
                            stride=stride, padding=padding,
                            out_padding=out_padding)
            )

        self.deconv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.deconv(x)


# Centernet object detection head has three heads of same structure for cls, wh and reg
class Head(nn.Module):
    def __init__(self, in_channel, out_channel,
                 bias_fill=False, bias_value=0.0):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.conv2.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ObjDetHead(nn.Module):
    def __init__(self, channels, num_classes, kernel_size=4,
                 stride=2, padding=1, out_padding=0):
        super(ObjDetHead, self).__init__()
        channels = channels[::-1]
        self.deconv = CenternetDeconv(channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      out_padding=out_padding)
        # bias setting according to the original implemention of centernet
        self.cls_head = Head(channels[-1], num_classes,
                             bias_fill=True, bias_value=-2.19)
        self.sigmoid = nn.Sigmoid()
        self.wh_head = Head(channels[-1], 2)
        self.reg_head = Head(channels[-1], 2)

    def forward(self, x):
        x = self.deconv(x[0])
        cls = self.cls_head(x)
        cls = self.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        return {"cls": cls, "wh": wh, "reg": reg}

