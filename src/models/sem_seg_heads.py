import torch
import torch.nn as nn
from models.backbones import Backbone
import torch.nn.functional as F


class UnetSemSegHead(nn.Module):
    def __init__(self, feat_channels, num_classes=4):
        """
        Unet like head for semantic segmentation task, summation instead of concatenation
        is used to combine high- and low-level features.
        feature of previous level is upsampled and merged with feature of next level,
        and then the merged feature is upsampled again and merged with...
        :param feat_channels: a list of number of channels of every feature_map
        self.decoder_blocks: blocks that upsample the features
        """
        super(UnetSemSegHead, self).__init__()
        feat_c = feat_channels[::-1]
        self.num_classes = num_classes
        self.decoder_block1 = DecoderBlock(feat_c[0], feat_c[1])
        self.decoder_block2 = DecoderBlock(feat_c[1], feat_c[2])
        self.decoder_block3 = DecoderBlock(feat_c[2], feat_c[3])

        self.last_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_c[3], feat_c[3] // 2, 1),
            nn.BatchNorm2d(feat_c[3] // 2),
            nn.ReLU(),
            nn.Conv2d(feat_c[3] // 2, feat_c[3] // 2, 3, padding=1),
            nn.BatchNorm2d(feat_c[3] // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_c[3] // 2, self.num_classes, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat_map):
        x = feat_map[1] + self.decoder_block1(feat_map[0], feat_map[1])
        x = feat_map[2] + self.decoder_block2(x, feat_map[2])
        x = feat_map[3] + self.decoder_block3(x, feat_map[3])
        x = self.last_block(x)

        return self.softmax(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # bottleneck architecture
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )

        self.deconv_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        """
        :param x1: features from high level
        :param x2: features from low level
        """
        x = self.conv_block1(x1)
        x = self.deconv_block(x)
        # x = center_crop(x, x2.size()[2], x2.size()[3])
        x = self.conv_block2(x)
        return x


# def center_crop(layer, max_height, max_width):
#     _, _, h, w = layer.size()
#     diffy = (h - max_height) // 2
#     diffx = (w - max_width) // 2
#     return layer[:, :, diffy:(diffy + max_height), diffx:(diffx + max_width)]


class DeepLabSemSegHead(nn.Module):
    def __init__(self, num_classes=4):
        super(DeepLabSemSegHead, self).__init__()
        self.aspp = ASPP()
        self.decoder = Decoder(num_classes=num_classes)
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, features):
        x4, x1 = features
        x4 = self.aspp(x4)
        x = self.decoder(x4, x1)
        x = self.upsample(x)

        return x


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        # TODO sync batchnorm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    # Scene understanding module
    def __init__(self):
        super(ASPP, self).__init__()
        inplanes = 2048
        dilations = [1, 6, 12, 18]
        # The pure 1 × 1 convolutional branch can learn complex cross-channel interactions.
        self.aspp1 = _ASPPModule(inplanes, 256, 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp2 = _ASPPModule(inplanes, 256, 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp3 = _ASPPModule(inplanes, 256, 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp4 = _ASPPModule(inplanes, 256, 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=nn.BatchNorm2d)
        # fullimageencoder to get image-level features [B, 256, 1, 1], remove it as
        # in deeplabv3+ origin paper, it has said that the global average pooling branch
        # of ASPP will hurt the performance on cityscapes
        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      nn.ReLU())
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes=4):
        super(Decoder, self).__init__()
        # 256 is the low-level features channels
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x4, x1):
        # x1 and x4 are the low and high level feature maps from DeepLabBackbone respectively.
        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)
        x = torch.cat((x4, x1), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    backbone = Backbone()
    head = DeepLabSemSegHead(backbone.feat_channels)
    print(head)
