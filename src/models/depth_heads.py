import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetDepthHead(nn.Module):
    def __init__(self, feat_channels):
        """
        Unet like architecture
        identical to sem_seg_head until the last layer
        """
        super(UnetDepthHead, self).__init__()
        feat_c = feat_channels[::-1]
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
            nn.Conv2d(feat_c[3] // 2, 1, 1)
        )

        # self.ordinalregression = OrdinalRegressionLayer()

        # self._init_weight()

    def forward(self, feat_map):
        x = feat_map[1] + self.decoder_block1(feat_map[0], feat_map[1])
        x = feat_map[2] + self.decoder_block2(x, feat_map[2])
        x = feat_map[3] + self.decoder_block3(x, feat_map[3])
        x = self.last_block(x)
        x = nn.Sigmoid()(x)

        return x


    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


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

        # self._init_weight()

    def forward(self, x1, x2):
        """
        :param x1: features from high level
        :param x2: features from low level
        """
        x = self.conv_block1(x1)
        x = self.deconv_block(x)
        # x = center_crop(x, x2.size()[2], x2.size()[3])
        return self.conv_block2(x)


    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


# def center_crop(layer, max_height, max_width):
#     _, _, h, w = layer.size()
#     diffy = (h - max_height) // 2
#     diffx = (w - max_width) // 2
#     return layer[:, :, diffy:(diffy + max_height), diffx:(diffx + max_width)]


class DeepLabDepthHead(nn.Module):
    # ord_num = num_channels // 2
    def __init__(self, num_channels=128):
        super(DeepLabDepthHead, self).__init__()
        self.aspp = ASPP()
        self.decoder = Decoder(num_channels=num_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(num_channels, num_channels // 2, 1, bias=False)
        # self.ordinalregression = OrdinalRegressionLayer()
        self.depthregression = torch.nn.Sequential(nn.Conv2d(num_channels // 2,
                                                             1, 3, 1, 1, bias=False))

        self._init_weight()

    def forward(self, features):
        x4, x1 = features
        x4 = self.aspp(x4)
        x = self.decoder(x4, x1)
        x = self.upsample(x)
        x = self.conv(x)
        depth = self.depthregression(x)

        return depth

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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
        # fullimageencoder to get image-level features [B, 256, 1, 1]
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

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
    def __init__(self, num_channels=188):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False) # 256 is the low-level features channels
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_channels, kernel_size=1, stride=1))
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


class OrdinalRegressionLayer(nn.Module):
    '''
    spacing-increasing discretization (SID) strategy
    to discretize depth and recast depth network learning as an ordinal regression problem.
    '''
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: B x H x W x C, B is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations,
        size is B x H X W X C (C = 2num_ord, num_ord is interval of SID)
        decode_label is the ordinal labels for each position of Image I
        """
        B, C, H, W = x.size()
        num_ord = C // 2
        x = x.view(-1, 2, num_ord, H, W)
        # ord_prob: [B, C, W, H], according to paper, we use log-softmax hier for loss calculation
        prob = F.log_softmax(x, dim=1).view(B, C, H, W)
        # probability that predicted ord is larger than num_ord
        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        # ord_prob = F.softmax(x, dim=1)[:, 1, :, :, :] probability
        # that predicted ord is smaller than num_ord
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        return prob, ord_label, ord_prob


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(UpConv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                        kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class AtrousConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(AtrousConv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class LocalPlanarGuidance(nn.Module):
    def __init__(self, upratio):
        super(LocalPlanarGuidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)


class BTS(nn.Module):
    #
    def __init__(self, feat_out_channels, num_features=512):
        super(BTS, self).__init__()
        self.upconv5 = UpConv(feat_out_channels[3], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[2], num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = UpConv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[1], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = AtrousConv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = AtrousConv(num_features // 2 + num_features // 4 + feat_out_channels[1], num_features // 4, 6)
        self.daspp_12 = AtrousConv(num_features + feat_out_channels[1], num_features // 4, 12)
        self.daspp_18 = AtrousConv(num_features + num_features // 4 + feat_out_channels[1], num_features // 4, 18)
        self.daspp_24 = AtrousConv(num_features + num_features // 2 + feat_out_channels[1], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, 128)
        self.lpg8x8 = LocalPlanarGuidance(8)

        self.upconv3 = UpConv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[0] + 1, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, 128)
        self.lpg4x4 = LocalPlanarGuidance(4)

        self.upconv2 = UpConv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(num_features // 8 + feat_out_channels[0] // 4 + 1, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, 128)
        self.lpg2x2 = LocalPlanarGuidance(2)

        self.upconv1 = UpConv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, 128, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False))

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[4], features[3], features[2], features[1]
        dense_features = torch.nn.ReLU()(features[0])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / 128
        depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / 128
        depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / 128

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        depth = self.get_depth(iconv1)

        # return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, depth
        return depth
