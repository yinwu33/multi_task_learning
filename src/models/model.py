import torch.nn as nn
from models.backbones import Backbone, DeepLabBackbone
from models.depth_heads import DeepLabDepthHead, UnetDepthHead, BTS
from models.sem_seg_heads import UnetSemSegHead, DeepLabSemSegHead
from models.obj_det_heads import ObjDetHead
from models.DCN.dcn_v2 import DCN
from configs.cfg import args


class Net(nn.Module):
    """
    build network with either single head or multiple heads
    for example, to build a network with backbone of resnet34 and
    a semantic segmentation head, pass arguments
    backbone=your_backbone, sem_seg_head=your_seg_sem_head
    and obj_det_head=None

    :param backbone: nn.Module
    :param xxx_xxx_head: nn.Module
    """

    def __init__(self, backbone,
                 sem_seg_head=None,
                 obj_det_head=None,
                 depth_head=None):
        super(Net, self).__init__()
        self.backbone = backbone
        self.heads = []
        if sem_seg_head is not None:
            self.sem_seg_head = sem_seg_head
            self.heads.append(self.sem_seg_head)
        if obj_det_head is not None:
            self.obj_det_head = obj_det_head
            self.heads.append(self.obj_det_head)
        if depth_head is not None:
            self.depth_head = depth_head
            self.heads.append(self.depth_head)

    def forward(self, x):
        """
        :return: a list of outputs from each head
        """
        features = self.backbone(x)
        return [head(features) for head in self.heads]


class SemNet(nn.Module):
    """
    build a network with only semantic segmentation head
    and backbone.

    :param network: string, name of models architecture
    """

    def __init__(self, network, pretrained=True,
                 sem_num_classes=4):
        super(SemNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.sem_seg_head = UnetSemSegHead(self.backbone.feat_channels,
                                           num_classes=sem_num_classes)
        self.net = Net(self.backbone, self.sem_seg_head)

    def forward(self, x):
        return self.net(x)


class ResRefineNet(nn.Module):
    def __init__(self, network, pretrained=True, sem_num_classes=4):
        super(ResRefineNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.sem_seg_head = RefineNet(num_features=self.backbone.feat_channels, 
                                      num_classes=sem_num_classes)
        self.net = Net(self.backbone, self.sem_seg_head)
    
    def forward(self, x):
        return self.net(x)

class SemNet2(nn.Module):
    #
    def __init__(self, network, pretrained=True,
                 sem_num_classes=4):
        super(SemNet2, self).__init__()
        self.backbone = DeepLabBackbone(network=network, pretrained=pretrained)
        self.sem_seg_head = DeepLabSemSegHead(num_classes=sem_num_classes)
        self.net = Net(self.backbone, self.sem_seg_head)

    def forward(self, x):
        return self.net(x)


class DetNet(nn.Module):
    """
    build a network with only object detection head
    and backbone.

    :param network: string, name of models architecture
    """

    def __init__(self, network, pretrained=True,
                 det_num_classes=12):
        super(DetNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.obj_det_head = ObjDetHead(self.backbone.feat_channels,
                                       num_classes=det_num_classes)
        self.net = Net(self.backbone, self.obj_det_head)

    def forward(self, x):
        return self.net(x)


class SemDetNet(nn.Module):
    def __init__(self, network, pretrained=True,
                 sem_num_classes=4, det_num_classes=21):
        super(SemDetNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.sem_seg_head = UnetSemSegHead(self.backbone.feat_channels,
                                           num_classes=sem_num_classes)
        self.obj_det_head = ObjDetHead(self.backbone.feat_channels,
                                       num_classes=det_num_classes)
        self.net = Net(self.backbone, self.sem_seg_head, self.obj_det_head)

    def forward(self, x):
        return self.net(x)


class DepthNet(nn.Module):
    # net for depth regression using deeplab architecture
    def __init__(self, network='resnet50', pretrained=True):
        super(DepthNet, self).__init__()
        self.backbone = DeepLabBackbone(network=network, pretrained=pretrained)
        self.depth_head = DeepLabDepthHead()
        self.net = Net(self.backbone, depth_head=self.depth_head)

    def forward(self, x):
        return self.net(x)


class DepthUNet(nn.Module):
    # net for depth regression using resunet architecture
    def __init__(self, network='resnet50', pretrained=True):
        super(DepthUNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.depth_head = UnetDepthHead(self.backbone.feat_channels)
        self.net = Net(self.backbone, depth_head=self.depth_head)

    def forward(self, x):
        return self.net(x)


# class ResRefineNet(nn.Module):
#     # resnet + refinenet
#     def __init__(self, network='resnet50', sem_num_classes=4, pretrained=True):
#         super(ResRefineNet, self).__init__()
#         self.backbone = Backbone(network=network, pretrained=pretrained)
#         self.sem_seg_head = RefineNet(self.backbone.feat_channels, num_classes=sem_num_classes)
#         self.net = Net(self.backbone, sem_seg_head=self.sem_seg_head)
#
#     def forward(self, x):
#         return self.net(x)


class BTSNet(nn.Module):
    # net for depth regression using BTS architecture
    def __init__(self, network='resnet101', pretrained=True):
        super(BTSNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        self.depth_head = BTS(self.backbone.feat_channels)
        self.net = Net(self.backbone, depth_head=self.depth_head)

    def forward(self, x):
        return self.net(x)


# class MultiTaskNet(nn.Module):
#     def __init__(self, network, pretrained=True,
#                  sem_num_classes=4, det_num_classes=9):
#         super(MultiTaskNet, self).__init__()
#         self.backbone = Backbone(network=network, pretrained=pretrained)
#         self.sem_seg_head = UnetSemSegHead(self.backbone.feat_channels,
#                                            num_classes=sem_num_classes)
#         self.obj_det_head = ObjDetHead(self.backbone.feat_channels,
#                                        num_classes=det_num_classes)
#         self.depth_head = UnetDepthHead(self.backbone.feat_channels)
#
#         self.net = Net(self.backbone, self.sem_seg_head,
#                        self.obj_det_head, self.depth_head)
#
#     def forward(self, x):
#         return self.net(x)


class ShareNet(nn.Module):
    # Multi-task learning network with resunet backbone.
    def __init__(self, network='resnet101', pretrained=True,
                 sem_num_classes=4, det_num_classes=9):
        super(ShareNet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        feat_channels = self.backbone.feat_channels
        feat_c = feat_channels[::-1]
        self.decoder_block1 = DecoderBlock(feat_c[0], feat_c[1])
        self.decoder_block2 = DecoderBlock(feat_c[1], feat_c[2])
        self.decoder_block3 = DecoderBlock(feat_c[2], feat_c[3])
        self.sem_head = ShareNetSem(feat_c, sem_num_classes)
        self.det_head = ShareNetDet(feat_channels, det_num_classes)
        # self.depth_head = ShareNetDepth(feat_c)
        self.depth_head = UnetDepthHead(feat_channels)


    def forward(self, x):
        feat_map = self.backbone(x)
        x = feat_map[1] + self.decoder_block1(feat_map[0], feat_map[1])
        x = feat_map[2] + self.decoder_block2(x, feat_map[2])
        x = feat_map[3] + self.decoder_block3(x, feat_map[3])
        sem_out = self.sem_head(x)
        det_out = self.det_head(x)
        # depth_out = self.depth_head(x)
        depth_out = self.depth_head(feat_map)

        return [sem_out, det_out, depth_out]


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
            DCN(in_channels // 4, in_channels // 4, 3, padding=1,
                dilation=1, deformable_groups=1),
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


# following are three baseline models for comparison.
class ShareNetSem(nn.Module):
    def __init__(self, feat_c, num_classes=4):
        super(ShareNetSem, self).__init__()
        self.sem_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_c[3], feat_c[3] // 2, 1),
            # nn.ConvTranspose2d(feat_c[3], feat_c[3] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(feat_c[3] // 2),
            nn.ReLU(),
            nn.Conv2d(feat_c[3] // 2, feat_c[3] // 2, 3, padding=1),
            nn.BatchNorm2d(feat_c[3] // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_c[3] // 2, num_classes, 1),
            # nn.ConvTranspose2d(feat_c[3] // 2, num_classes, kernel_size=2, stride=2)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.sem_decoder(x)
        return self.softmax(x)


class ShareNetDet(nn.Module):
    def __init__(self, channels, num_classes):
        super(ShareNetDet, self).__init__()
        channels = channels[::-1]
        # bias setting according to the original implemention of centernet
        self.cls_head = Head(channels[-1], num_classes,
                             bias_fill=True, bias_value=-2.19)
        self.sigmoid = nn.Sigmoid()
        self.wh_head = Head(channels[-1], 2)
        self.reg_head = Head(channels[-1], 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = self.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        return {"cls": cls, "wh": wh, "reg": reg}


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


class ShareNetDepth(nn.Module):
    def __init__(self, feat_c):
        super(ShareNetDepth, self).__init__()
        self.depth_decoder = nn.Sequential(
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.depth_decoder(x)
        return self.sigmoid(x)



if __name__ == "__main__":
    net = ShareNet()
    print(net)
