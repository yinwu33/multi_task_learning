import math
import collections
import torch.nn as nn
import torchvision.models as models


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Backbone(nn.Module):
    """
    download models architecture in torchvison.models subpackage
    and extract backbone from it.
    available models architectures are listed in
    https://pytorch.org/docs/stable/torchvision/models.html

    :param network: name of network
    """
    def __init__(self, network='resnet34', pretrained=True):
        super(Backbone, self).__init__()
        resnet = getattr(models, network)(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # this part is complicated, it is to get the number of channels of every feature_map
        self.feat_channels = []
        for i, module in self.backbone._modules.items():
            # if int(i) > 3:
            #     self.feat_channels.append(list(list(module.children())[-1].children())[-1].num_features)
            if int(i) > 3:
                if network == 'resnet34' or network == 'resnet18':
                    self.feat_channels.append(list(list(module.children())[-1].children())[-1].num_features)
                else:
                    self.feat_channels.append(list(list(module.children())[-1].children())[-2].num_features)


    def forward(self, x):
        features = []
        for i, module in self.backbone._modules.items():
            x = module(x)
            if int(i) == 2:
                features.append(x)
            if int(i) > 3:
                features.append(x)
        return features[::-1]


class DeepLabBackbone(nn.Module):
    '''
    pretrained resnet backbone for Deeplab v3+ network, output stride=16 instead of 32
    currently only support resnet with more than 50 layers.
    for layer 1,2,3,4
    strides = [1, 2, 2, 1]
    '''
    def __init__(self, network='resnet50', pretrained=True):
        super(DeepLabBackbone, self).__init__()
        pretrained_model = Backbone(network=network, pretrained=pretrained)

        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))

        self.bn1 = nn.BatchNorm2d(128)
        modules = pretrained_model._modules['backbone']
        self.relu = modules[2] # relu
        self.maxpool = modules[3] # maxpool
        self.layer1 = modules[4] # layer 1
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1),
                                         stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1),
                                                 stride=(1, 1), bias=False)

        self.layer2 = modules[5] # layer 2

        self.layer3 = modules[6] # layer 3

        self.layer4 = modules[7] # layer 4
        self.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                         dilation=2, padding=2, bias=False)
        self.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1),
                                                 stride=(1, 1), bias=False)

        # clear memory
        del pretrained_model

        weights_init(self.conv1)
        weights_init(self.bn1)
        weights_init(self.layer1[0].conv1)
        weights_init(self.layer1[0].downsample[0])
        weights_init(self.layer4[0].conv2)
        weights_init(self.layer4[0].downsample[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4,  x1]


if __name__ == "__main__":
    backbone = Backbone('resnet101')
    backbone2 = DeepLabBackbone()
    print(backbone.feat_channels)
    print(backbone2)




