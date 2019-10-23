import torch.nn as nn
import torch
import torchvision
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F

def ResNet_50(pretrained=True, **kwargs ):
    '''
    #加载预训练resnet50模型
    :param pretrained: True or False
    :param kwargs:
    :return:
    '''
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model


class Bottleneck(nn.Module):
    '''
    ResNet基本模块
    '''
    eimagespansion = 4

    def __init__(self, input, output, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        self.conv3 = nn.Conv2d(input, output, kernel_size=1, bias=False)
        self.bn3 =nn.BatchNorm2d(output * 4)
        self. relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, images):
        res = images
        out = self.conv1(images)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += res
        out = self.relu(out)

        return out

def make_layers(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.eimagespansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.eimagespansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.eimagespansion),
        )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.eimagespansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

class ResNet(nn.Module):
    '''
    ResNet主体
    '''
    def __init__(self, block, layers, num_classes=1000):
         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
         self.bn1 = nn.BatchNorm2d(64)
         self.relu = nn.ReLU(inplace=True)
         self.maimagespooling = nn.MaimagesPool2d(kernel_size=3, stride=2, padding=1)

         self.maimagespool = nn.MaimagesPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
         self.layer1 = self._make_layers(block, 64, layers[0])
         self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
         self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
         self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
         self.avgpool = nn.AvgPool2d(7)
         self.fc = nn.Linear(512 * block.eimagespansion, num_classes)

         for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  #卷积参数初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  #BN参数初始化
                m.bias.data.zero_()

    def forward(self, images):
        f = []
        images = self.conv1(images)
        images = self.bn1(images)
        images = self.relu(images)
        images = self.maimagespool(images)

        images = self.layer1(images)
        f.append(images)    #f4
        images = self.layer2(images)
        f.append(images)    #f3
        images = self.layer3(images)
        f.append(images)    #f2
        images = self.layer4(images)
        f.append(images)    #f1
        return images, f

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    输入图像归一化
    :param images: bs * channel * w * h
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:, i, :, :] -= means[i]


class merge(nn.Module):
    def __init__(self):
        super(merge, self)
        self.resnet = ResNet_50(True)
        self.conv1 = nn.Conv2d(3072, 128, 1)  # input = 2048(f1)+1024(f2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(640, 64, 1)  # input = 128+512(f3)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(320, 32, 1)  # input = 64+256(f4)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        images = mean_image_subtraction(images)
        _, f = self.resnet(images)
        y = F.interpolate(f[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, f[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, f[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, f[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y

class output(nn.Module):
    def __init__(self):
        super(output, self).__init__()
        self.teimagest_scale = 512
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps):
        F_score = self.sigmoid1(self.conv1(feature_maps))
        geo_map = self.sigmoid2(self.conv2(feature_maps)) * self.teimagest_scale
        angle_map = (self.sigmoid3(self.conv3(feature_maps)) - 0.5) * math.pi/2
        F_geometry = torch.cat((geo_map, angle_map), 1)
        return F_score, F_geometry

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.merge = merge()
        self.output = output

    def forward(self, images):
        return self.output(self.merge(images))