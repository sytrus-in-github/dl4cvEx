import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import numpy as np
from ..data_utils import labels_list


NCLASS = len(labels_list)-1


def pad_32x(tensor):
    """pad tensor with 0 so that its shape is blocks of 32"""
    n,c,h,w = tensor.size()
    if h % 32 == 0 and w % 32 == 0:
        return tensor
    h_new = (h-1) // 32 * 32 + 32
    w_new = (w-1) // 32 * 32 + 32
    return F.pad(tensor, (0, h_new-h, 0, w_new-w))
#    tensor_out = tensor.new(n, c, h_new, w_new).fill_(0)
#    c_tensor_out = tensor_out.narrow(2, 0, h).narrow(3, 0, w)
#    c_tensor_out.copy_(tensor)
#    return tensor_out


class SegmentationNetwork(nn.Module):

    def __init__(self):

        super(SegmentationNetwork, self).__init__()

        

        ############################################################################

        #                             YOUR CODE                                    #

        ############################################################################
        # get pretrained resnet
        self.backbone_model = resnet50(pretrained=True)
        # discard avgpooling and final dense layer
        self.backbone_model.avgpool = None
        self.backbone_model.group2 = None
        # add FCN-like heads
        h = OrderedDict()
        h['conv'] = nn.Conv2d(2048, NCLASS, 1)
        h['deconv'] = nn.ConvTranspose2d(NCLASS, NCLASS, 64, padding=16, stride=32, bias=False)
        initial_weight = get_upsampling_weight(NCLASS, NCLASS, 64)
        h['deconv'].weight.data.copy_(initial_weight)
        self.seghead32 = nn.Sequential(h)
        h = OrderedDict()
        h['conv'] = nn.Conv2d(1024, NCLASS, 1)
        h['deconv'] = nn.ConvTranspose2d(NCLASS, NCLASS, 32, padding=8, stride=16, bias=False)
        initial_weight = get_upsampling_weight(NCLASS, NCLASS, 32)
        h['deconv'].weight.data.copy_(initial_weight)
        self.seghead16 = nn.Sequential(h)
        h = OrderedDict()
        h['conv'] = nn.Conv2d(512, NCLASS, 1)
        h['deconv'] = nn.ConvTranspose2d(NCLASS, NCLASS, 16, padding=4, stride=8, bias=False)
        initial_weight = get_upsampling_weight(NCLASS, NCLASS, 16)
        h['deconv'].weight.data.copy_(initial_weight)
        self.seghead8 = nn.Sequential(h)
        

    def forward(self, x):

        """

        Forward pass of the convolutional neural network. Should not be called

        manually but by calling a model instance directly.



        Inputs:

        - x: PyTorch input Variable

        """

        ############################################################################

        #                             YOUR CODE                                    #

        ############################################################################
        _, _, h, w = x.size()
#        x = Variable(pad_32x(x.data))
        x = pad_32x(x)
#        print x.size()
        x = self.backbone_model.group1(x)
#        print x.size()
        x = self.backbone_model.layer1(x)
#        print x.size()
        x = self.backbone_model.layer2(x)
        x8 = self.seghead8(x)
#        print x.size()
        x = self.backbone_model.layer3(x)
#        print x.size()
        x16 = self.seghead16(x)        
        x = self.backbone_model.layer4(x)
#        print x.size()
        x = self.seghead32(x)
#        print x.size()
        x = x + x16 + x8
        x = x.narrow(2, 0, h).narrow(3, 0, w)        
        return x



    def save(self, path):

        """

        Save model with its parameters to the given path. Conventionally the

        path should end with "*.model".



        Inputs:

        - path: path string

        """

        print 'Saving model... %s' % path

        torch.save(self, path)


# code adapted from https://github.com/aaron-xichen/pytorch-playground/


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def load_state_dict(model, model_urls, model_root):
    import re
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)

        return x


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_state_dict(model, model_urls['resnet50'], model_root)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        load_state_dict(model, model_urls['resnet101'], model_root)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        load_state_dict(model, model_urls['resnet152'], model_root)
    return model


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model


# from https://github.com/wkentaro/pytorch-fcn/


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()