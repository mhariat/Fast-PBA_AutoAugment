import torch
import torch.nn as nn
import math

from FastAutoAugment.networks.shakedrop import ShakeDrop
from FastAutoAugment.networks.gates.rnn_gates import *


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                       featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes * 1))
        self.conv3 = nn.Conv2d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                       featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class GateLayer(nn.Module):
    def __init__(self, embed_dim, pool_size, inplanes):
        super(GateLayer, self).__init__()
        self.inplanes = inplanes
        self.pool_size = pool_size
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels=self.inplanes,
                              out_channels=self.embed_dim,
                              kernel_size=1,
                              stride=1)

    def forward(self, x):
        x = nn.AvgPool2d(self.pool_size)(x)
        x = self.conv(x)
        return x


class PyramidSkipNet(nn.Module):

    def __init__(self, depth, alpha, num_classes, bottleneck=True, embed_dim=10, hidden_dim=10):
        super(PyramidSkipNet, self).__init__()
        self.inplanes = 16
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        if bottleneck:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        self.num_layers = n

        self.addrate = alpha / (3 * n * 1.0)
        self.ps_shakedrop = [1. - (1.0 - (0.5 / (3 * n)) * (i + 1)) for i in range(3 * n)]

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.original_pyramidal_make_layer(block, n)
        self.pyramidal_make_layer(block, n, stride=2, group_id=1, pool_size=16)
        self.pyramidal_make_layer(block, n, stride=2, group_id=2, pool_size=8, last_gate=False)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        assert len(self.ps_shakedrop) == 0, self.ps_shakedrop

    def original_pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample, p_shakedrop=self.ps_shakedrop.pop(0)))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1, p_shakedrop=self.ps_shakedrop.pop(0)))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def pyramidal_make_layer(self, block, block_depth, stride=1, pool_size=32, group_id=1, last_gate=True):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        self.featuremap_dim = self.featuremap_dim + self.addrate
        layer = block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample, p_shakedrop=self.ps_shakedrop.pop(0))
        gate_layer = GateLayer(embed_dim=self.embed_dim, pool_size=pool_size, inplanes=int(round(self.featuremap_dim))*layer.outchannel_ratio)
        setattr(self, 'group{}_ds{}'.format(group_id, 0), layer.downsample)
        setattr(self, 'group{}_layer{}'.format(group_id, 0), layer)
        setattr(self, 'group{}_gate{}'.format(group_id, 0), gate_layer)
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layer = block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1,
                          p_shakedrop=self.ps_shakedrop.pop(0))
            gate_layer = GateLayer(embed_dim=self.embed_dim, pool_size=pool_size, inplanes=int(round(temp_featuremap_dim))*layer.outchannel_ratio)
            setattr(self, 'group{}_ds{}'.format(group_id, i), layer.downsample)
            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)
            skip_gate = (i == block_depth - 1) and not last_gate
            if not skip_gate:
                setattr(self, 'group{}_gate{}'.format(group_id, i), gate_layer)
            self.featuremap_dim = temp_featuremap_dim

        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        x = getattr(self, 'group1_layer0')(x)
        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x

        for g in range(2):
            for i in range(0 + int(g == 0), self.num_layers):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                x_channels = x.size(1)
                prev_channels = prev.size(1)
                if x_channels != prev_channels:
                    padding = torch.cuda.FloatTensor(batch_size, x_channels - prev_channels,
                                                     x.size(2), x.size(3)).fill_(0)
                    prev = torch.cat((prev, padding), 1)
                prev = x = mask.expand_as(x) * x + (1 - mask).expand_as(prev) * prev
                if not (g == 1 and (i == self.num_layers - 1)):
                    gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                    mask, gprob = self.control(gate_feature)
                    gprobs.append(gprob)
                    masks.append(mask.squeeze())

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, masks, gprobs
