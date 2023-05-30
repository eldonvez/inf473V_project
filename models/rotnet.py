import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)/2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

        feat = F.avg_pool2d(feat, feat.size(3)).view(-1, self.nChannels)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class NetworkInNetwork(nn.Module):
    def __init__(self, opt):
        super(NetworkInNetwork, self).__init__()

        num_classes = opt['num_classes']
        num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
        use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True


        assert(num_stages >= 3)
        nChannels  = 192
        nChannels2 = 160
        nChannels3 = 96

        early_features = nn.Sequential()
        # first two conv layers

        latent_features = nn.Sequential()
        # last two conv layers before global average pooling

        classifier = nn.Sequential()
        # linear classifier


        def extract_features(self):
            # replace latent features with identity
            self.latent_features = nn.Sequential()
            