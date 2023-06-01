import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)//2
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

class RotNet(nn.Module):
    def __init__(self, num_classes=4):
        super(RotNet, self).__init__()

        num_classes = 4
        num_inchannels = 3
        num_stages = 5


        assert(num_stages >= 3)
        nChannels  = 192
        nChannels2 = 160
        nChannels3 = 96

        early_features = [nn.Sequential() for _ in range(2)]
        early_features[0].add_module('conv1', BasicBlock(num_inchannels, nChannels, 5))
        early_features[0].add_module('conv2', BasicBlock(nChannels,  nChannels2, 1))
        early_features[0].add_module('conv3', BasicBlock(nChannels2, nChannels3, 1))
        early_features[0].add_module('B1_MaxPool', nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        early_features[1].add_module('Block2_ConvB1',  BasicBlock(nChannels3, nChannels, 5))
        early_features[1].add_module('Block2_ConvB2',  BasicBlock(nChannels,  nChannels, 1))
        early_features[1].add_module('Block2_ConvB3',  BasicBlock(nChannels,  nChannels, 1))
        early_features[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))
        latent_features = [nn.Sequential() for _ in range(3)]
        # last two conv layers before global average pooling
        latent_features[0].add_module('Block3_ConvB1',  BasicBlock(nChannels, nChannels, 3))
        latent_features[0].add_module('Block3_ConvB2',  BasicBlock(nChannels, nChannels, 1))
        latent_features[0].add_module('Block3_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        latent_features[0].add_module('Block3_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))
        
        latent_features[1].add_module('Block4_ConvB1',  BasicBlock(nChannels, nChannels, 3))
        latent_features[1].add_module('Block4_ConvB2',  BasicBlock(nChannels, nChannels, 1))
        latent_features[1].add_module('Block4_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        # latent_features[2].add_module('Block5_ConvB1',  BasicBlock(nChannels, nChannels, 3))
        # latent_features[2].add_module('Block5_ConvB2',  BasicBlock(nChannels, nChannels, 1))
        # latent_features[2].add_module('Block5_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        latent_features[2].add_module('GlobalAveragePooling',  GlobalAveragePooling())
        latent_features[2].add_module('Classifier', nn.Linear(nChannels, num_classes))

        self.early_features = nn.Sequential(*early_features)
        self.latent_features = nn.Sequential(*latent_features)
        self.classifier = nn.Sequential(nn.Identity())
        # print total memory size of the model
        print('Memory size of the model: {} MB'.format(sum(p.numel() for p in self.parameters())/1000000.0))

    def to_classifier(self, num_classes):
        # replace latent features with identity
        self.latent_features = nn.Identity()
        self.classifier.add_module('Classifier_ConvB1',  BasicBlock(192, 192, 3))
        self.classifier.add_module('Classifier_ConvB2',  BasicBlock(192, 192, 1))
        self.classifier.add_module('Classifier_ConvB3',  BasicBlock(192, 192, 1))
        self.classifier.add_module('GlobalAvgPool',  GlobalAveragePooling())
        self.classifier.add_module('Linear_F',      nn.Linear(192, num_classes))
        for param in self.classifier.parameters():
            param.requires_grad = True

        
    def forward(self, x):
        out = self.early_features(x)
        out = self.latent_features(out)
        out = self.classifier(out)

        return out