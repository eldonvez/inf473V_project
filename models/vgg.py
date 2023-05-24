# Define a vgg net model 
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super.__init__()
        self.pretrained = pretrained
        self.vgg = torchvision.models.vgg16(pretrained=self.pretrained)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.vgg.classifier[6].requires_grad = True
        # freeze all layers except the last one
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        for param in self.vgg.classifier[:-1].parameters():
            param.requires_grad = True
        return 
        
    def forward(self, x):
        return self.vgg(x)
    