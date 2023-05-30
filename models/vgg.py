# Define a vgg net model 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG(nn.Module):
    def __init__(self, num_classes,frozen=True,pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.vgg = torchvision.models.vgg13(weights = None if not pretrained else 'DEFAULT')
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.vgg.classifier[6].requires_grad = True
        # freeze all layers except the last one
        if frozen:
            for param in self.vgg.features.parameters():
                param.requires_grad = False
            for param in self.vgg.classifier.parameters():
                param.requires_grad = True
        
        return 
        
    def forward(self, x):
        return self.vgg(x)
    


if __name__ == "__main__":
    model = VGG(10)
    print(model)