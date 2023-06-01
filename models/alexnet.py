import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AlexNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.alexnet(weights = None if not pretrained else 'DEFAULT')
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)
        if frozen:
            # freeze all layers except the last one
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True



    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
if __name__ == "__main__":
        model = AlexNet(48)
        print(model)