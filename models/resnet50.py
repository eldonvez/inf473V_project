import torchvision
import torch.nn as nn
import torch


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50()
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model = ResNetFinetune(48)
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)
    print(model(torch.randn(1, 3, 224, 224)))