import torchvision
import torch.nn as nn


class ResNetFinetuneSmaller(nn.Module):
    def __init__(self, num_classes, frozen=False, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights = None if not pretrained else 'ResNet18_Weights.default')
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
    model = ResNetFinetuneSmaller(10)
    print(model)