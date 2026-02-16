import torch
import torch.nn as nn
import torchvision.models as models


class CropStressResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CropStressResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze all parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the original fully connected (classifier) layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
