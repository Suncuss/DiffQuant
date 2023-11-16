import torch.nn as nn
import torchvision.models as models


class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        
        # Replace the last fully connected layer to output a single continuous value
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)


def getResNet18RegressionModel():
    return ResNetRegression()

