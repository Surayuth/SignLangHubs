import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights, mobilenet_v2, MobileNet_V2_Weights


# always use this base class
class SL_VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
    
class SL_Mobilenet_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        return self.model(x)