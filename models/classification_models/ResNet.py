import torch
import torch.nn as nn
import torchvision.models as models

weights = models.ResNet50_Weights.DEFAULT
resnet_model = models.resnet50(weights=weights)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features , 3)


