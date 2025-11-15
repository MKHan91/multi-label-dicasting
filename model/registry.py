import torch
import torch.nn as nn
import torchvision.models as models

MODEL_REGISTRY = {
   "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
   "resnet101": models.resnet101(wegiths=models.ResNet101_Weights.DEFAULT),
   "resnet152": models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
}


