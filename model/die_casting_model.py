import torch
import torch.nn as nn
from registry import MODEL_REGISTRY


class MultiLabelwithDensity(nn.Module):
    def __init__(self, cfg, num_classes=3):
        super().__init__()

        base = MODEL_REGISTRY[cfg]
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # Full Connected layer 전까지
        in_features = base.fc.in_features
        
        self.classifier = nn.Linear(in_features + 1, num_classes)
        
        
    def forward(self, x, density):
        # 클래스가 실행이 됨.
        feature = self.backbone(x)
        feature = torch.flatten(feature, 1)
        feature = torch.cat([feature, density], dim=1)
        
        output = self.classifier(feature)
        
        return output
    
    
if __name__ == "__main__":
    mld = MultiLabelwithDensity(num_classes=4)
    image = torch.randn(2, 3, 224, 224)
    density = torch.randn(2, 1)
    mld(image, density)