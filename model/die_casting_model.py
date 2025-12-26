import torch
import torch.nn as nn
from model.registry import MODEL_REGISTRY


class MultiLabelwithDensity(nn.Module):
    def __init__(self, cfg, num_classes=3):
        super().__init__()

        base = MODEL_REGISTRY[cfg.arch_name]
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
    

class ResNet50_Decoder(nn.Module):
    def __init__(self):
        super(ResNet50_Decoder, self).__init__()
        
        # ResNet-50의 마지막 layer4 출력 채널은 2048입니다.
        # 단계별로 채널을 줄이며 해상도를 높입니다.
        
        self.decoder = nn.Sequential(
            # Stage 1: 2048 -> 1024 (7x7 -> 14x14)
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # Stage 2: 1024 -> 512 (14x14 -> 28x28)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Stage 3: 512 -> 256 (28x28 -> 56x56)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4: 256 -> 64 (56x56 -> 112x112)
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 5: Final Reconstruction (112x112 -> 224x224)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 이미지 픽셀 값(0~1) 출력을 위해 Sigmoid 사용
        )

    def forward(self, x):
        return self.decoder(x)
    
    
class AnomalyDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        base = MODEL_REGISTRY[cfg.arch_name]
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.decoder = ResNet50_Decoder()
        
    def forward(self, x):
        latent = self.encoder(x) # [Batch, 2048, 7, 7]
        reconstructed = self.decoder(latent) # [Batch, 3, 224, 224]
        return reconstructed
    
    