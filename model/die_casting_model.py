import torch
import torch.nn as nn
from model.registry import MODEL_REGISTRY


# region - Encoder1
class Encoder1(nn.Module):
    def __init__(self, cfg):
        super(Encoder1, self).__init__()
        # 사전 학습된 모델 로드 (가장 마지막 global pooling과 fc 레이어 제외)
        res =  MODEL_REGISTRY[cfg.arch_name]
        self.initial = nn.Sequential(res.conv1, res.bn1, res.relu) # 1/2 크기 (112)
        self.maxpool = res.maxpool # 1/4 크기 (56)
        self.layer1 = res.layer1   # 1/4 크기 (56)
        self.layer2 = res.layer2   # 1/8 크기 (28)
        self.layer3 = res.layer3   # 1/16 크기 (14)
        self.layer4 = res.layer4   # 1/32 크기 (7) - Bridge 역할

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(2048*7*7, 128)
        
        
    def forward(self, x):
        x = self.initial(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Multi-scale 특징 추출
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        latent = self.flatten(f4)
        latent = self.bottleneck(latent)
        
        return latent, [f2, f3, f4]
    

# region - Encoder2
class Encoder2(nn.Module):
    def __init__(self, cfg):
        super(Encoder2, self).__init__()
        # 사전 학습된 모델 로드 (가장 마지막 global pooling과 fc 레이어 제외)
        res =  MODEL_REGISTRY[cfg.arch_name]
        self.initial = nn.Sequential(res.conv1, res.bn1, res.relu) # 1/2 크기 (112)
        self.maxpool = res.maxpool # 1/4 크기 (56)
        self.layer1 = res.layer1   # 1/4 크기 (56)
        self.layer2 = res.layer2   # 1/8 크기 (28)
        self.layer3 = res.layer3   # 1/16 크기 (14)
        self.layer4 = res.layer4   # 1/32 크기 (7) - Bridge 역할

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(2048*7*7, 128)
        
        
    def forward(self, x):
        x = self.initial(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Multi-scale 특징 추출
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # for i, b in enumerate(self.layer4):
        #     if i == len(self.layer4) - 1: # 마지막 블록일 때
        #         # Bottleneck 내부의 마지막 BN까지만 연산하고 ReLU는 건너뜀
        #         identity = b.downsample(s4) if b.downsample is not None else s4
        #         out = b.conv1(s4)
        #         out = b.bn1(out)
        #         out = b.relu(out)
        #         out = b.conv2(out)
        #         out = b.bn2(out)
        #         out = b.relu(out)
        #         out = b.conv3(out)
        #         bridge = b.bn3(out)
        #         bridge += identity
        #     else:
        #         s4 = b(s4)
        latent = self.flatten(f4)
        latent = self.bottleneck(latent)
        
        return latent, [f2, f3, f4]
    

# region - Decoder1
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        # ResNet-50의 마지막 layer4 출력 채널은 2048입니다.
        # 단계별로 채널을 줄이며 해상도를 높입니다.
        
        self.fc = nn.Linear(128, 2048*7*7)
        
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
        x = self.fc(x)
        x = x.view(-1, 2048, 7, 7)
        x = self.decoder(x)
        
        return x


# region - Decoder2
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        # Stage 1: bridge (2048) + s4 (1024) -> 1024
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1) # cat 후 채널 조정
        
        # Stage 2: x (1024) + s3 (512) -> 512
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)
        
        # Stage 3: x (512) + s2 (256) -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        
        # Stage 4: x (256) + s1 (64) -> 64
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        
        # Stage 5: 최종 해상도 복구 (224x224)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, skips, bridge):
        # skips = [s1, s2, s3, s4]
        s1, s2, s3, s4 = skips
        
        # Stage 1 (7x7 -> 14x14)
        x = self.up1(bridge)
        x = torch.cat([x, s4], dim=1) # Skip Connection!
        x = nn.functional.relu(self.conv1(x))
        
        # Stage 2 (14x14 -> 28x28)
        x = self.up2(x)
        x = torch.cat([x, s3], dim=1)
        x = nn.functional.relu(self.conv2(x))
        
        # Stage 3 (28x28 -> 56x56)
        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = nn.functional.relu(self.conv3(x))
        
        # Stage 4 (56x56 -> 112x112)
        x = self.up4(x)
        x = torch.cat([x, s1], dim=1)
        x = nn.functional.relu(self.conv4(x))
        
        # Final Stage (112x112 -> 224x224)
        x = self.final_up(x)
        x = self.sigmoid(self.final_conv(x))
        
        return x
    
    
    
class AnomalyDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder1 = Encoder1(cfg)
        self.encoder2 = Encoder2(cfg)
        self.decoder1 = Decoder1()
        self.decoder2 = Decoder2()
        
        
    def forward(self, x):
        z, self.ms_features = self.encoder1(x)
        recon = self.decoder1(z) # [Batch, 3, 224, 224]
        # recon = self.decoder2(skip_layers, latent) # [Batch, 3, 224, 224]
        
        return z, recon
    
    