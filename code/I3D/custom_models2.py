import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision import models
import torchvision.transforms as transforms

# Define the I3D Feature Extractor
class I3DFeatureExtractor(nn.Module):
    def __init__(self, i3d_model):
        super(I3DFeatureExtractor, self).__init__()
        # Exclude the last layers (avg_pool, dropout, logits)
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                (k, i3d_model._modules[k]) for k in list(i3d_model.end_points.keys())
            ])
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

#i3d_feature_extractor = I3DFeatureExtractor(i3d)

# Define the Vision Transformer Model
class SignLanguageViT(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageViT, self).__init__()
        # Load a pre-trained ViT model
        self.vit = models.vit_b_16(pretrained=True)
        # Modify the classifier head to match the number of classes
        # Corrected code:
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: [batch_size * num_frames, 3, H, W]
        logits = self.vit(x)
        return logits

# Combine the I3D Feature Extractor and ViT
class SignLanguageRecognitionModelVision(nn.Module):
    def __init__(self, i3d_feature_extractor, num_classes):
        super(SignLanguageRecognitionModelVision, self).__init__()
        self.feature_extractor = i3d_feature_extractor
        self.num_classes = num_classes
        self.vit_model = SignLanguageViT(num_classes=num_classes)
        # A convolution to map I3D features to 3 channels for ViT
        self.conv = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        batch_size, C, T, H, W = x.size()
        # Extract features using I3D
        features = self.feature_extractor(x)
        # features shape: [batch_size, channels, frames, height, width]
        _, channels, frames, height, width = features.shape
        # Reshape features for ViT
        # We treat each frame's features as an image
        features = features.permute(0, 2, 1, 3, 4)  # [batch_size, frames, channels, H, W]
        features = features.reshape(batch_size * frames, channels, height, width)  # [batch_size * frames, channels, H, W]
        # Map to 3 channels
        features = self.conv(features)  # [batch_size * frames, 3, H, W]
        # Resize to 224x224 for ViT
        features = F.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
        # Pass through ViT
        logits = self.vit_model(features)  # [batch_size * frames, num_classes]
        # Reshape logits back to [batch_size, frames, num_classes]
        logits = logits.view(batch_size, frames, self.num_classes)
        # Average over the time dimension
        logits = logits.mean(dim=1)  # [batch_size, num_classes]
        return logits
