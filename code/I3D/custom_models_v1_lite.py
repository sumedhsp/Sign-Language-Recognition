# Define the I3D Feature Extractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


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

# Define Positional Encoding for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# Define the Transformer Model
class SignLanguageTransformer(nn.Module):
    def __init__(self, d_model=1024, nhead=2, num_layers=1, num_classes=100):
        super(SignLanguageTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean pooling over the sequence length
        x = self.classifier(x)
        return x

# Combine the I3D Feature Extractor and Transformer
class SignLanguageRecognitionModel(nn.Module):
    def __init__(self, i3d_feature_extractor, num_classes):
        super(SignLanguageRecognitionModel, self).__init__()
        self.feature_extractor = i3d_feature_extractor
        self.transformer = SignLanguageTransformer(d_model=1024, nhead=2, num_layers=1, num_classes=num_classes)

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        features = self.feature_extractor(x)
        # features shape: [batch_size, channels, frames, height, width]
        batch_size, channels, frames, height, width = features.shape

	# **Limit Input Frames Here**
        # Downsample to a fixed number of frames (e.g., 16)
        target_frames = 16  # Desired sequence length
        if frames > target_frames:
            indices = torch.linspace(0, frames - 1, target_frames).long().to(x.device)
            features = features[:, :, indices, :, :]


        # Spatial pooling
        features = F.adaptive_avg_pool3d(features, (frames, 1, 1))
        features = features.view(batch_size, channels, frames)  # [batch_size, channels, frames]
        features = features.permute(0, 2, 1)  # [batch_size, frames, channels]
        logits = self.transformer(features)
        return logits
