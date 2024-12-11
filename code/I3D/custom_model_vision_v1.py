import torch
import torch.nn as nn
import torch.nn.functional as F
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
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches**0.5, num_patches**0.5]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=256, num_heads=4, depth=4, num_classes=1000, mlp_dim=2048, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Shape: [batch_size, in_channels, img_size, img_size]
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # Shape: [batch_size, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, 1 + num_patches, embed_dim]
        x = x + self.pos_embed  # Add positional embedding
        x = self.dropout(x)

        x = self.transformer(x)  # Shape: [batch_size, 1 + num_patches, embed_dim]
        cls_token_final = x[:, 0]  # Extract the class token

        logits = self.mlp_head(cls_token_final)  # Shape: [batch_size, num_classes]
        return logits

# Combine the I3D Feature Extractor and Vision Transformer
class SignLanguageRecognitionModelViT(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=32, in_channels=3, embed_dim=256, num_heads=4, depth=4, mlp_dim=2048, dropout=0.1):
        super(SignLanguageRecognitionModelViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            num_classes=num_classes,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def forward(self, x):
        # x shape: [batch_size, in_channels, img_size, img_size]
        logits = self.vit(x)  # Forward through Vision Transformer
        return logits

# Example usage
if __name__ == "__main__":
    model = SignLanguageRecognitionModelViT(num_classes=2000, img_size=224, patch_size=32, in_channels=3)
    dummy_input = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB images of size 224x224
    outputs = model(dummy_input)
    print(outputs.shape)  # Should output: [8, 2000]
