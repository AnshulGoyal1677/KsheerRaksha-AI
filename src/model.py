import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class GaitGuardModel(nn.Module):
    """
    GaitGuard AI Spatiotemporal Lameness Classifier.
    Combines a Pretrained 2D CNN backbone (ResNet-18) for spatial frame feature extraction
    with a Bidirectional GRU (BiGRU) and temporal pooling for sequential gait locomotion modeling.
    """
    def __init__(self, num_classes=2, hidden_dim=64, pretrained=True, freeze_backbone=False):
        super(GaitGuardModel, self).__init__()
        
        # Load Pretrained Backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        # Feature dimension before classification layer
        self.feature_dim = base_resnet.fc.in_features  # 512
        base_resnet.fc = nn.Identity()
        self.backbone = base_resnet
        
        # Freeze lower layers (conv1, bn1, layer1, layer2, layer3) to prevent overfitting on small video dataset
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
                
        # Temporal Modeling: Bi-directional GRU
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Combined temporal representation dimension:
        # 2 * hidden_dim (bi-directional) * 2 (mean pool + max pool) = 4 * hidden_dim = 256
        recurrent_dim = hidden_dim * 2 * 2
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(recurrent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, T, C, H, W)
           e.g. (Batch, 16, 3, 224, 224)
        """
        B, T, C, H, W = x.shape
        
        # Merge batch and time dimensions for spatial feature extraction
        x_reshaped = x.view(B * T, C, H, W)
        spatial_features = self.backbone(x_reshaped)  # (B * T, 512)
        
        # Reshape back to temporal sequence: (B, T, 512)
        temporal_sequence = spatial_features.view(B, T, self.feature_dim)
        
        # Recurrent temporal modeling
        gru_out, _ = self.gru(temporal_sequence)  # (B, T, 128)
        
        # Temporal aggregation: Mean pooling + Max pooling along time dimension
        mean_pool = torch.mean(gru_out, dim=1)  # (B, 128)
        max_pool, _ = torch.max(gru_out, dim=1)  # (B, 128)
        gait_descriptor = torch.cat([mean_pool, max_pool], dim=1)  # (B, 256)
        
        # Class logits
        logits = self.classifier(gait_descriptor)  # (B, 2)
        return logits
