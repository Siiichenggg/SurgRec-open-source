from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 3, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResNet3DWrapper(nn.Module):
    """
    3D ResNet wrapper for video processing with DINO pretrained weights
    Adapts 2D ResNet50 to process video sequences
    """
    def __init__(self, 
                 num_classes=1000,
                 num_frames=16,
                 tubelet_size=2,
                 use_checkpoint=True,
                 fc_drop_rate=0.0,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 use_mean_pooling=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_checkpoint = use_checkpoint
        self.use_mean_pooling = use_mean_pooling
        
        # Create base ResNet50 architecture
        self.backbone = models.resnet50(pretrained=False)
        
        # Remove the final classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Modify first conv layer to handle temporal dimension
        # Convert 2D conv to 3D conv for temporal processing
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=64,
            kernel_size=(tubelet_size, 7, 7),
            stride=(tubelet_size, 2, 2),
            padding=(0, 3, 3),
            bias=False
        )
        
        # Initialize 3D conv weights from 2D conv weights
        with torch.no_grad():
            # Replicate 2D weights across temporal dimension and normalize
            weight_2d = original_conv1.weight  # [64, 3, 7, 7]
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)  # [64, 3, tubelet_size, 7, 7]
            weight_3d = weight_3d / tubelet_size  # Normalize to maintain same scale
            self.backbone.conv1.weight.copy_(weight_3d)
        
        # Add temporal pooling layers
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        if use_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.feature_dim)
        else:
            self.fc_norm = None
            
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize head
        if hasattr(self.head, 'weight'):
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def _forward_backbone_2d(self, x):
        """Forward through 2D ResNet backbone"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_features(self, x):
        """
        Forward pass for feature extraction
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Reshape for 3D processing
        # Process video through modified backbone
        x = self._forward_backbone_2d(x)  # [B, feature_dim]
        
        if self.fc_norm is not None:
            x = self.fc_norm(x)
            
        return x

    def forward(self, x):
        """
        Full forward pass
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        """
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x
    
    def get_num_layers(self):
        """Return number of layers (for compatibility)"""
        return 4  # ResNet has 4 main layers
    
    def no_weight_decay(self):
        """Return parameters that should not have weight decay"""
        return set()
    
    def get_classifier(self):
        """Get the classification head"""
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        """Reset the classification head"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()


class ResNet3DVideo(nn.Module):
    """
    3D ResNet for video classification, compatible with DINO pretrained weights
    """
    def __init__(self, 
                 num_classes=1000,
                 num_frames=16,
                 tubelet_size=2,
                 use_checkpoint=True,
                 fc_drop_rate=0.0,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 use_mean_pooling=True,
                 pretrained_2d=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_checkpoint = use_checkpoint
        self.use_mean_pooling = use_mean_pooling
        
        # Create ResNet50 backbone
        if pretrained_2d:
            self.backbone = models.resnet50(pretrained=False)
        else:
            self.backbone = models.resnet50(pretrained=False)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Remove original classification head
        self.backbone.fc = nn.Identity()
        
        # Add temporal processing
        self.temporal_conv = nn.Conv3d(
            self.feature_dim, self.feature_dim, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head with normalization
        if use_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.feature_dim)
        else:
            self.fc_norm = None
            
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        if hasattr(self.head, 'weight'):
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)
                
        # Initialize temporal conv
        nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.temporal_conv.bias is not None:
            nn.init.constant_(self.temporal_conv.bias, 0)

    def forward_backbone(self, x):
        """Forward through ResNet backbone"""
        # x: [B*T, C, H, W]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # [B*T, feature_dim]
        return x

    def forward_features(self, x):
        """
        Forward pass for feature extraction
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Reshape to process frames independently
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        # Forward through backbone
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_backbone, x, use_reentrant=False)
        else:
            x = self.forward_backbone(x)  # [B*T, feature_dim]
        
        # Reshape back to include temporal dimension
        x = x.view(B, T, self.feature_dim)  # [B, T, feature_dim]
        x = x.permute(0, 2, 1).contiguous()  # [B, feature_dim, T]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, T, 1, 1]
        
        # Apply temporal convolution
        x = self.temporal_conv(x)  # [B, feature_dim, T, 1, 1]
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)  # [B, feature_dim, 1, 1, 1]
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, feature_dim]
        
        # Apply normalization
        if self.fc_norm is not None:
            x = self.fc_norm(x)
            
        return x

    def forward(self, x):
        """
        Full forward pass
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        """
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x
    
    def get_num_layers(self):
        """Return number of layers (for compatibility)"""
        return 4  # ResNet has 4 main layers
    
    def no_weight_decay(self):
        """Return parameters that should not have weight decay"""
        return {'temporal_conv.bias'} if hasattr(self.temporal_conv, 'bias') else set()
    
    def get_classifier(self):
        """Get the classification head"""
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        """Reset the classification head"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()


# Main model class for compatibility
DinoVisionTransformer = ResNet3DVideo

# Compatibility wrapper to match original VisionTransformer interface
VisionTransformer = DinoVisionTransformer


@register_model
def dino_resnet50_patch16_224(pretrained=False, **kwargs):
    """DINO ResNet50 model for video"""
    # Extract specific parameters to avoid duplicate keyword arguments
    model_kwargs = {
        'num_classes': kwargs.pop('num_classes', 1000),
        'num_frames': kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        'tubelet_size': kwargs.pop('tubelet_size', 2),
        'use_checkpoint': kwargs.pop('use_checkpoint', True),
        'fc_drop_rate': kwargs.pop('fc_drop_rate', 0.0),
        'drop_rate': kwargs.pop('drop_rate', 0.0),
        'drop_path_rate': kwargs.pop('drop_path_rate', 0.0),
        'use_mean_pooling': kwargs.pop('use_mean_pooling', True),
    }
    
    model = ResNet3DVideo(**model_kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def endofm_vit_small_patch16_224(pretrained=False, **kwargs):
    """DINO Small model (ResNet50 based)"""
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def endofm_vit_base_patch16_224(pretrained=False, **kwargs):
    """DINO Base model (ResNet50 based)"""
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def endofm_vit_large_patch16_224(pretrained=False, **kwargs):
    """DINO Large model (ResNet50 based)"""
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


# Keep original model registrations for backward compatibility
@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    return endofm_vit_small_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    return endofm_vit_base_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    return endofm_vit_large_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = ResNet3DVideo(
        num_classes=kwargs.get('num_classes', 1000),
        num_frames=kwargs.get('all_frames', 16),
        tubelet_size=kwargs.get('tubelet_size', 2),
        use_checkpoint=kwargs.get('use_checkpoint', True),
        fc_drop_rate=kwargs.get('fc_drop_rate', 0.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.get('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model