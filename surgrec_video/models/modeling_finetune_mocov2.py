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
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        **kwargs
    }


def _strip_timm_factory_kwargs(kwargs):
    kwargs = dict(kwargs)
    for key in ('pretrained_cfg', 'pretrained_cfg_overlay', 'checkpoint_path',
                'cache_dir', 'scriptable', 'exportable', 'no_jit',
                'pretrained_strict', 'features_only', 'out_indices'):
        kwargs.pop(key, None)
    return kwargs


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


class MoCoV2ResNet3DVideo(nn.Module):
    """
    3D ResNet for video classification, compatible with MoCo V2 pretrained weights
    MoCo V2 uses ResNet50 backbone with momentum contrastive learning for self-supervised learning
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
        
        # Create ResNet50 backbone (same as MoCo V2)
        # Use pretrained=False since we'll load MoCo V2 weights manually
        self.backbone = models.resnet50(pretrained=False)
        
        # Get feature dimension from ResNet50
        self.feature_dim = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove original classification head
        self.backbone.fc = nn.Identity()
        
        # Add 3D temporal processing for video
        # MoCo V2 is 2D, so we add temporal dimension handling
        self.temporal_conv = nn.Conv3d(
            self.feature_dim, self.feature_dim, 
            kernel_size=(3, 1, 1), 
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.temporal_bn = nn.BatchNorm3d(self.feature_dim)
        self.temporal_relu = nn.ReLU(inplace=True)
        
        # Global average pooling for temporal dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head with normalization (following MoCo V2 style)
        if use_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.feature_dim)
        else:
            self.fc_norm = None
            
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for new components"""
        # Initialize classification head
        if hasattr(self.head, 'weight') and self.head.weight is not None:
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)
                
        # Initialize temporal conv
        nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize temporal batch norm
        nn.init.constant_(self.temporal_bn.weight, 1)
        nn.init.constant_(self.temporal_bn.bias, 0)

    def forward_backbone(self, x):
        """Forward through ResNet backbone (2D processing)"""
        # x: [B*T, C, H, W]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global average pooling
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
        
        # Reshape to process frames independently through 2D ResNet
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        # Forward through 2D ResNet backbone
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_backbone, x, use_reentrant=False)
        else:
            x = self.forward_backbone(x)  # [B*T, feature_dim]
        
        # Reshape back to include temporal dimension
        x = x.view(B, T, self.feature_dim)  # [B, T, feature_dim]
        x = x.permute(0, 2, 1).contiguous()  # [B, feature_dim, T]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, T, 1, 1]
        
        # Apply 3D temporal processing
        x = self.temporal_conv(x)  # [B, feature_dim, T, 1, 1]
        x = self.temporal_bn(x)
        x = self.temporal_relu(x)
        
        # Temporal pooling
        x = self.temporal_pool(x)  # [B, feature_dim, 1, 1, 1]
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
        """Return number of layers (for layer decay)"""
        return 4  # ResNet has 4 main layers (layer1, layer2, layer3, layer4)
    
    def no_weight_decay(self):
        """Return parameters that should not have weight decay"""
        nwd = set()
        # Add bias terms
        for name, param in self.named_parameters():
            if 'bias' in name:
                nwd.add(name)
        # Add batch norm parameters
        for name, param in self.named_parameters():
            if 'bn' in name or 'norm' in name:
                nwd.add(name)
        return nwd
    
    def get_classifier(self):
        """Get the classification head"""
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        """Reset the classification head"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()


# Main model class for compatibility
ResNet3DVideo = MoCoV2ResNet3DVideo
DinoVisionTransformer = MoCoV2ResNet3DVideo
VisionTransformer = DinoVisionTransformer


@register_model
def mocov2_resnet50_patch16_224(pretrained=False, **kwargs):
    """MoCo V2 ResNet50 model for video"""
    kwargs = _strip_timm_factory_kwargs(kwargs)
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
    
    model = MoCoV2ResNet3DVideo(**model_kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def dino_resnet50_patch16_224(pretrained=False, **kwargs):
    """MoCo V2 ResNet50 model for video (compatibility alias)"""
    return mocov2_resnet50_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def endofm_vit_small_patch16_224(pretrained=False, **kwargs):
    """MoCo V2 Small model (ResNet50 based)"""
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def endofm_vit_base_patch16_224(pretrained=False, **kwargs):
    """MoCo V2 Base model (ResNet50 based)"""
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def endofm_vit_large_patch16_224(pretrained=False, **kwargs):
    """MoCo V2 Large model (ResNet50 based)"""
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
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
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    return endofm_vit_large_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    kwargs = _strip_timm_factory_kwargs(kwargs)
    model = MoCoV2ResNet3DVideo(
        num_classes=kwargs.pop('num_classes', 1000),
        num_frames=kwargs.pop('all_frames', kwargs.pop('num_frames', 16)),
        tubelet_size=kwargs.pop('tubelet_size', 2),
        use_checkpoint=kwargs.pop('use_checkpoint', True),
        fc_drop_rate=kwargs.pop('fc_drop_rate', 0.0),
        drop_rate=kwargs.pop('drop_rate', 0.0),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        use_mean_pooling=kwargs.pop('use_mean_pooling', True),
        **kwargs)
    model.default_cfg = _cfg()
    return model