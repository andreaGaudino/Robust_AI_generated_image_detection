"""
Model architectures for ImSecu2026 Challenge

Team: Antonello Di Pede, Dario Gosmar, Andrea Gaudino
Course: IMSECU 2026 - Eurecom
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoImageProcessor, AutoModel

import warnings
warnings.filterwarnings('ignore')

class CLIPDetector(nn.Module):
    """
    CLIP-based detector with frozen backbone
    
    Args:
        model_name: CLIP model variant (default: ViT-B/16)
        freeze_backbone: If True, only train classifier head
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze_backbone=True, only_features=True):
        super().__init__()
        
        print(f"Loading {model_name}...")
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # Freeze backbone weights
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen (only classifier trainable)")
        
        # Get feature dimension
        feature_dim = self.clip.config.vision_config.hidden_size
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.only_features = only_features
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✓ Parameters: {trainable:,} trainable / {total:,} total")
    
    def forward(self, images, return_tokens=False):
        """
        Forward pass
        
        Args:
            images: Tensor of shape (batch, 3, 448, 448)
            return_tokens: If True, return returns (patch_tokens, (Htok,Wtok))
        Returns:
            logits: Tensor of shape (batch, 1)
        """
        # Extract features from CLIP (no gradients)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=images,interpolate_pos_encoding=True)
            if return_tokens:
                # last_hidden_state: (B, 1 + T, D) with CLS at index 0
                hs = outputs.last_hidden_state
                patch_tokens = hs[:, 1:, :]  # (B, T, D), drop CLS

                # For ViT, T = (H/patch)*(W/patch)
                patch = self.clip.config.vision_config.patch_size  # 16 for ViT-B/16
                Htok = images.shape[-2] // patch
                Wtok = images.shape[-1] // patch
                return patch_tokens, (Htok, Wtok)
            
            features = outputs.pooler_output  # (batch, feature_dim)
            
        
        if self.only_features:
            return features
        logits = self.classifier(features)
        return logits
    

class DINOv2Detector(nn.Module):
    """
    DINOv2-based detector with frozen backbone
    
    Args:
        model_name: DINOv2 model variant (default: ViT-S/14)
        freeze_backbone: If True, only train classifier head
    """
    
    def __init__(self, model_name="facebook/dinov2-with-registers-base", freeze_backbone=True, only_features=True):
        super().__init__()
        
        print(f"Loading {model_name}...")
        self.dino = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone weights
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen (only classifier trainable)")
        
        # Get feature dimension
        feature_dim = self.dino.config.hidden_size
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.only_features = only_features
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✓ Parameters: {trainable:,} trainable / {total:,} total")
    
    def forward(self, pixel_values, return_tokens=False):
        """
        Forward pass
        
        Args:
            pixel_values: ideally already preprocessed for DINOv2 (B, 3, H, W)
            return_tokens: If True, return returns (patch_tokens, (Htok,Wtok))
        Returns:
            features: (B, D) if only_features else logits: (B, 1)
        """
        # Extract features from DINOv2 (no gradients)
        with torch.no_grad():
            try:
                outputs = self.dino(pixel_values=pixel_values, interpolate_pos_encoding=True)
            except TypeError:
                outputs = self.dino(pixel_values=pixel_values)
            hs = outputs.last_hidden_state  # (B, 1 + R + T, D)

            if return_tokens:
                R = int(getattr(self.dino.config, "num_register_tokens", 0))  # 4 in your config
                # Token order in HF Dinov2WithRegistersModel: [CLS] + [REG...]*R + [PATCH...]*T
                patch_tokens = hs[:, 1 + R :, :]  # (B, T, D), drop CLS+registers

                patch = int(self.dino.config.patch_size)  # 14
                Htok = pixel_values.shape[-2] // patch  # 32
                Wtok = pixel_values.shape[-1] // patch  # 32
                return patch_tokens, (Htok, Wtok)

            features = hs[:, 0, :]  # CLS token (batch, feature_dim)
        
        if self.only_features:
            return features
        
        logits = self.classifier(features)
        return logits


def _tokens_to_grid(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # (B, T, D) -> (B, D, H, W)
    B, T, D = x.shape
    if T != h * w:
        raise ValueError(f"Token count {T} != h*w {h*w}")
    return x.transpose(1, 2).reshape(B, D, h, w)

def _grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    # (B, D, H, W) -> (B, H*W, D)
    B, D, H, W = x.shape
    return x.reshape(B, D, H * W).transpose(1, 2)

class RegionAlignedFusionNet(nn.Module):
    """
    Region-aligned CLIP+DINO token fusion:
    tokens at the same spatial location are fused together.
    """
    def __init__(self,
                 dim_dino=768,
                 dim_clip=768,
                 fusion_dim=512,
                 stream_heads=8,
                 spatial_heads=8,
                 spatial_depth=2,
                 num_classes=1,
                 target="dino"):  # "dino"->32x32, "clip"->28x28
        super().__init__()
        if fusion_dim % stream_heads != 0 or fusion_dim % spatial_heads != 0:
            raise ValueError("fusion_dim must be divisible by the number of heads")

        self.target = target

        self.proj_dino = nn.Linear(dim_dino, fusion_dim)
        self.proj_clip = nn.Linear(dim_clip, fusion_dim)

        # attention over the 2-token stream axis per location
        self.stream_attn = nn.MultiheadAttention(fusion_dim, stream_heads, batch_first=True)
        self.stream_ln = nn.LayerNorm(fusion_dim)

        # optional spatial transformer over fused location tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=spatial_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(enc_layer, num_layers=spatial_depth)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Linear(fusion_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self,
                dino_tokens: torch.Tensor, dino_hw: tuple[int, int],
                clip_tokens: torch.Tensor, clip_hw: tuple[int, int]):
        """
        dino_tokens: (B, Td, 768) patch tokens only
        clip_tokens: (B, Tc, 768) patch tokens only
        """
        Hd, Wd = dino_hw
        Hc, Wc = clip_hw
        B = dino_tokens.shape[0]

        # Project
        d = self.proj_dino(dino_tokens)  # (B, Td, F)
        c = self.proj_clip(clip_tokens)  # (B, Tc, F)

        # To grids
        d_grid = _tokens_to_grid(d, Hd, Wd)  # (B, F, Hd, Wd)
        c_grid = _tokens_to_grid(c, Hc, Wc)  # (B, F, Hc, Wc)

        # Choose target grid (keeping "same region" means matching grids; resizing is the compromise)
        if self.target == "dino":
            Ht, Wt = Hd, Wd  # 32x32 at 448 with patch14
        elif self.target == "clip":
            Ht, Wt = Hc, Wc  # 28x28 at 448 with patch16
        else:
            raise ValueError("target must be 'dino' or 'clip'")

        if (Hd, Wd) != (Ht, Wt):
            d_grid = F.interpolate(d_grid, size=(Ht, Wt), mode="bicubic", align_corners=False)
        if (Hc, Wc) != (Ht, Wt):
            c_grid = F.interpolate(c_grid, size=(Ht, Wt), mode="bicubic", align_corners=False)

        # Back to aligned tokens (same spatial index == same region)
        d_tok = _grid_to_tokens(d_grid)  # (B, P, F)
        c_tok = _grid_to_tokens(c_grid)  # (B, P, F)
        P = d_tok.shape[1]

        # Per-location fusion: for each position p, fuse [d_tok[p], c_tok[p]]
        streams = torch.stack([d_tok, c_tok], dim=2)  # (B, P, 2, F)
        x = streams.reshape(B * P, 2, -1)            # (B*P, 2, F)
        attn_out, _ = self.stream_attn(x, x, x, need_weights=False)
        x = self.stream_ln(x + attn_out)
        fused = x.mean(dim=1).reshape(B, P, -1)      # (B, P, F)

        # Spatial modeling
        cls = self.cls_token.expand(B, 1, -1)
        seq = torch.cat([cls, fused], dim=1)         # (B, 1+P, F)
        seq = self.spatial_encoder(seq)

        logits = self.head(self.norm(seq[:, 0, :]))  # (B, 1)
        return logits
    

class TransFusionNet(nn.Module):
    """
    Full Network Transformer fusion:

    | DINOV2 |                 | CLIP |
    |--------|   Stream-wise   |------|
    |        |   + Spatial     |      |
    |        |   Transformer   |      |
    |        |                 |      |
    """

    def __init__(self,
                 dim_dino=768,
                 dim_clip=768,
                 fusion_dim=512,
                 num_heads=8,
                 num_layers=4,
                 num_classes=1):
        super().__init__()
        self.clipDet = CLIPDetector(only_features=True)
        self.dinoDet = DINOv2Detector(only_features=True)

        for p in self.clipDet.parameters():
            p.requires_grad = False
        for p in self.dinoDet.parameters():
            p.requires_grad = False
        
        self.fusionNet = RegionAlignedFusionNet(
            dim_dino=dim_dino,
            dim_clip=dim_clip,
            fusion_dim=fusion_dim,
            stream_heads=num_heads,
            spatial_heads=num_heads,
            spatial_depth=num_layers,
            num_classes=num_classes,
            target="dino"
        )


    def forward(self, images):
        """
        Docstring for forward
        
        :param images: the (B, 3, 448, 448) input images
        """
        # Get tokens from both backbones
        dino_tokens, dino_hw = self.dinoDet(images, return_tokens=True)
        clip_tokens, clip_hw = self.clipDet(images, return_tokens=True)

        # Fuse and classify
        logits = self.fusionNet(dino_tokens, dino_hw, clip_tokens, clip_hw)

        return logits
                
        