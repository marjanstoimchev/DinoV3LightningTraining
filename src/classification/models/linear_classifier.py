#!/usr/bin/env python3
"""
Linear Classifier for downstream classification tasks.
Loads pretrained encoder from SSL checkpoint and adds classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add paths for DINOv3 modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'dinov3'))
sys.path.append(str(Path(__file__).parent.parent.parent))

from dinov3.models.vision_transformer import DinoVisionTransformer


class LinearClassifier(nn.Module):
    """
    Linear classifier on top of pretrained DINOv3 encoder.
    Can load pretrained weights from DINOv3 SSL checkpoint and optionally freeze the backbone.

    Args:
        num_classes: Number of output classes
        encoder_type: Which encoder to load from SSL checkpoint.
                     "teacher" (default, recommended) - EMA encoder, more stable
                     "student" - trained encoder
    """

    def __init__(self, num_classes, img_size=256, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4.0, num_storage_tokens=None,
                 rope_theta=100.0, drop_path_rate=0.0, init_values=1e-5,
                 pretrained_path=None, use_cls_token=True, encoder_type="teacher",
                 mask_k_bias=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type

        # Auto-detect architecture from checkpoint if not specified
        detected_storage_tokens = num_storage_tokens
        detected_mask_k_bias = mask_k_bias

        if pretrained_path and (num_storage_tokens is None or mask_k_bias is None):
            detected_storage_tokens, detected_mask_k_bias = self._detect_architecture(
                pretrained_path, encoder_type, num_storage_tokens, mask_k_bias
            )

        # Use defaults if still None (DINOv3 official defaults)
        if detected_storage_tokens is None:
            detected_storage_tokens = 4
        if detected_mask_k_bias is None:
            detected_mask_k_bias = True

        self._num_storage_tokens = detected_storage_tokens
        self._mask_k_bias = detected_mask_k_bias

        # Encoder (DINOv3 Vision Transformer)
        self.encoder = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            n_storage_tokens=detected_storage_tokens,
            layerscale_init=init_values,  # For ls1.gamma and ls2.gamma
            mask_k_bias=detected_mask_k_bias,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            drop_path_rate=drop_path_rate,
            pos_embed_rope_base=rope_theta,
        )

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        self._init_head()

    def _detect_architecture(self, path, encoder_type, num_storage_tokens, mask_k_bias):
        """Auto-detect architecture settings from checkpoint."""
        print(f"Auto-detecting architecture from checkpoint: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # Determine prefix based on checkpoint format
        if encoder_type == "teacher":
            prefixes = ["ssl_model.teacher.backbone._orig_mod.", "ssl_model.teacher.backbone.", ""]
        else:
            prefixes = ["ssl_model.student.backbone._orig_mod.", "ssl_model.student.backbone.", ""]

        detected_storage = num_storage_tokens
        detected_mask_bias = mask_k_bias

        # Try to find storage_tokens
        if detected_storage is None:
            for prefix in prefixes:
                key = f"{prefix}storage_tokens"
                if key in state:
                    detected_storage = state[key].shape[1]  # Shape is [1, n_tokens, embed_dim]
                    print(f"  Detected n_storage_tokens={detected_storage} from checkpoint")
                    break
            if detected_storage is None:
                # No storage_tokens found, assume 0
                detected_storage = 0
                print(f"  No storage_tokens found, using n_storage_tokens=0")

        # Try to find qkv.bias_mask to detect mask_k_bias
        if detected_mask_bias is None:
            for prefix in prefixes:
                key = f"{prefix}blocks.0.attn.qkv.bias_mask"
                if key in state:
                    detected_mask_bias = True
                    print(f"  Detected mask_k_bias=True from checkpoint")
                    break
            if detected_mask_bias is None:
                # No bias_mask found, assume False
                detected_mask_bias = False
                print(f"  No qkv.bias_mask found, using mask_k_bias=False")

        return detected_storage, detected_mask_bias

    def _init_head(self):
        """Initialize classification head"""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _load_pretrained(self, path):
        """Load pretrained weights from DINOv3 model checkpoint.

        Uses self.encoder_type to determine which encoder to load:
        - "teacher" (default): Load teacher backbone (EMA, recommended for evaluation)
        - "student": Load student backbone

        Handles torch.compile wrapped models (with ._orig_mod. prefix).
        """
        print(f"Loading pretrained weights from {path}")
        print(f"Encoder type: {self.encoder_type}")

        # Load checkpoint with weights_only=False to handle older checkpoints
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # Determine which encoder prefix to look for based on encoder_type
        if self.encoder_type == "teacher":
            primary_prefix = "ssl_model.teacher.backbone"
            fallback_prefix = "ssl_model.student.backbone"
        else:  # student
            primary_prefix = "ssl_model.student.backbone"
            fallback_prefix = "ssl_model.teacher.backbone"

        # Keys to exclude from backbone (heads only)
        exclude_prefixes = ['dino_head.', 'ibot_head.', 'gram_head.']

        def is_backbone_key(key):
            """Check if key is a backbone weight (not head)"""
            for prefix in exclude_prefixes:
                if key.startswith(prefix):
                    return False
            return True

        # Extract encoder weights with explicit preference
        encoder_state = {}
        found_primary = False
        found_fallback = False
        found_direct = False

        for key, value in state.items():
            new_key = None
            # Check for primary encoder type with torch.compile wrapper (._orig_mod.)
            if f"{primary_prefix}._orig_mod." in key:
                new_key = key.replace(f"{primary_prefix}._orig_mod.", "")
                found_primary = True
            # Check for primary encoder type without torch.compile wrapper
            elif f"{primary_prefix}." in key:
                new_key = key.replace(f"{primary_prefix}.", "")
                found_primary = True

            if new_key and is_backbone_key(new_key):
                encoder_state[new_key] = value

        # If primary not found, try fallback (other encoder type)
        if not found_primary:
            for key, value in state.items():
                new_key = None
                if f"{fallback_prefix}._orig_mod." in key:
                    new_key = key.replace(f"{fallback_prefix}._orig_mod.", "")
                    found_fallback = True
                elif f"{fallback_prefix}." in key:
                    new_key = key.replace(f"{fallback_prefix}.", "")
                    found_fallback = True

                if new_key and is_backbone_key(new_key):
                    encoder_state[new_key] = value

        # If still not found, try direct keys (official DINOv3 checkpoint format)
        if not found_primary and not found_fallback:
            # Check if this looks like an official checkpoint (has direct backbone keys)
            if 'cls_token' in state or 'patch_embed.proj.weight' in state:
                print("Detected official DINOv3 checkpoint format (direct keys)")
                for key, value in state.items():
                    if is_backbone_key(key):
                        encoder_state[key] = value
                        found_direct = True

        # Load encoder weights
        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=True)
            if found_primary:
                print(f"Loaded {primary_prefix} weights (primary choice)")
            elif found_fallback:
                print(f"Warning: {primary_prefix} not found, loaded {fallback_prefix} weights (fallback)")
            elif found_direct:
                print("Loaded official DINOv3 checkpoint weights (direct format)")
            else:
                print("Loaded generic encoder weights")
            print(f"Loaded {len(encoder_state)} encoder weights")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        else:
            print("Warning: No encoder weights found in checkpoint")

    def freeze_backbone(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Get features from encoder
        features = self.encoder(x, is_training=False)  # DINOv3 specific

        # Use CLS token for classification (features should be [CLS] token output)
        if self.use_cls_token:
            # The output of DINOv3 encoder should be the CLS token features
            if isinstance(features, (tuple, list)):
                # If it returns multiple values, take the first one (usually CLS tokens)
                features = features[0]

            # Ensure features are properly shaped
            if len(features.shape) > 2:
                # Take the CLS token (first token) if sequence dimension exists
                features = features[:, 0]  # Take [CLS] token

        # Classification head
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor):
        """Extract features without classification head"""
        features = self.encoder(x, is_training=False)
        if isinstance(features, (tuple, list)):
            features = features[0]

        if len(features.shape) > 2:
            features = features[:, 0]  # Take [CLS] token

        return features


def create_linear_classifier(num_classes, img_size=256, pretrained_path=None, use_cls_token=True, encoder_type="teacher"):
    """Create a linear classifier with DINOv3 backbone.

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        pretrained_path: Path to pretrained SSL checkpoint
        use_cls_token: Whether to use CLS token for classification (vs mean pooling)
        encoder_type: Which encoder to load - "teacher" (default, recommended) or "student"
    """
    return LinearClassifier(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_storage_tokens=4,
        pretrained_path=pretrained_path,
        use_cls_token=use_cls_token,
        encoder_type=encoder_type,
    )


# Backward compatibility alias
DINOv3LinearClassifier = LinearClassifier
create_dinov3_linear_classifier = create_linear_classifier
