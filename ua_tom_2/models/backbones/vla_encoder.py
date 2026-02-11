"""
Visual encoders for VLA integration.

Provides:
- CNNEncoder: Simple CNN for fast experiments
- ResNetEncoder: ResNet-18 backbone
- OpenVLAWrapper: Integration with OpenVLA (when available)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for torchvision
try:
    from torchvision.models import resnet18, ResNet18_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Check for OpenVLA
OPENVLA_AVAILABLE = False
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    OPENVLA_AVAILABLE = True
except ImportError:
    pass


class CNNEncoder(nn.Module):
    """
    Simple CNN encoder for 128x128 RGB images.
    Fast and lightweight for experiments.
    """
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.fc = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] or [B, T, C, H, W]
        Returns:
            [B, output_dim] or [B, T, output_dim]
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            feat = self.norm(self.fc(self.conv(x.reshape(B * T, C, H, W))))
            return feat.reshape(B, T, -1)
        return self.norm(self.fc(self.conv(x)))


class ResNetEncoder(nn.Module):
    """
    ResNet-18 encoder with optional freezing.
    """
    
    def __init__(self, output_dim: int = 256, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        
        if TORCHVISION_AVAILABLE:
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            
            self.fc = nn.Linear(512, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        else:
            # Fallback to CNN
            print("Warning: torchvision not available, using CNN fallback")
            self.backbone = None
            self.fallback = CNNEncoder(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return self.fallback(x)
        
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        
        # Resize to 224x224 for ResNet
        if x.shape[-1] != 224:
            x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        
        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(x).squeeze(-1).squeeze(-1)
        else:
            feat = self.backbone(x).squeeze(-1).squeeze(-1)
        
        feat = self.norm(self.fc(feat))
        
        if is_sequence:
            feat = feat.reshape(B, T, -1)
        
        return feat


class OpenVLAWrapper(nn.Module):
    """
    Wrapper for OpenVLA model.
    
    Falls back to simulated backbone if OpenVLA not available.
    Real OpenVLA requires ~16GB+ GPU memory.
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        freeze: bool = True,
        model_path: str = "openvla/openvla-7b",
        use_lite: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_openvla = False
        
        if use_lite or not OPENVLA_AVAILABLE:
            if not use_lite:
                print("OpenVLA not available, using simulated backbone")
            self.encoder = CNNEncoder(output_dim)
        else:
            self._load_openvla(model_path, output_dim, freeze)
    
    def _load_openvla(self, model_path: str, output_dim: int, freeze: bool):
        """Load actual OpenVLA model."""
        try:
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            hidden_size = self.vla.config.hidden_size
            self.proj = nn.Linear(hidden_size, output_dim)
            
            if freeze:
                for p in self.vla.parameters():
                    p.requires_grad = False
            
            self.use_openvla = True
            print(f"OpenVLA loaded (hidden_size={hidden_size})")
            
        except Exception as e:
            print(f"Failed to load OpenVLA: {e}")
            print("Falling back to simulated backbone")
            self.encoder = CNNEncoder(output_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        instruction: str = "Pick up the object",
    ) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] or [B, T, C, H, W]
            instruction: Task instruction for OpenVLA
        Returns:
            Features [B, output_dim] or [B, T, output_dim]
        """
        if self.use_openvla:
            return self._forward_openvla(images, instruction)
        return self.encoder(images)
    
    def _forward_openvla(self, images: torch.Tensor, instruction: str) -> torch.Tensor:
        """Forward through actual OpenVLA."""
        from PIL import Image
        import numpy as np
        
        is_sequence = images.dim() == 5
        if is_sequence:
            B, T, C, H, W = images.shape
            all_features = []
            for t in range(T):
                feat_t = self._process_single_frame(images[:, t], instruction)
                all_features.append(feat_t)
            return torch.stack(all_features, dim=1)
        
        return self._process_single_frame(images, instruction)
    
    def _process_single_frame(self, images: torch.Tensor, instruction: str) -> torch.Tensor:
        """Process single frame through OpenVLA."""
        from PIL import Image
        import numpy as np
        
        B = images.shape[0]
        device = images.device
        
        # Convert to PIL
        pil_images = []
        for i in range(B):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        # Process
        inputs = self.processor(
            text=[instruction] * B,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = self.vla(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            features = hidden_states.mean(dim=1)
        
        return self.proj(features)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scale


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, linear: nn.Linear, rank: int = 16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)
        
        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def get_encoder(
    encoder_type: str,
    output_dim: int = 256,
    training_mode: str = 'frozen',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create visual encoder.
    
    Args:
        encoder_type: 'cnn', 'resnet', 'openvla'
        output_dim: Output feature dimension
        training_mode: 'frozen', 'lora', 'full'
    """
    freeze = training_mode == 'frozen'
    
    if encoder_type == 'cnn':
        return CNNEncoder(output_dim)
    elif encoder_type == 'resnet':
        return ResNetEncoder(output_dim, freeze=freeze)
    elif encoder_type == 'openvla':
        return OpenVLAWrapper(
            output_dim,
            freeze=freeze,
            use_lite=kwargs.get('use_lite', True),
            model_path=kwargs.get('model_path', 'openvla/openvla-7b'),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
