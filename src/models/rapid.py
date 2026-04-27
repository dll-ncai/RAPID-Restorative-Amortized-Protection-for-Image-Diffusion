"""RAPID (Restormer-based single-pass attack) implementation."""

import os
import torch
import torch.nn as nn

from .restormer import Restormer


class RAPID(nn.Module):
    """RAPID: Real-time Adversarial Protection for Image Diffusion.
    
    Single-pass Restormer-based perturbation generator.
    """
    
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: list = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = 'BiasFree',
    ):
        super().__init__()
        self.model = Restormer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
    
    def load_weights(self, weights_path: str, device: torch.device = None):
        """Load pretrained weights from checkpoint."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        if device is None:
            device = next(self.parameters()).device
            
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate perturbation.
        
        Args:
            x: Input tensor in range [-1, 1], shape (B, C, H, W)
            
        Returns:
            Perturbation tensor, shape (B, C, H, W)
        """
        return self.model(x)


def rapid_attack(
    x: torch.Tensor,
    model: RAPID,
    epsilon: float = 12 / 255.0,
) -> torch.Tensor:
    """Apply RAPID attack to generate adversarial example.
    
    Args:
        x: Clean image tensor in range [-1, 1], shape (B, C, H, W)
        model: Trained RAPID model
        epsilon: Perturbation budget (L-inf norm)
        
    Returns:
        Adversarial example in range [-1, 1]
    """
    with torch.no_grad():
        delta_raw = model(x)
        delta = torch.tanh(delta_raw) * epsilon
        x_adv = torch.clamp(x + delta, -1, 1)
    
    return x_adv