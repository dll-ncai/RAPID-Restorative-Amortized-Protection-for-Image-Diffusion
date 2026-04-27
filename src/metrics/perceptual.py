"""Perceptual metrics: LPIPS, SSIM, PSNR."""

from typing import Optional
import numpy as np
import torch
import lpips

from ..utils.transforms import tensor_to_numpy


class PerceptualMetrics:
    """Wrapper for perceptual similarity metrics.
    
    Computes LPIPS, SSIM, and PSNR between image pairs.
    """
    
    def __init__(
        self,
        lpips_net: str = 'vgg',
        device: Optional[torch.device] = None,
    ):
        """Initialize perceptual metrics.
        
        Args:
            lpips_net: LPIPS network backbone ('vgg', 'alex', 'squeeze')
            device: Compute device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device).eval()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> float:
        """Compute Learned Perceptual Image Patch Similarity.
        
        Args:
            img1: Image tensor [-1, 1], shape (B, C, H, W)
            img2: Image tensor [-1, 1], shape (B, C, H, W)
            
        Returns:
            LPIPS score (lower = more similar)
        """
        return self.lpips_fn(img1, img2).mean().item()
    
    @torch.no_grad()
    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> float:
        """Compute Structural Similarity Index.
        
        Args:
            img1: Image tensor [-1, 1], shape (B, C, H, W)
            img2: Image tensor [-1, 1], shape (B, C, H, W)
            
        Returns:
            SSIM score (higher = more similar)
        """
        from skimage.metrics import structural_similarity as ssim
        
        img1_np = tensor_to_numpy(img1)
        img2_np = tensor_to_numpy(img2)
        
        scores = []
        for i in range(img1_np.shape[0]):
            score = ssim(img1_np[i], img2_np[i], channel_axis=-1, data_range=1.0)
            scores.append(score)
        
        return np.mean(scores)
    
    @torch.no_grad()
    def compute_psnr(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio.
        
        Args:
            img1: Image tensor [-1, 1], shape (B, C, H, W)
            img2: Image tensor [-1, 1], shape (B, C, H, W)
            
        Returns:
            PSNR in dB (higher = more similar)
        """
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        img1_np = tensor_to_numpy(img1)
        img2_np = tensor_to_numpy(img2)
        
        scores = []
        for i in range(img1_np.shape[0]):
            score = psnr(img1_np[i], img2_np[i], data_range=1.0)
            scores.append(score)
        
        return np.mean(scores)
    
    def compute_all(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> dict:
        """Compute all perceptual metrics.
        
        Args:
            img1: Image tensor [-1, 1]
            img2: Image tensor [-1, 1]
            
        Returns:
            Dict with 'lpips', 'ssim', 'psnr' keys
        """
        return {
            'lpips': self.compute_lpips(img1, img2),
            'ssim': self.compute_ssim(img1, img2),
            'psnr': self.compute_psnr(img1, img2),
        }


# Convenience functions
def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, model: Optional[PerceptualMetrics] = None) -> float:
    """Compute LPIPS score."""
    if model is None:
        model = PerceptualMetrics()
    return model.compute_lpips(img1, img2)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM score."""
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    metrics = PerceptualMetrics()
    return metrics.compute_ssim(img1, img2)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR score."""
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    metrics = PerceptualMetrics()
    return metrics.compute_psnr(img1, img2)