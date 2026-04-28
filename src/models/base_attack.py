"""Base attack class and utilities."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


def compute_facelock_score(
    input1: torch.Tensor,
    input2: torch.Tensor,
    aligner: nn.Module,
    fr_model: nn.Module,
) -> torch.Tensor:
    """Compute cosine similarity between aligned faces."""
    # Move models to same device as input
    aligner = aligner.to(input1.device)
    fr_model = fr_model.to(input1.device)
    
    target_dtype = next(aligner.parameters()).dtype
    
    input1 = input1.to(dtype=target_dtype)
    input2 = input2.to(dtype=target_dtype)
    
    aligned_x1, _, aligned_ldmks1, _, _, _ = aligner(input1)
    aligned_x2, _, aligned_ldmks2, _, _, _ = aligner(input2)
    
    input_signature = inspect.signature(fr_model.model.net.forward)
    if input_signature.parameters.get('keypoints') is not None:
        feat1 = fr_model(aligned_x1, aligned_ldmks1)
        feat2 = fr_model(aligned_x2, aligned_ldmks2)
    else:
        feat1 = fr_model(aligned_x1)
        feat2 = fr_model(aligned_x2)
    
    return F.cosine_similarity(feat1, feat2)


def get_editshield_emb(
    img: torch.Tensor,
    vae: nn.Module,
    scheduler,
) -> torch.Tensor:
    """Get EditShield embedding from image."""
    dtype = vae.dtype
    img = img.to(dtype)
    
    dist = vae.encode(img).latent_dist
    raw_latents = dist.sample()
    
    latents_scaled = raw_latents * vae.config.scaling_factor
    
    noise = torch.randn_like(latents_scaled)
    bsz = latents_scaled.shape[0]
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (bsz,), device=latents_scaled.device
    ).long()
    noisy_latents = scheduler.add_noise(latents_scaled, noise, timesteps)
    
    cond_latents = vae.encode(img).latent_dist.sample()
    
    return torch.cat([noisy_latents, cond_latents], dim=1)


class BaseAttack(ABC):
    """Base class for adversarial attacks."""
    
    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        resources: dict,
        epsilon: float = 12 / 255.0,
        step_size: float = 0.02,
        iters: int = 40,
    ) -> torch.Tensor:
        """Apply attack to input image.
        
        Args:
            x: Clean image tensor in range [-1, 1]
            resources: Dictionary containing required models
            epsilon: Perturbation budget
            step_size: Optimization step size
            iters: Number of iterations
            
        Returns:
            Adversarial example
        """
        pass
    
    def _linear_step_decay(self, step_size: float, iters: int, i: int) -> float:
        """Decay step size linearly from step_size to step_size/100."""
        return step_size - (step_size - step_size / 100) / iters * i
    
    def _init_perturbation(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Initialize random perturbation within epsilon budget."""
        noise = torch.rand_like(x) * 2 * epsilon - epsilon
        return torch.clamp(x + noise, -1, 1)
    
    def _project(self, x_adv: torch.Tensor, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Project perturbation back to epsilon budget."""
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        return torch.clamp(x + delta, -1, 1)