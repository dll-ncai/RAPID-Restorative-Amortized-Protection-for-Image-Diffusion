"""FaceLock attack implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attack import BaseAttack, compute_facelock_score


class FaceLockAttack(BaseAttack):
    """FaceLock: Learned attack using face recognition guidance.
    
    Uses face recognition model and aligner to maximize face verification
    loss, while also considering VAE reconstruction and LPIPS loss.
    """
    
    def __call__(
        self,
        x: torch.Tensor,
        resources: dict,
        epsilon: float = 12 / 255.0,
        step_size: float = 0.02,
        iters: int = 40,
    ) -> torch.Tensor:
        """Apply FaceLock attack.
        
        Args:
            x: Clean image tensor [-1, 1], shape (B, C, H, W)
            resources: Dict with 'vae', 'aligner', 'fr_model', 'lpips_fn'
            epsilon: Perturbation budget
            step_size: Optimization step size
            iters: Number of PGD iterations
        """
        vae = resources['vae']
        aligner = resources['aligner']
        fr_model = resources['fr_model']
        lpips_fn = resources['lpips_fn']
        
        x_adv = self._init_perturbation(x, epsilon)
        
        # Get reference latent
        with torch.no_grad():
            clean_latent = vae.encode(x.to(vae.dtype)).latent_dist.mean
        
        for i in range(iters):
            x_adv.requires_grad = True
            
            actual_step_size = self._linear_step_decay(step_size, iters, i)
            
            # Forward through VAE
            latent = vae.encode(x_adv.to(vae.dtype)).latent_dist.mean
            image_recon = vae.decode(latent).sample.clamp(-1, 1)
            
            # Loss components
            loss_cvl = compute_facelock_score(image_recon, x, aligner, fr_model)
            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image_recon.float(), x.float()).mean()
            
            # Compound loss with phase-based weighting
            loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + \
                   loss_encoder * 0.2 + \
                   loss_lpips * (1 if i > iters * 0.25 else 0.0)
            
            # Gradient ascent (maximize loss)
            grad = torch.autograd.grad(loss, [x_adv], create_graph=False)[0]
            
            x_adv = x_adv + grad.detach().sign() * actual_step_size
            
            x_adv = self._project(x_adv, x, epsilon).detach()
        
        return x_adv


def facelock_attack(
    x: torch.Tensor,
    resources: dict,
    epsilon: float = 12 / 255.0,
    step_size: float = 0.02,
    iters: int = 40,
) -> torch.Tensor:
    """Convenience function for FaceLock attack."""
    attack = FaceLockAttack()
    return attack(x, resources, epsilon, step_size, iters)