"""PhotoGuard (Encoder Attack) implementation."""

import torch
from .base_attack import BaseAttack


class PhotoGuardAttack(BaseAttack):
    """PhotoGuard: Encoder attack using VAE latent norm minimization.
    
    Uses gradient descent to maximize VAE latent norm, which disrupts
    the encoder's ability to reconstruct meaningful images.
    """
    
    def __call__(
        self,
        x: torch.Tensor,
        resources: dict,
        epsilon: float = 12 / 255.0,
        step_size: float = 0.02,
        iters: int = 40,
    ) -> torch.Tensor:
        """Apply PhotoGuard attack.
        
        Args:
            x: Clean image tensor [-1, 1], shape (B, C, H, W)
            resources: Dict with 'vae' key
            epsilon: Perturbation budget
            step_size: Optimization step size
            iters: Number of PGD iterations
        """
        vae = resources['vae']
        x_adv = self._init_perturbation(x, epsilon)
        
        for i in range(iters):
            x_adv.requires_grad = True
            
            actual_step_size = self._linear_step_decay(step_size, iters, i)
            
            latents = vae.encode(x_adv.to(vae.dtype)).latent_dist.mean
            loss = latents.norm()
            
            grad = torch.autograd.grad(loss, [x_adv], create_graph=False)[0]
            
            # Gradient descent (minimize loss = maximize disruption)
            x_adv = x_adv - grad.detach().sign() * actual_step_size
            
            x_adv = self._project(x_adv, x, epsilon).detach()
        
        return x_adv


def photoguard_attack(
    x: torch.Tensor,
    resources: dict,
    epsilon: float = 12 / 255.0,
    step_size: float = 0.02,
    iters: int = 40,
) -> torch.Tensor:
    """Convenience function for PhotoGuard attack."""
    attack = PhotoGuardAttack()
    return attack(x, resources, epsilon, step_size, iters)