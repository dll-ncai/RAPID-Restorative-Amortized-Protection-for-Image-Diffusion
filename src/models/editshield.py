"""EditShield attack implementation."""

import torch
import torch.optim as optim
from .base_attack import BaseAttack, get_editshield_emb


class EditShieldAttack(BaseAttack):
    """EditShield: Latent space diffusion attack.
    
    Uses DDPM scheduler to create target embeddings and maximizes
    distance between original and adversarial latent representations.
    """
    
    def __call__(
        self,
        x: torch.Tensor,
        resources: dict,
        epsilon: float = 12 / 255.0,
        step_size: float = 0.02,
        iters: int = 40,
    ) -> torch.Tensor:
        """Apply EditShield attack.
        
        Args:
            x: Clean image tensor [-1, 1], shape (B, C, H, W)
            resources: Dict with 'vae', 'ddpm_scheduler'
            epsilon: Perturbation budget
            step_size: Adam learning rate
            iters: Number of optimization iterations
        """
        vae = resources['vae']
        scheduler = resources['ddpm_scheduler']
        
        # Get target embedding
        with torch.no_grad():
            tgt_emb = get_editshield_emb(x, vae, scheduler)
        
        x_adv = self._init_perturbation(x, epsilon)
        
        optimizer = optim.Adam([x_adv], lr=step_size)
        
        for i in range(iters):
            x_adv.requires_grad = True
            
            img_emb = get_editshield_emb(x_adv, vae, scheduler)
            
            # Negative MSE = maximize distance
            loss_mse = -F.mse_loss(img_emb.float(), tgt_emb.float())
            loss_perceptual = F.mse_loss(x_adv, x) * 0.1
            total_loss = loss_mse + loss_perceptual
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x_adv.data = self._project(x_adv.data, x, epsilon)
        
        return x_adv


def editshield_attack(
    x: torch.Tensor,
    resources: dict,
    epsilon: float = 12 / 255.0,
    step_size: float = 0.02,
    iters: int = 40,
) -> torch.Tensor:
    """Convenience function for EditShield attack."""
    attack = EditShieldAttack()
    return attack(x, resources, epsilon, step_size, iters)