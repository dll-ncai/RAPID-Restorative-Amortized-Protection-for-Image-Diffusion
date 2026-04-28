"""RAPID Evaluation Pipeline - Main Entry Point."""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm

from src.config import EvalConfig
from src.models import (
    RAPID, rapid_attack,
    photoguard_attack,
    facelock_attack,
    editshield_attack,
)
from src.models.pipeline import ImageEditor
from src.metrics import PerceptualMetrics, CLIPScore
from src.data import create_eval_loader, get_default_transform


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    method: str
    image_idx: int
    prompt: str
    inference_time: float
    lpips: float
    ssim: float
    psnr: float
    clip_score: float


class Evaluator:
    """Main evaluation class for RAPID and competing methods."""
    
    def __init__(self, config: EvalConfig):
        """Initialize evaluator with config.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = config.device
        
        # Initialize components
        self._init_models()
        self._init_metrics()
        
        # Data
        self.data_loader = None
        self.prompts = config.prompts
    
    def _init_models(self):
        """Initialize all model components."""
        if self.config.verbose:
            print("Initializing models...")
        
        # RAPID model
        self.rapid_model = RAPID()
        if os.path.exists(self.config.rapid_weights_path):
            self.rapid_model.load_weights(self.config.rapid_weights_path, self.device)
            self.rapid_model.to(self.device)
            self.rapid_model.eval()
            if self.config.verbose:
                print("  ✓ RAPID model loaded")
        else:
            print(f"  ⚠ Warning: RAPID weights not found at {self.config.rapid_weights_path}")
        
        # Image editor (SD I2P)
        if self.config.verbose:
            print("  Loading Stable Diffusion InstructPix2Pix...")
        
        self.editor = ImageEditor(
            device=str(self.device),
            edit_steps=self.config.diffusion.edit_steps,
            guidance_scale=self.config.diffusion.guidance_scale,
            image_guidance_scale=self.config.diffusion.image_guidance_scale,
        )
        
        if self.config.verbose:
            print("  ✓ Image editor loaded")
        
        # Add DDPM scheduler for EditShield
        from diffusers import DDPMScheduler
        ddpm_scheduler = DDPMScheduler.from_pretrained(
            "/storage/2/models/diffusion/instruct-pix2pix",
            subfolder="scheduler"
        )
        
        # Resources dict for attacks
        self.resources = {
            'vae': self.editor.pipeline.vae,
            'ddpm_scheduler': ddpm_scheduler,
            'lpips_fn': PerceptualMetrics(device=self.device).lpips_fn,
        }
        
        # FaceLock models (optional - loaded on demand)
        self._face_models_loaded = False
        self._face_models = {}
    
    def _init_metrics(self):
        """Initialize metric calculators."""
        self.perceptual_metrics = PerceptualMetrics(device=self.device)
        self.clip_score = CLIPScore(device=self.device)
        
        if self.config.verbose:
            print("  ✓ Metrics initialized")
    
    def _load_face_models(self):
        """Load FaceLock models (aligner + FR) on demand."""
        if self._face_models_loaded:
            return
        
        # Import here to avoid loading unless needed
        from src.utils.downloads import download_face_models
        
        if self.config.verbose:
            print("  Loading FaceLock models...")
        
        try:
            self._face_models = download_face_models(
                aligner_path="/storage/2/models/facelock/aligner/aligner",
                fr_model_path="/storage/2/.cache/huggingface/fr_model",
            )
            self.resources['aligner'] = self._face_models['aligner']
            self.resources['fr_model'] = self._face_models['fr_model']
            self._face_models_loaded = True
            if self.config.verbose:
                print("  ✓ FaceLock models loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load FaceLock models: {e}")
    
    def _generate_attacks(self, x: torch.Tensor, method: str) -> torch.Tensor:
        """Generate adversarial examples for a method.
        
        Args:
            x: Clean image tensor [-1, 1]
            method: Attack method name
            
        Returns:
            Adversarial image tensor
        """
        epsilon = self.config.attack.epsilon
        step_size = self.config.attack.step_size
        iters = self.config.attack.num_iter
        
        start = time.perf_counter()
        
        if method == "rapid":
            x_adv = rapid_attack(x, self.rapid_model, epsilon)
        elif method == "photoguard":
            x_adv = photoguard_attack(x, self.resources, epsilon, step_size, iters)
        elif method == "facelock":
            self._load_face_models()
            x_adv = facelock_attack(x, self.resources, epsilon, step_size, iters)
        elif method == "editshield":
            x_adv = editshield_attack(x, self.resources, epsilon, step_size, iters)
        elif method == "clean":
            x_adv = x.clone()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return x_adv
    
    def _edit_image(self, x: torch.Tensor, prompt: str) -> torch.Tensor:
        """Edit image with SD I2P.
        
        Args:
            x: Image tensor [-1, 1]
            prompt: Text prompt
            
        Returns:
            Edited image tensor [-1, 1]
        """
        from src.utils.transforms import tensor_to_pil, pil_to_tensor
        
        pil_img = tensor_to_pil(x)
        edited_pil = self.editor.edit(pil_img, prompt)
        edited_tensor = pil_to_tensor(edited_pil, self.device)
        
        return edited_tensor
    
    def evaluate(
        self,
        data_loader,
        methods: list[str],
        num_images: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run full evaluation.
        
        Args:
            data_loader: DataLoader for images
            methods: List of methods to evaluate
            num_images: Limit number of images (None = all)
            
        Returns:
            DataFrame with results
        """
        results = []
        total_batches = len(data_loader)
        if num_images is not None:
            total_batches = min(total_batches, num_images)
        
        pbar = tqdm(total=total_batches, desc="Evaluating")
        
        for batch_idx, x in enumerate(data_loader):
            if num_images is not None and batch_idx >= num_images:
                break
            
            x = x.to(self.device)
            prompt = self.prompts[batch_idx % len(self.prompts)]
            
            # Generate attacks
            for method in methods:
                # Generate adversarial example
                start = time.perf_counter()
                x_adv = self._generate_attacks(x, method)
                inference_time = time.perf_counter() - start
                
                # Edit with SD I2P
                edited_adv = self._edit_image(x_adv, prompt)
                edited_clean = self._edit_image(x, prompt) if method != "clean" else None
                
                # Compute metrics
                if method == "clean":
                    # Clean has no perturbation
                    lpips = 0.0
                    ssim = 1.0
                    psnr = 100.0
                else:
                    metrics = self.perceptual_metrics.compute_all(edited_clean, edited_adv)
                    lpips = metrics['lpips']
                    ssim = metrics['ssim']
                    psnr = metrics['psnr']
                
                clip = self.clip_score.compute(edited_adv, prompt)
                
                result = EvaluationResult(
                    method=method,
                    image_idx=batch_idx,
                    prompt=prompt,
                    inference_time=inference_time,
                    lpips=lpips,
                    ssim=ssim,
                    psnr=psnr,
                    clip_score=clip,
                )
                results.append(result)
            
            pbar.update(1)
        
        pbar.close()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'method': r.method,
                'image_idx': r.image_idx,
                'prompt': r.prompt,
                'inference_time': r.inference_time,
                'LPIPS': r.lpips,
                'SSIM': r.ssim,
                'PSNR': r.psnr,
                'CLIP_Score': r.clip_score,
            }
            for r in results
        ])
        
        return df
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'editor'):
            self.editor.cleanup()
        torch.cuda.empty_cache()


def main(config: EvalConfig):
    """Main entry point.
    
    Args:
        config: Evaluation configuration
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Determine methods
    if "all" in config.methods:
        methods = ["clean", "photoguard", "facelock", "editshield", "rapid"]
    else:
        methods = config.methods
    
    if config.verbose:
        print(f"Evaluating methods: {methods}")
        print(f"Device: {config.device}")
        print(f"Images: {config.data.num_eval_images}")
        print(f"Epsilon: {config.attack.epsilon}")
        print(f"Iterations: {config.attack.num_iter}")
        print()
    
    # Create data loader
    data_loader = create_eval_loader(
        data_dir="data",
        num_images=config.data.num_eval_images,
        start_idx=config.data.start_index,
        batch_size=config.image.batch_size,
        size=config.image.size,
    )
    
    # Create evaluator and run
    evaluator = Evaluator(config)
    
    try:
        results_df = evaluator.evaluate(
            data_loader,
            methods=methods,
        )
        
        # Save results
        results_path = os.path.join(config.output_dir, "results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Print summary
        if config.verbose:
            print("\n" + "=" * 60)
            print("EVALUATION SUMMARY")
            print("=" * 60)
            
            summary = results_df.groupby('method').agg({
                'LPIPS': 'mean',
                'SSIM': 'mean',
                'PSNR': 'mean',
                'CLIP_Score': 'mean',
                'inference_time': 'mean',
            }).round(4)
            
            print(summary)
            print(f"\nResults saved to: {results_path}")
        
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    config = parse_args()
    main(config)