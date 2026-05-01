"""Stable Diffusion InstructPix2Pix pipeline wrapper."""

import os
from typing import Optional
from PIL import Image

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler


class ImageEditor:
    """Wrapper for Stable Diffusion InstructPix2Pix.
    
    Provides simple interface for text-guided image editing.
    Supports both local paths and auto-download from HuggingFace.
    """
    
    def __init__(
        self,
        model_path: str = "/storage/2/models/diffusion/instruct-pix2pix",
        repo_id: str = "diffusers/instruct-pix2pix-78",
        device: str = "cuda",
        edit_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
    ):
        """Initialize the image editor.
        
        Args:
            model_path: Local path to InstructPix2Pix model OR HuggingFace repo ID
            repo_id: HuggingFace model ID for auto-download if local path doesn't exist
            device: Device to load models on
            edit_steps: Number of denoising steps
            guidance_scale: Text guidance scale
            image_guidance_scale: Image guidance scale
        """
        self.device = device
        self.edit_steps = edit_steps
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.model_path = model_path
        self.repo_id = repo_id
        
        # Determine source: local path or auto-download
        if os.path.exists(model_path):
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        else:
            # Auto-download from HuggingFace
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        
        # Use Euler Ancestral scheduler for faster sampling
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
        
        self.pipeline = pipeline.to(device)
        
        # Set components to eval mode
        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
        
        # Enable memory optimizations
        if device == "cuda":
            self.pipeline.enable_attention_slicing()
    
    @torch.no_grad()
    def edit(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Image.Image:
        """Edit an image based on a text prompt.
        
        Args:
            image: Input PIL Image
            prompt: Text instruction for editing
            
        Returns:
            Edited PIL Image
        """
        result = self.pipeline(
            prompt,
            image=image,
            num_inference_steps=self.edit_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
        )
        
        return result.images[0]
    
    @torch.no_grad()
    def edit_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[Image.Image]:
        """Edit multiple images with corresponding prompts.
        
        Args:
            images: List of input PIL Images
            prompts: List of text instructions
            
        Returns:
            List of edited PIL Images
        """
        results = []
        for image, prompt in zip(images, prompts):
            edited = self.edit(image, prompt)
            results.append(edited)
        return results
    
    def to(self, device: str):
        """Move pipeline to device."""
        self.device = device
        self.pipeline.to(device)
        return self
    
    def cleanup(self):
        """Clean up GPU memory."""
        del self.pipeline
        torch.cuda.empty_cache()