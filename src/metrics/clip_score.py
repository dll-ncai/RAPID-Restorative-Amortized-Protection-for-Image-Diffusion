"""CLIP-based image-text similarity metric."""

from typing import Optional
import torch
from PIL import Image

import clip

from ..utils.transforms import tensor_to_pil


class CLIPScore:
    """CLIP-based image-text alignment metric.
    
    Computes cosine similarity between edited images and text prompts.
    Lower scores indicate better protection (model failed to follow prompt).
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-L/14',
        device: Optional[torch.device] = None,
    ):
        """Initialize CLIP model.
        
        Args:
            model_name: CLIP model variant
            device: Compute device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model, self.preprocess = clip.load(model_name, device)
        self.model.eval()
    
    @torch.no_grad()
    def compute(
        self,
        image: torch.Tensor,
        prompt: str,
    ) -> float:
        """Compute cosine similarity between image and text prompt.
        
        Args:
            image: Image tensor [-1, 1], shape (B, C, H, W)
            prompt: Text prompt
            
        Returns:
            Cosine similarity score (higher = better alignment)
        """
        # Convert tensor to PIL
        pil_img = tensor_to_pil(image)
        
        # Preprocess for CLIP
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # Tokenize text
        text_input = clip.tokenize([prompt]).to(self.device)
        
        # Compute features
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        # Normalize and compute similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    @torch.no_grad()
    def compute_batch(
        self,
        images: list[torch.Tensor],
        prompts: list[str],
    ) -> list[float]:
        """Compute CLIP scores for batch of images and prompts.
        
        Args:
            images: List of image tensors [-1, 1]
            prompts: List of text prompts
            
        Returns:
            List of similarity scores
        """
        scores = []
        for img, prompt in zip(images, prompts):
            score = self.compute(img, prompt)
            scores.append(score)
        return scores
    
    @torch.no_grad()
    def compute_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features.
        
        Args:
            image: Image tensor [-1, 1]
            
        Returns:
            Normalized feature tensor
        """
        pil_img = tensor_to_pil(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    @torch.no_grad()
    def compute_text_features(self, text: str) -> torch.Tensor:
        """Extract CLIP text features.
        
        Args:
            text: Text string
            
        Returns:
            Normalized feature tensor
        """
        text_input = clip.tokenize([text]).to(self.device)
        
        features = self.model.encode_text(text_input)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features


def compute_clip_score(
    image: torch.Tensor,
    prompt: str,
    clip_model: Optional[CLIPScore] = None,
) -> float:
    """Convenience function to compute CLIP score."""
    if clip_model is None:
        clip_model = CLIPScore()
    return clip_model.compute(image, prompt)