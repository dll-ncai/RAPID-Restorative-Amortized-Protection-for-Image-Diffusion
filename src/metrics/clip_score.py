"""CLIP-based image-text similarity metric."""

from typing import Optional
import torch
from PIL import Image
import os

from src.utils.transforms import tensor_to_pil


def load_clip_model(model_path: str, repo_id: str, device: torch.device):
    """Load CLIP model - supports local path or auto-download from HuggingFace.
    
    Args:
        model_path: Local path to CLIP model weights (.pt file)
        repo_id: HuggingFace model ID for auto-download
        device: Device to load on
        
    Returns:
        model, preprocess
    """
    # Try to import clip - prefer the package version
    try:
        import clip as clip_pkg
        use_transformers = False
    except ImportError:
        use_transformers = True
    
    if use_transformers:
        # Use transformers CLIP model (auto-download supported)
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(repo_id)
        processor = CLIPProcessor.from_pretrained(repo_id)
        model.to(device)
        model.eval()
        return model, processor, "transformers"
    else:
        # Use old clip package with local weights
        model, preprocess = clip_pkg.load(model_path, device)
        model.eval()
        return model, preprocess, "clip"


class CLIPScore:
    """CLIP-based image-text alignment metric.
    
    Computes cosine similarity between edited images and text prompts.
    Lower scores indicate better protection (model failed to follow prompt).
    """
    
    def __init__(
        self,
        model_path: str = "/storage/2/models/clip/ViT-L-14.pt",
        repo_id: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
    ):
        """Initialize CLIP model.
        
        Args:
            model_path: Path to CLIP model weights (local .pt file)
            repo_id: HuggingFace model ID for auto-download if local not found
            device: Compute device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Determine which model to load
        if os.path.exists(model_path):
            self.model, self.preprocess, self.backend = self._load_clip(model_path)
        else:
            # Auto-download from HuggingFace
            if self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(repo_id)
            self.model.to(self.device)
            self.model.eval()
            self.preprocess = CLIPProcessor.from_pretrained(repo_id)
            self.backend = "transformers"
    
    def _load_clip(self, model_path: str):
        """Load CLIP from local path."""
        try:
            import clip as clip_pkg
            model, preprocess = clip_pkg.load(model_path, self.device)
            model.eval()
            return model, preprocess, "clip"
        except ImportError:
            # Fall back to transformers
            from transformers import CLIPModel, CLIPProcessor
            # For .pt weights, we need to convert or use a different approach
            # Try loading as state dict
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'visual' in state_dict:
                # It's a CLIP state dict - convert to transformers format
                repo_id = "openai/clip-vit-large-patch14"
                model = CLIPModel.from_pretrained(repo_id)
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()
                preprocess = CLIPProcessor.from_pretrained(repo_id)
                return model, preprocess, "transformers"
            else:
                # Just use transformers default
                repo_id = "openai/clip-vit-large-patch14"
                model = CLIPModel.from_pretrained(repo_id)
                model.to(self.device)
                model.eval()
                preprocess = CLIPProcessor.from_pretrained(repo_id)
                return model, preprocess, "transformers"
    
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
        pil_img = tensor_to_pil(image)
        
        if self.backend == "clip":
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)
            
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        else:
            # Transformers CLIP
            inputs = self.preprocess(text=prompt, images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
        
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
        
        if self.backend == "clip":
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
        else:
            inputs = self.preprocess(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
        
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
        if self.backend == "clip":
            text_input = clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(text_input)
        else:
            inputs = self.preprocess(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
        
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