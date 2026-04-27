"""Image processing utilities and tensor transformations."""

from typing import Optional
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] tensor to [0, 1] numpy array (H, W, C).
    
    Args:
        tensor: Image tensor in range [-1, 1], shape (B, C, H, W)
        
    Returns:
        Numpy array in range [0, 1], shape (B, H, W, C)
    """
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    img = img.permute(0, 2, 3, 1).numpy()
    return img


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [-1, 1] tensor to PIL Image.
    
    Args:
        tensor: Image tensor in range [-1, 1], shape (B, C, H, W)
        
    Returns:
        PIL Image (from first image in batch)
    """
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    return T.ToPILImage()(img[0])


def pil_to_tensor(pil_image: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert PIL Image to [-1, 1] tensor.
    
    Args:
        pil_image: PIL Image
        device: Target device
        
    Returns:
        Image tensor in range [-1, 1], shape (1, C, H, W)
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    tensor = transform(pil_image).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def numpy_to_tensor(arr: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert [0, 1] numpy array to [-1, 1] tensor.
    
    Args:
        arr: Numpy array in range [0, 1], shape (H, W, C) or (B, H, W, C)
        device: Target device
        
    Returns:
        Image tensor in range [-1, 1]
    """
    tensor = torch.from_numpy(arr).float()
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor * 2 - 1
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_image_transform(size: int = 512) -> T.Compose:
    """Get standard image transformation pipeline.
    
    Args:
        size: Target image size (default 512 for SD I2P)
        
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])