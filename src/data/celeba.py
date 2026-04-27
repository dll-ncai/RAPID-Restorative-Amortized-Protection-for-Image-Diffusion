"""CelebA-HQ dataset loader for evaluation."""

import os
import io
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CelebAEvalDataset(Dataset):
    """CelebA-HQ evaluation dataset.
    
    Loads images from the CelebA-HQ parquet dataset, supporting configurable
    start index and number of images (for train/test split).
    """
    
    def __init__(
        self,
        data_dir: str = "celeba-hq",
        start_idx: int = 5000,
        num_images: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Initialize CelebA dataset.
        
        Args:
            data_dir: Directory containing CelebA-HQ parquet file
            start_idx: Starting index (after training split)
            num_images: Number of images to load
            transform: Optional transforms to apply
            download: Whether to download if not found (not supported for parquet)
        """
        self.data_dir = data_dir
        self.start_idx = start_idx
        self.num_images = num_images
        self.transform = transform
        
        # Load parquet file
        parquet_path = os.path.join(data_dir, "dataset.parquet")
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"CelebA-HQ dataset not found at {parquet_path}"
            )
        
        self.df = pd.read_parquet(parquet_path)
        
        # Select range
        self.df = self.df.iloc[start_idx:start_idx + num_images]
        
        if len(self.df) == 0:
            raise ValueError(
                f"No images found at index {start_idx}. "
                f"Dataset may have fewer than {start_idx + num_images} images."
            )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return an image.
        
        Args:
            idx: Image index
            
        Returns:
            Transformed image tensor
        """
        row = self.df.iloc[idx]
        image_bytes = row['image']['bytes']
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image


def get_default_transform(size: int = 512) -> T.Compose:
    """Get default transform for CelebA images.
    
    Args:
        size: Target image size (default 512 for SD I2P)
        
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # Range [-1, 1]
    ])


def create_eval_loader(
    data_dir: str = "celeba-hq",
    num_images: int = 100,
    start_idx: int = 4000,
    batch_size: int = 1,
    size: int = 512,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for evaluation.
    
    Args:
        data_dir: Data directory
        num_images: Number of images to evaluate
        start_idx: Starting index
        batch_size: Batch size
        size: Image size
        
    Returns:
        DataLoader for evaluation
    """
    transform = get_default_transform(size)
    
    dataset = CelebAEvalDataset(
        data_dir=data_dir,
        start_idx=start_idx,
        num_images=num_images,
        transform=transform,
        download=False,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Using image bytes, no need for workers
        pin_memory=True,
    )