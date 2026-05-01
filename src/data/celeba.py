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
    Supports both local files and auto-download from HuggingFace.
    """
    
    def __init__(
        self,
        data_dir: str = "celeba-hq",
        file_name: str = "dataset.parquet",
        repo_id: Optional[str] = None,
        start_idx: int = 5000,
        num_images: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Initialize CelebA dataset.
        
        Args:
            data_dir: Directory containing CelebA-HQ parquet file
            file_name: Name of the parquet file
            repo_id: HuggingFace repo ID for auto-download (optional)
            start_idx: Starting index (after training split)
            num_images: Number of images to load
            transform: Optional transforms to apply
            download: Whether to download if not found
        """
        self.data_dir = data_dir
        self.file_name = file_name
        self.start_idx = start_idx
        self.num_images = num_images
        self.transform = transform
        self.repo_id = repo_id
        
        # Load parquet file
        parquet_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(parquet_path):
            if download and repo_id:
                # Auto-download from HuggingFace
                self._download_from_huggingface(repo_id, data_dir)
            else:
                raise FileNotFoundError(
                    f"CelebA-HQ dataset not found at {parquet_path}. "
                    f"Use --dataset-path to specify local path or repo_id for auto-download."
                )
        
        self.df = pd.read_parquet(parquet_path)
    
    def _download_from_huggingface(self, repo_id: str, save_dir: str):
        """Download dataset from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            save_dir: Local directory to save to
        """
        from huggingface_hub import hf_hub_download
        
        os.makedirs(save_dir, exist_ok=True)
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=self.file_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        
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
    file_name: str = "dataset.parquet",
    repo_id: Optional[str] = None,
    num_images: int = 100,
    start_idx: int = 4000,
    batch_size: int = 1,
    size: int = 512,
    download: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for evaluation.
    
    Args:
        data_dir: Data directory
        file_name: Name of the parquet file
        repo_id: HuggingFace repo ID for auto-download (optional)
        num_images: Number of images to evaluate
        start_idx: Starting index
        batch_size: Batch size
        size: Image size
        download: Whether to download if not found
        
    Returns:
        DataLoader for evaluation
    """
    transform = get_default_transform(size)
    
    dataset = CelebAEvalDataset(
        data_dir=data_dir,
        file_name=file_name,
        repo_id=repo_id,
        start_idx=start_idx,
        num_images=num_images,
        transform=transform,
        download=download,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Using image bytes, no need for workers
        pin_memory=True,
    )