"""CelebA-HQ dataset loader for evaluation."""

import os
import zipfile
from typing import Optional, Callable
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CelebAEvalDataset(Dataset):
    """CelebA-HQ evaluation dataset.
    
    Loads images from the CelebA-HQ dataset, supporting configurable
    start index and number of images (for train/test split).
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        start_idx: int = 5000,
        num_images: int = 100,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """Initialize CelebA dataset.
        
        Args:
            data_dir: Directory containing CelebA data
            start_idx: Starting index (after training split)
            num_images: Number of images to load
            transform: Optional transforms to apply
            download: Whether to download if not found
        """
        self.data_dir = data_dir
        self.start_idx = start_idx
        self.num_images = num_images
        self.transform = transform
        
        # Find image directory
        self.image_dir = self._find_image_dir()
        
        if self.image_dir is None:
            if download:
                self._download()
                self.image_dir = self._find_image_dir()
            else:
                raise FileNotFoundError(
                    f"CelebA images not found in {data_dir}. "
                    "Set download=True to download."
                )
        
        # Get sorted list of image files
        self.image_files = sorted(os.listdir(self.image_dir))
        
        # Select range after training split
        self.image_files = self.image_files[start_idx:start_idx + num_images]
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found at index {start_idx}. "
                f"Dataset may have fewer than {start_idx + num_images} images."
            )
    
    def _find_image_dir(self) -> Optional[str]:
        """Find the images directory in data_dir."""
        possible_dirs = [
            os.path.join(self.data_dir, "celeba", "images"),
            os.path.join(self.data_dir, "celeba-hq", "images"),
            os.path.join(self.data_dir, "images"),
        ]
        
        for d in possible_dirs:
            if os.path.isdir(d):
                return d
        return None
    
    def _download(self, url: str = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1") -> None:
        """Download and extract CelebA dataset."""
        print(f"Downloading CelebA from {url}...")
        
        zip_path = os.path.join(self.data_dir, "celeba.zip")
        
        # Download with progress
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            print(f"\rDownloaded: {percent}%", end="", flush=True)
        
        urlretrieve(url, zip_path, reporthook=progress)
        print("\nDownload complete. Extracting...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Clean up zip
        os.remove(zip_path)
        print("Extraction complete.")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return an image.
        
        Args:
            idx: Image index
            
        Returns:
            Transformed image tensor
        """
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
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
    data_dir: str = "data",
    num_images: int = 100,
    start_idx: int = 5000,
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
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )