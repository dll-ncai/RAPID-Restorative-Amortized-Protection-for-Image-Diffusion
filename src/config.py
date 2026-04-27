import os
import random
import argparse
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


def setup_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AttackConfig:
    """Attack hyperparameters."""
    epsilon: float = 12 / 255.0
    num_iter: int = 40
    step_size: float = 0.02


@dataclass
class ImageConfig:
    """Image processing configuration."""
    size: int = 512
    batch_size: int = 1


@dataclass
class DataConfig:
    """Dataset configuration."""
    num_eval_images: int = 100
    start_index: int = 4000
    url: str = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1"


@dataclass
class DiffusionConfig:
    """Stable Diffusion InstructPix2Pix configuration."""
    edit_steps: int = 50
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5


DEFAULT_PROMPTS: List[str] = [
    "Turn the person's hair pink",
    "Let the person turn bald",
    "Let the person have a tattoo",
    "Let the person wear purple makeup",
    "Let the person grow a moustache",
    "Turn the person into a zombie",
    "Change the skin color to Avatar blue",
    "Add elf-like ears",
    "Let the person wear sunglasses",
    "Let the person wear a police suit",
]


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    seed: int = 42
    device: torch.device = field(default_factory=get_device)

    attack: AttackConfig = field(default_factory=AttackConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS)

    rapid_weights_path: str = "restormer.pth"
    restormer_repo_path: str = "Restormer"
    output_dir: str = "results"

    methods: List[str] = field(default_factory=lambda: ["all"])

    save_images: bool = False
    verbose: bool = True

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        setup_seed(self.seed)


def parse_args() -> EvalConfig:
    """Parse command line arguments into EvalConfig."""
    parser = argparse.ArgumentParser(
        description="RAPID Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-images", type=int, default=100,
        help="Number of test images to evaluate",
    )
    parser.add_argument(
        "--start-idx", type=int, default=5000,
        help="Starting index in dataset (after training split)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help="Methods to evaluate: photoguard, facelock, editshield, rapid, or all",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save visualization samples",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--epsilon", type=float, default=12 / 255.0,
        help="Perturbation budget (L-inf)",
    )
    parser.add_argument(
        "--iterations", type=int, default=40,
        help="Number of attack iterations",
    )

    args = parser.parse_args()

    config = EvalConfig(
        seed=42,
        device=torch.device(args.device),
        data=DataConfig(
            num_eval_images=args.num_images,
            start_index=args.start_idx,
        ),
        attack=AttackConfig(
            epsilon=args.epsilon,
            num_iter=args.iterations,
        ),
        output_dir=args.output_dir,
        methods=args.methods,
        save_images=args.save_images,
        verbose=not args.no_verbose,
    )

    return config


# Backwards compatibility - module-level constants
DEVICE = get_device()
SEED = 42
EPSILON = 12 / 255.0
NUM_ITER = 40
STEP_SIZE = 0.02
IMAGE_SIZE = 512
BATCH_SIZE = 1
NUM_EVAL_IMAGES = 100
START_INDEX = 4000
PROMPTS = DEFAULT_PROMPTS
EDIT_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5

setup_seed(SEED)